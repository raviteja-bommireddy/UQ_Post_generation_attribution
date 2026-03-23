"""
pipeline_utils.py
=================
Shared utilities for the UQ pipeline.

Provides:
  - LLM client  : generate_answer, extract_atomic_claims, generate_all_perturbations
  - BM25 index  : build_bm25_index, bm25_retrieve
  - NLI model   : load_nli_model, nli_score_claim

All other modules import from here. Do NOT duplicate model loading elsewhere.
"""

import os
import time
import logging
import getpass
from pathlib import Path

import numpy as np
from openai import OpenAI

log = logging.getLogger("uq_pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# 1. API KEY
# ─────────────────────────────────────────────────────────────────────────────

def load_api_key() -> str:
    """
    Load OpenRouter API key from .env, environment variable, or interactive prompt.
    Call this once at the start of main.py.
    """
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key

    # Interactive fallback
    print("OPENROUTER_API_KEY not found.")
    key = getpass.getpass("Paste your OpenRouter API key: ").strip()
    save = input("Save to .env? (y/n): ").strip().lower()
    if save == "y":
        Path(".env").write_text(f"OPENROUTER_API_KEY={key}\n")
        print("Saved to .env")
    return key


# ─────────────────────────────────────────────────────────────────────────────
# 2. LLM CLIENT
# ─────────────────────────────────────────────────────────────────────────────

def build_llm_client(api_key: str, model: str = "meta-llama/llama-3.1-70b-instruct") -> dict:
    """
    Build and return a dict containing the OpenAI client + model name.
    Using a dict so config is explicit when passed between functions.
    """
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    return {"client": client, "model": model}


def _llm_call(llm: dict, messages: list, max_tokens: int = 512,
              temperature: float = 0.0, seed: int = 42,
              max_retries: int = 6) -> str:
    """
    Internal helper: single LLM call with exponential-backoff retry on 429/500/503.
    temperature=0.0 for deterministic outputs; set >0 for multi-run claim extraction.
    """
    wait = 60
    for attempt in range(1, max_retries + 1):
        try:
            resp = llm["client"].chat.completions.create(
                model=llm["model"],
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            code = getattr(getattr(e, "response", None), "status_code", 0)
            if code in (429, 500, 503) and attempt < max_retries:
                log.warning(f"API retry {attempt}/{max_retries} [{code}] in {wait}s")
                time.sleep(wait)
                wait = min(wait * 2, 300)
            else:
                log.error(f"API error: {e}")
                return ""


def generate_answer(llm: dict, question: str) -> str:
    """
    Step 3: Generate a factual multi-sentence answer for one ASQA question.
    Temperature=0 for reproducibility.
    """
    return _llm_call(llm, [
        {"role": "system", "content": "You are a knowledgeable assistant. Answer factually and completely."},
        {"role": "user",   "content": f"Answer the following question thoroughly:\n\n{question}"},
    ], max_tokens=512, temperature=0.0)


def extract_atomic_claims(llm: dict, answer: str, temperature: float = 0.0) -> list:
    """
    Step 4: Decompose answer into atomic, independently verifiable claims.
    Each claim = exactly one verifiable fact.
    temperature=0.0 for main run; use >0 for UQ4 multi-run consistency check.
    """
    prompt = (
        "Break the following answer into atomic, independently verifiable factual claims.\n"
        "Rules:\n"
        "- One fact per claim. No compound sentences.\n"
        "- No opinions, meta-statements, or question restatements.\n"
        "- Output ONLY the claims, one per line. No numbers, no bullets.\n\n"
        f"Answer:\n{answer}"
    )
    raw = _llm_call(llm, [{"role": "user", "content": prompt}],
                    max_tokens=512, temperature=temperature)
    return [line.strip() for line in raw.splitlines() if len(line.strip()) > 15]


# ─────────────────────────────────────────────────────────────────────────────
# 3. PERTURBATION GENERATION
# ─────────────────────────────────────────────────────────────────────────────

# Prompt templates for each of the 5 perturbation categories.
# Category → maps to UQ stage it is designed to stress-test.
PERT_PROMPTS = {
    # ── C1: Preserve attribution (→ UQ₁ Attribution) ──
    "C1_paraphrase": (
        "Rephrase the claim using different words. Preserve all entities, dates, and quantifiers exactly. "
        "Return ONLY the rephrased claim.\n\nClaim: {claim}"
    ),
    "C1_voice_change": (
        "Rewrite the claim by changing its grammatical voice (active↔passive). "
        "Keep all facts identical. Return ONLY the rewritten claim.\n\nClaim: {claim}"
    ),
    # ── C2: Destroy attribution (→ UQ₁ Attribution) ──
    "C2_negation": (
        "Negate the main predicate of the claim. Keep all other parts unchanged. "
        "Return ONLY the negated claim.\n\nClaim: {claim}"
    ),
    "C2_factual_inversion": (
        "Replace the key fact (e.g., reason, outcome) with a plausible but wrong alternative. "
        "Keep entities and dates. Return ONLY the modified claim.\n\nClaim: {claim}"
    ),
    "C2_year_shift": (
        "Change the year or number in the claim to a plausible but incorrect value. "
        "Return ONLY the modified claim.\n\nClaim: {claim}"
    ),
    "C2_entity_swap_destroy": (
        "Replace the main named entity (person, place, or organisation) with a different but plausible "
        "entity of the same type such that the claim becomes factually wrong. "
        "Return ONLY the modified claim.\n\nClaim: {claim}"
    ),
    # ── C3: Specificity / Retrieval probe (→ UQ₂ Retrieval) ──
    "C3_entity_swap_probe": (
        "Replace the main named entity with a different plausible entity of the same type. "
        "The claim should remain internally coherent. "
        "Return ONLY the modified claim.\n\nClaim: {claim}"
    ),
    "C3_category_swap": (
        "Replace the category or domain noun (e.g., Nobel Prize in Physics → in Chemistry) "
        "with a plausible alternative. Return ONLY the modified claim.\n\nClaim: {claim}"
    ),
    # ── C4: Breadth / Abstraction (→ UQ₃ Hallucination Content) ──
    "C4_category_generalize": (
        "Generalize the claim by replacing specific entities with broader category descriptions. "
        "Keep the core proposition. Return ONLY the generalized claim.\n\nClaim: {claim}"
    ),
    "C4_temporal_abstraction": (
        "Replace the specific year or date with a broader time reference (e.g., 'in the 1920s', "
        "'in the early 20th century'). Return ONLY the modified claim.\n\nClaim: {claim}"
    ),
    # ── C5: Depth / Specification (→ UQ₃ Hallucination Content) ──
    "C5_causal_detail": (
        "Add one specific, plausible causal detail explaining WHY or HOW the event in the claim occurred. "
        "Return ONLY the extended claim.\n\nClaim: {claim}"
    ),
    "C5_consequence_detail": (
        "Add one specific, plausible consequence or outcome detail to the claim. "
        "Return ONLY the extended claim.\n\nClaim: {claim}"
    ),
}

# Ordered list of all perturbation keys — defines the canonical order throughout the pipeline.
ALL_PERT_KEYS = list(PERT_PROMPTS.keys())  # 12 keys total

# Keys for C5 (depth/specification) — used by uq_content.py for Specification Entailment Gap.
C5_KEYS = ["C5_causal_detail", "C5_consequence_detail"]


def generate_all_perturbations(llm: dict, claim: str) -> dict:
    """
    Step 5: Generate all 12 perturbations for a single atomic claim.

    Returns:
        dict mapping pert_key → perturbed claim text
        e.g. {"C1_paraphrase": "...", "C2_negation": "...", ...}
    """
    result = {}
    for key, prompt_template in PERT_PROMPTS.items():
        prompt = prompt_template.format(claim=claim)
        text = _llm_call(llm, [{"role": "user", "content": prompt}],
                          max_tokens=200, temperature=0.0)
        result[key] = text if text else claim  # fallback to original if empty
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. BM25 INDEX
# ─────────────────────────────────────────────────────────────────────────────

def build_bm25_index(num_questions: int = 17):
    """
    Step 2: Load ASQA dataset, extract corpus passages, build BM25 index.
    Also returns the list of questions.

    Returns:
        questions     : list of question strings
        bm25          : BM25Okapi index object
        corpus_texts  : list of passage strings (parallel to bm25)
        corpus_ids    : list of passage ID strings (parallel to bm25)
    """
    from datasets import load_dataset
    from rank_bm25 import BM25Okapi

    print("Loading ASQA dataset...")
    asqa = load_dataset("din0s/asqa", split="train", trust_remote_code=True)
    questions = [
        asqa[i]["ambiguous_question"]
        for i in range(min(num_questions, len(asqa)))
    ]
    print(f"  {len(questions)} questions loaded")

    corpus_texts, corpus_ids = [], []
    n_rows = min(len(asqa), num_questions * 5)

    for i in range(n_rows):
        item = asqa[i]
        for qa in item.get("qa_pairs", []):
            for key in ("context", "long_answer"):
                text = (qa.get(key) or "").strip()
                if len(text) > 30:
                    corpus_texts.append(text)
                    corpus_ids.append(f"asqa_{i}_qa_{len(corpus_texts)}")
                    break
        for ann in item.get("annotations", []):
            for page in ann.get("knowledge", []):
                for sent in page.get("content", [])[:5]:
                    sent = sent.strip()
                    if len(sent) > 30:
                        corpus_texts.append(sent)
                        corpus_ids.append(f"wiki_{i}_{len(corpus_texts)}")

    # Optional: ALCE Wikipedia extension
    try:
        print("Loading ALCE Wikipedia corpus...")
        wiki = load_dataset("princeton-nlp/sup-simcse-wiki",
                            split="train[:5000]", trust_remote_code=True)
        for j, ex in enumerate(wiki):
            text = ((ex.get("sent") or "") or (ex.get("text") or "")).strip()
            if len(text) > 30:
                corpus_texts.append(text)
                corpus_ids.append(f"alce_{j}")
        print(f"  Added {len(wiki):,} ALCE passages")
    except Exception as e:
        print(f"  ALCE skipped: {e}")

    print(f"  Total corpus: {len(corpus_texts):,} passages — building BM25...")
    tokenized = [doc.lower().split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized)
    print("  BM25 index ready")
    return questions, bm25, corpus_texts, corpus_ids


def bm25_retrieve(query: str, bm25, corpus_texts: list, corpus_ids: list,
                  top_k: int = 5) -> list:
    """
    Retrieve top_k passages for a query using the BM25 index.

    Returns:
        list of dicts: [{"id": str, "text": str, "score": float}, ...]
    """
    scores  = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [
        {"id": corpus_ids[i], "text": corpus_texts[i], "score": float(scores[i])}
        for i in top_idx
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 5. NLI MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_nli_model(model_name: str = "cross-encoder/nli-deberta-v3-base"):
    """
    Load the NLI cross-encoder model.
    Call once; reuse the returned object for all scoring.

    Label order for deberta-v3-base: [contradiction=0, entailment=1, neutral=2]
    """
    from sentence_transformers import CrossEncoder
    print(f"Loading NLI model: {model_name} ...")
    model = CrossEncoder(model_name, max_length=512)
    print("  NLI model ready")
    return model


def nli_score_claim(claim: str, docs: list, nli_model) -> dict:
    """
    Step 6 (NLI): Compute entailment probability for each (doc, claim) pair.

    Args:
        claim     : the claim text (base or perturbed)
        docs      : list of {"id", "text", "score"} dicts from bm25_retrieve
        nli_model : loaded CrossEncoder

    Returns:
        dict:
            per_doc_entailment : list of float  — one per doc
            max_score          : float          — max entailment (= attribution score)
            mean_score         : float
            weighted_score     : float          — BM25-rank-weighted mean
    """
    if not docs:
        return {"per_doc_entailment": [], "max_score": 0.0,
                "mean_score": 0.0, "weighted_score": 0.0}

    # Cross-encoder input: (premise=doc, hypothesis=claim)
    pairs  = [(d["text"], claim) for d in docs]
    logits = nli_model.predict(pairs, apply_softmax=True)  # shape: (n_docs, 3)

    # Column index 1 = entailment probability
    entail = [float(row[1]) for row in logits]

    bm25_scores = np.array([d["score"] for d in docs], dtype=float)
    weights = bm25_scores / bm25_scores.sum() if bm25_scores.sum() > 0 \
              else np.ones(len(docs)) / len(docs)

    return {
        "per_doc_entailment": entail,
        "max_score":          float(max(entail)),
        "mean_score":         float(np.mean(entail)),
        "weighted_score":     float(np.dot(weights, entail)),
    }
