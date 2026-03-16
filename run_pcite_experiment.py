"""
=============================================================================
  UNCERTAINTY QUANTIFICATION IN POST-GENERATION ATTRIBUTION (P-Cite)
  Consolidated Single-File Multi-Model Experiment
=============================================================================

This script runs the entire attribution uncertainty pipeline across multiple LLMs
on the ASQA dataset in a single run. The output is stored cleanly under the 
"results" directory.

WHAT THIS SCRIPT DOES (per claim):
  1. Generate answer → Extract claims
  2. Generate K=5 semantic perturbations per claim
  3. Retrieve top-3 docs via BM25 for original + each perturbation
  4. Embed retrieved docs with sentence-transformers (GPU-accelerated)
  5. Compute U_attr = Variance(cosine_similarities) across perturbations

HOW TO RUN:
  1. Put your OpenRouter API key in the `.env` file!
  2. Modify the "STEP 1: USER CONFIGURATION" section below to your liking.
  3. Run: python run_pcite_experiment.py
"""

# ========================================================================
# STEP 1: USER CONFIGURATION (Edit this Section)
# ========================================================================

# --- Models to Evaluate ---
MODELS_TO_TEST = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
]

# Provide shorter/friendly names for saving result directories
MODEL_SHORT_NAMES = {
    "meta-llama/llama-3.3-70b-instruct:free"       : "llama-3.3-70b",
    "google/gemma-3-27b-it:free"                    : "gemma-3-27b",
    "mistralai/mistral-small-3.1-24b-instruct:free" : "mistral-small-24b",
    "nvidia/nemotron-3-super-120b-a12b:free"        : "nemotron-120b",
    "qwen/qwen3-next-80b-a3b-instruct:free"         : "qwen3-80b",
}

# --- Experiment Settings ---
NUM_INSTANCES = 10        # Number of ASQA questions to test per model (e.g. 10, 40, or up to 948)
NUM_PERTURBATIONS = 5      # K perturbations per claim
TOP_K_DOCS = 3             # Number of BM25 retrieved docs per query

# --- Data Source Toggles ---
USE_ALCE_CORPUS = True     # Use ALCE gold passages (from din0s/asqa)
USE_WIKIPEDIA   = True     # Use general Wikipedia passages (streamed from huggingface)
MAX_WIKI_PASSAGES = 10000  # How many Wikipedia chunks to load if USE_WIKIPEDIA is True

# --- API & Performance Settings ---
PARALLEL_WORKERS = 3       # Max concurrent API requests for perturbations (keep low for free-tier)
MAX_RETRIES = 6            # Retries when rate limited (429) happens
MAX_TOKENS_ANSWER = 512    # Generation token limit for answering
MAX_TOKENS_PERTURB = 256   # Generation token limit for perturbing

# --- Local Output Path ---
import pathlib
import os
RESULTS_DIR = pathlib.Path("./results")


# ========================================================================
# STEP 2: IMPORTS
# ========================================================================
print("Loading libraries...")
import os
import json
import logging
import random
import re
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from datasets import load_dataset
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()

# System config
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("pcite")
logger.setLevel(logging.INFO)

if "OPENROUTER_API_KEY" not in os.environ or not os.environ["OPENROUTER_API_KEY"]:
    raise ValueError("\\n❌ Missing OPENROUTER_API_KEY. Please add it to your .env file or environment variables.")
else:
    print("✅ Verified OpenRouter API Key.")


# ========================================================================
# STEP 3: CORE LOGIC & CLASSES
# ========================================================================

class LLMClient:
    """Handles communication with the OpenRouter API with retry & rate limiting logic."""
    def __init__(self, model: str):
        self.client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = model

    def _call(self, messages: list[dict], max_tokens: int) -> str:
        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                content = r.choices[0].message.content
                if content is None:
                    raise ValueError("Null content")
                time.sleep(0.5)  # Quick cooldown
                return content.strip()
            except Exception as exc:
                last_exc = exc
                err = str(exc)
                is_rate = "429" in err or "rate" in err.lower() or "402" in err
                wait = min(60.0 * (2 ** (attempt - 1)), 600) if is_rate else 10.0 * attempt
                logger.warning(f"  API {attempt}/{MAX_RETRIES} {'[429 RATE-LIMITED]' if is_rate else ''}: retrying in {wait:.0f}s")
                time.sleep(wait)
        raise RuntimeError(f"API failed after {MAX_RETRIES} retries") from last_exc

    def generate_answer(self, question: str) -> str:
        return self._call([
            {"role": "system", "content": "You are a knowledgeable assistant. Answer factual questions accurately in 3-6 sentences."},
            {"role": "user", "content": f"Answer: {question}"},
        ], max_tokens=MAX_TOKENS_ANSWER)

    def generate_one_perturbation(self, claim: str) -> str:
        text = self._call([
            {"role": "system", "content": "Rephrase the factual claim without changing its meaning. Return ONLY the rephrased claim."},
            {"role": "user", "content": f"Rephrase: {claim}"},
        ], max_tokens=MAX_TOKENS_PERTURB)
        for prefix in ("rephrased claim:", "claim:", "rephrase:"):
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
        return text if text else claim

def extract_claims(text: str) -> list[str]:
    """Split text into sentence-level claims."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 15]

class BM25Retriever:
    """Local text retrieval using BM25."""
    def __init__(self, passages: list[dict]):
        self.passages = passages
        tokenized = [p["text"].lower().split() for p in passages]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[-top_k:][::-1]
        return [self.passages[i] for i in top_idx]

class Embedder:
    """Generates text embeddings using sentence-transformers."""
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        logger.info(f"  Embedding model loaded on {DEVICE}")

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)

    def mean_embedding(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts).mean(axis=0)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(sklearn_cosine(a.reshape(1, -1), b.reshape(1, -1))[0, 0])

def compute_uncertainty(similarities: list[float]) -> float:
    return float(np.var(similarities)) if len(similarities) >= 2 else 0.0


# ========================================================================
# STEP 4: DATA LOADING
# ========================================================================
def load_data():
    print(f"\\n📚 Loading ASQA dataset (first {NUM_INSTANCES} instances) ...")
    try:
        asqa_ds = load_dataset("din0s/asqa", split="dev")
    except Exception:
        asqa_ds = load_dataset("din0s/asqa", split="dev", trust_remote_code=True)
    
    asqa_records = list(asqa_ds.select(range(min(NUM_INSTANCES, len(asqa_ds)))))
    questions = [r["ambiguous_question"] for r in asqa_records]
    print(f"✅ Loaded {len(questions)} questions.")

    print("\\n📚 Loading retrieval corpus ...")
    all_passages = []

    if USE_ALCE_CORPUS:
        try:
            print("   -> Loading ALCE gold passages ...")
            alce_ds = load_dataset("din0s/asqa", split="dev")
            alce_passages = []
            for record in alce_ds:
                if "qa_pairs" in record:
                    for qa in record["qa_pairs"]:
                        if "context" in qa and qa["context"]:
                            ctx = qa["context"]
                            text = ctx if isinstance(ctx, str) else " ".join(ctx) if isinstance(ctx, list) else str(ctx)
                            alce_passages.append({"title": record.get("ambiguous_question", "ALCE"), "text": text, "source": "ALCE"})
            all_passages.extend(alce_passages)
            print(f"      ✅ ALCE: {len(alce_passages)} passages")
        except Exception as e:
            print(f"      ⚠️  ALCE load failed: {e}")

    if USE_WIKIPEDIA:
        try:
            print(f"   -> Loading Wikipedia passages (up to {MAX_WIKI_PASSAGES:,}) ...")
            wiki_ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
            wiki_passages = []
            for article in wiki_ds:
                text = article.get("text", "")
                title = article.get("title", "Wikipedia")
                if not text or len(text) < 100: continue
                chunks = [text[i:i+300] for i in range(0, min(len(text), 2000), 300)]
                for chunk in chunks:
                    if len(chunk.strip()) > 50:
                        wiki_passages.append({"title": title, "text": chunk.strip(), "source": "Wikipedia"})
                if len(wiki_passages) >= MAX_WIKI_PASSAGES: break
            all_passages.extend(wiki_passages[:MAX_WIKI_PASSAGES])
            print(f"      ✅ Wikipedia: {len(wiki_passages[:MAX_WIKI_PASSAGES])} passages")
        except Exception as e:
            print(f"      ⚠️  Wikipedia load failed: {e}")

    print(f"\\n📊 Total retrieval corpus: {len(all_passages):,} passages")
    if not all_passages:
        raise RuntimeError("No passages loaded! Please enable ALCE or Wikipedia in settings.")
    
    return questions, all_passages


# ========================================================================
# STEP 5: PIPELINE EXECUTION WRAPPER
# ========================================================================
def run_pipeline_for_model(model_id: str, questions: list[str], retriever: BM25Retriever, embedder: Embedder) -> list[dict]:
    short = MODEL_SHORT_NAMES.get(model_id, model_id.split("/")[-1])
    model_dir = RESULTS_DIR / short
    model_dir.mkdir(parents=True, exist_ok=True)

    llm = LLMClient(model_id)
    records = []
    
    jsonl_path = model_dir / "results.jsonl"
    if jsonl_path.exists(): jsonl_path.unlink()

    print(f"\\n\\n🔄 Processing {short} ...")
    pbar = tqdm(questions, desc=f"{short}", unit="q", ncols=95)
    for q_idx, question in enumerate(pbar):
        pbar.set_postfix(claims=len(records))

        # 1. Answer question
        try:
            answer = llm.generate_answer(question)
        except Exception as e:
            logger.warning(f"  ❌ Answer failed Q{q_idx+1}: {e}")
            continue

        # 2. Extract facts
        claims = extract_claims(answer)
        if not claims: continue

        for claim in claims:
            # 3. Process original claim
            orig_docs = retriever.retrieve(claim, top_k=TOP_K_DOCS)
            orig_emb = embedder.mean_embedding([d["text"] for d in orig_docs])

            # 4. Generate Perturbations
            def _gen(c):
                try: return llm.generate_one_perturbation(c)
                except: return c

            with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
                perturbations = [f.result() for f in as_completed([pool.submit(_gen, claim) for _ in range(NUM_PERTURBATIONS)])]

            # 5. Retrieve + Embed for perturbations
            similarities = []
            for perturb in perturbations:
                p_docs = retriever.retrieve(perturb, top_k=TOP_K_DOCS)
                p_emb = embedder.mean_embedding([d["text"] for d in p_docs])
                similarities.append(cosine_sim(orig_emb, p_emb))

            # 6. Store
            u_attr = compute_uncertainty(similarities)
            record = {
                "model": model_id, "model_short": short,
                "question": question, "generated_answer": answer,
                "fi": claim, "perturbations": perturbations,
                "similarities": [round(s, 6) for s in similarities],
                "uncertainty": round(u_attr, 6),
            }
            records.append(record)
            
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\\n")
        
        # Cooldown per question
        time.sleep(3)
    pbar.close()
    return records


def process_and_save_stats(records: list[dict], model_id: str):
    short = MODEL_SHORT_NAMES.get(model_id, model_id.split("/")[-1])
    model_dir = RESULTS_DIR / short
    if not records:
        print(f"  ⚠️  No records generated for {short}")
        return {}

    all_u = [r["uncertainty"] for r in records]
    
    # Save CSV
    csv_rows = [{"model": short, "question": r["question"], "claim": r["fi"],
                 "uncertainty": r["uncertainty"], "sim_mean": round(np.mean(r["similarities"]), 6),
                 "sim_std": round(np.std(r["similarities"]), 6),
                 **{f"sim_{i+1}": s for i, s in enumerate(r["similarities"])}} for r in records]
    pd.DataFrame(csv_rows).to_csv(model_dir / "results.csv", index=False)
    
    # Generate Stats
    q_u = defaultdict(list)
    for r in records: q_u[r["question"]].append(r["uncertainty"])
    stats = {
        "model_short": short, "model_full": model_id,
        "num_questions": len(q_u), "num_claims": len(records),
        "mean_U_attr": round(statistics.mean(all_u), 6),
        "std_U_attr": round(statistics.stdev(all_u) if len(all_u)>1 else 0, 6),
        "median_U_attr": round(statistics.median(all_u), 6),
        "max_U_attr": round(max(all_u), 6),
        "high_uncertainty_pct": round(100 * sum(1 for u in all_u if u > 0.03) / len(all_u), 2),
    }
    with open(model_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  📊 {short}: {stats['num_claims']} claims | Mean U={stats['mean_U_attr']:.4f} | Max U={stats['max_U_attr']:.4f}")
    return stats


# ========================================================================
# MAIN SCRIPT EXECUTION
# ========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  🚀 P-CITE EXPERIMENT (SINGLE-FILE RUNNER)")
    print("=" * 70)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Dataset & Indexing
    questions, all_passages = load_data()
    print("\\n🔧 Building BM25 index ...")
    retriever = BM25Retriever(all_passages)
    print(f"   ✅ BM25 Ready")
    
    print(f"\\n🧠 Loading embedding model directly on {DEVICE} ...")
    embedder = Embedder()
    print(f"   ✅ Embedder Ready")

    # 2. Iterate Models
    all_model_stats = []
    for m_idx, model_id in enumerate(MODELS_TO_TEST):
        print(f"\\n{'='*70}")
        print(f"  Model {m_idx+1} / {len(MODELS_TO_TEST)}: {model_id}")
        print(f"{'='*70}")
        try:
            records = run_pipeline_for_model(model_id, questions, retriever, embedder)
            if records:
                stats = process_and_save_stats(records, model_id)
                all_model_stats.append(stats)
            if m_idx < len(MODELS_TO_TEST) - 1:
                print("  ⏸️  Yielding 30s to OpenRouter rate limits...")
                time.sleep(30)
        except Exception as e:
            print(f"  ❌ Failed for model {model_id}: {e}")

    # 3. Final Multi-Model Report
    if all_model_stats:
        print("\\n\\n" + "="*80)
        print("  📊 ATTRIBUTION UNCERTAINTY MULTI-MODEL COMPARISON")
        print("="*80)
        df_comp = pd.DataFrame([{
            "Model": s["model_short"], "Questions": s["num_questions"], "Claims": s["num_claims"],
            "Mean_U": s["mean_U_attr"], "Std_U": s["std_U_attr"], "Max_U": s["max_U_attr"],
            "High_U(%)": s["high_uncertainty_pct"]
        } for s in all_model_stats]).sort_values("Mean_U", ascending=True)

        print("\\n" + df_comp.to_string(index=False) + "\\n")
        
        comp_csv = RESULTS_DIR / "model_comparison_aggregate.csv"
        df_comp.to_csv(comp_csv, index=False)
        with open(RESULTS_DIR / "model_comparison_aggregate.json", "w") as f:
            json.dump(all_model_stats, f, indent=2)
            
        print(f"💾 Aggregate results saved to: {comp_csv}")
    else:
        print("\\n❌ Experiment didn't complete for any models.")
