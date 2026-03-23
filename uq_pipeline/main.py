"""
main.py
=======
UQ in Post-Generation Attribution — Main Pipeline

Full end-to-end orchestration. Runs all UQ stages and saves results.

Usage:
    python main.py --num_questions 3 --w1 0.4 --w2 0.35 --w3 0.25

Arguments:
    --num_questions  : how many ASQA questions to process (default=5)
    --w1             : weight for UQ₁ Attribution         (default=0.40)
    --w2             : weight for UQ₂ Retrieval            (default=0.35)
    --w3             : weight for UQ₃ Hallucination/Content (default=0.25)
                       Set w3=0 to skip content UQ entirely.
    --run_uq4        : flag to run UQ₄ claim extraction diagnostic (default=False)
    --top_k          : BM25 top-k documents to retrieve   (default=5)

Weights constraint: w1 + w2 + w3 must equal 1.0 (enforced at runtime).

Output:
    results/attribution_uncertainty.jsonl  — per-claim full record
    results/attribution_uncertainty.csv    — flat per-claim summary
    results/question_summary.json          — per-question aggregates

Pipeline flow (mirrors README.md):
    Step 1 : Load API key + config
    Step 2 : Build BM25 index from ASQA + ALCE corpus
    Step 3 : Generate LLM answer per question
    Step 4 : Extract atomic claims from answer
    Step 5 : Generate 12 perturbations per claim (5 categories)
    Step 6 : Retrieve top-k docs + NLI score for base + all 12 perturbs
    Step 7 : Compute UQ₁ (Attribution)      via uq_attribution.py
    Step 8 : Compute UQ₂ (Retrieval)        via uq_retrieval.py
    Step 9 : Compute UQ₃ (Content)          via uq_content.py  [if w3 > 0]
    Step 10: Compute UQ₄ (Claim Extraction) via uq_claim_extraction.py [if --run_uq4]
    Step 11: Compute composite U_attr = w1*UQ1 + w2*UQ2 + w3*UQ3
    Step 12: Save records + print report
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ── Local modules ──────────────────────────────────────────────────────────
from pipeline_utils import (
    load_api_key,
    build_llm_client,
    build_bm25_index,
    bm25_retrieve,
    load_nli_model,
    nli_score_claim,
    generate_answer,
    extract_atomic_claims,
    generate_all_perturbations,
    ALL_PERT_KEYS,
)
from uq_attribution      import compute_uq_attribution,      interpret_uq1
from uq_retrieval        import compute_uq_retrieval,        interpret_uq2
from uq_content          import compute_uq_content,          interpret_uq3
from uq_claim_extraction import compute_uq_claim_extraction

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger("uq_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="UQ in Post-Generation Attribution")
    p.add_argument("--num_questions", type=int,   default=5,    help="Number of ASQA questions (default=5)")
    p.add_argument("--w1",            type=float, default=0.40, help="Weight for UQ1 Attribution (default=0.40)")
    p.add_argument("--w2",            type=float, default=0.35, help="Weight for UQ2 Retrieval (default=0.35)")
    p.add_argument("--w3",            type=float, default=0.25, help="Weight for UQ3 Content (default=0.25; 0=skip)")
    p.add_argument("--top_k",         type=int,   default=5,    help="BM25 top-k docs to retrieve (default=5)")
    p.add_argument("--run_uq4",       action="store_true",      help="Run UQ4 claim extraction diagnostic")
    p.add_argument("--seed",          type=int,   default=42,   help="Random seed")
    return p.parse_args()


def validate_weights(w1: float, w2: float, w3: float) -> None:
    """Ensure weights are non-negative and sum to 1.0."""
    if any(w < 0 for w in [w1, w2, w3]):
        sys.exit("Error: All weights must be >= 0.")
    total = w1 + w2 + w3
    if abs(total - 1.0) > 1e-6:
        sys.exit(f"Error: Weights must sum to 1.0 (got {total:.4f}). "
                 f"Adjust w1={w1}, w2={w2}, w3={w3}.")


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_composite(uq1_norm: float, uq2_norm: float, uq3_norm: float,
                      w1: float, w2: float, w3: float) -> dict:
    """
    Step 11: Compute composite U_attr.

    U_attr = w1 * UQ1_norm + w2 * UQ2_norm + w3 * UQ3_norm
    All components already normalized to [0, 1].

    If w3=0, UQ3 is simply ignored (contributes 0 to the sum).
    """
    u_attr = w1 * uq1_norm + w2 * uq2_norm + w3 * uq3_norm
    level  = "HIGH" if u_attr >= 0.5 else ("MODERATE" if u_attr >= 0.2 else "STABLE")
    return {"U_attr": u_attr, "level": level}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    validate_weights(args.w1, args.w2, args.w3)

    print("\n" + "=" * 65)
    print("  UQ IN POST-GENERATION ATTRIBUTION")
    print(f"  Questions: {args.num_questions} | top_k: {args.top_k}")
    print(f"  Weights → w1(UQ1)={args.w1} | w2(UQ2)={args.w2} | w3(UQ3)={args.w3}")
    print(f"  UQ3 (Content): {'ACTIVE' if args.w3 > 0 else 'SKIPPED (w3=0)'}")
    print(f"  UQ4 (Claim Extraction): {'ACTIVE (diagnostic)' if args.run_uq4 else 'SKIPPED'}")
    print("=" * 65 + "\n")

    # ── STEP 1: Load API key + build LLM client ─────────────────────────────
    api_key = load_api_key()
    llm     = build_llm_client(api_key)

    # ── STEP 2: Build BM25 index ─────────────────────────────────────────────
    questions, bm25_index, corpus_texts, corpus_ids = build_bm25_index(
        num_questions=args.num_questions
    )

    # ── Load NLI model (once) ────────────────────────────────────────────────
    nli_model = load_nli_model()

    # ── Storage ───────────────────────────────────────────────────────────────
    all_records = []
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)
    JSONL_PATH  = RESULTS_DIR / "attribution_uncertainty.jsonl"
    if JSONL_PATH.exists():
        JSONL_PATH.unlink()

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN LOOP: one question at a time
    # ─────────────────────────────────────────────────────────────────────────
    for q_idx, question in tqdm(enumerate(questions), total=len(questions),
                                 desc="Questions"):
        print(f"\n{'─'*60}")
        print(f"Q-{q_idx+1:02d}: {question}")

        # ── STEP 3: Generate answer ──────────────────────────────────────────
        answer = generate_answer(llm, question)
        if not answer:
            print(f"  [skip] Empty answer for Q-{q_idx+1}")
            continue
        print(f"  Answer: {answer[:100]}...")

        # ── STEP 4: Extract atomic claims ────────────────────────────────────
        claims = extract_atomic_claims(llm, answer, temperature=0.0)
        if not claims:
            print(f"  [skip] No claims extracted for Q-{q_idx+1}")
            continue
        print(f"  Extracted {len(claims)} atomic claims")

        # ── STEP 10: UQ₄ — run ONCE per answer (diagnostic only) ─────────────
        uq4_result = None
        if args.run_uq4:
            print("  Computing UQ₄ (claim extraction diagnostic)...")
            uq4_result = compute_uq_claim_extraction(llm, answer, nli_model)
            print(f"  {uq4_result['interpretation']}")

        # ─── PER-CLAIM LOOP ───────────────────────────────────────────────────
        for c_idx, claim in tqdm(enumerate(claims), total=len(claims),
                                  leave=False, desc=f"  Claims Q{q_idx+1}"):

            # ── STEP 5: Generate 12 perturbations ────────────────────────────
            # Returns dict: {pert_key → perturbed_claim_text}
            perturbations = generate_all_perturbations(llm, claim)

            # ── STEP 6: Retrieve docs + NLI score ────────────────────────────
            # Build two parallel maps:
            #   docs_map       : {variant_key → list of doc dicts}
            #   nli_scores_map : {variant_key → nli_score_claim() output}

            docs_map       = {}
            nli_scores_map = {}

            # Base claim
            base_docs              = bm25_retrieve(claim, bm25_index,
                                                   corpus_texts, corpus_ids,
                                                   top_k=args.top_k)
            docs_map["base"]       = base_docs
            nli_scores_map["base"] = nli_score_claim(claim, base_docs, nli_model)

            # All 12 perturbations
            for key, pert_text in perturbations.items():
                p_docs              = bm25_retrieve(pert_text, bm25_index,
                                                    corpus_texts, corpus_ids,
                                                    top_k=args.top_k)
                docs_map[key]       = p_docs
                nli_scores_map[key] = nli_score_claim(pert_text, p_docs, nli_model)

            # ── STEP 7: UQ₁ — Attribution ────────────────────────────────────
            uq1 = compute_uq_attribution(nli_scores_map)
            print(f"    [{c_idx+1}] {interpret_uq1(uq1)}")

            # ── STEP 8: UQ₂ — Retrieval ──────────────────────────────────────
            uq2 = compute_uq_retrieval(docs_map)
            print(f"    [{c_idx+1}] {interpret_uq2(uq2)}")

            # ── STEP 9: UQ₃ — Content (skipped if w3=0) ─────────────────────
            uq3 = None
            if args.w3 > 0:
                uq3 = compute_uq_content(nli_scores_map)
                print(f"    [{c_idx+1}] {interpret_uq3(uq3)}")

            uq3_norm = uq3["UQ3_norm"] if uq3 else 0.0

            # ── STEP 11: Composite U_attr ─────────────────────────────────────
            composite = compute_composite(
                uq1["UQ1_norm"], uq2["UQ2_norm"], uq3_norm,
                args.w1, args.w2, args.w3
            )
            print(f"    [{c_idx+1}] → U_attr={composite['U_attr']:.4f} [{composite['level']}]")

            # ── Build & save record ───────────────────────────────────────────
            record = {
                "q_idx":    q_idx,
                "question": question,
                "answer":   answer,
                "c_idx":    c_idx,
                "claim":    claim,

                # Perturbations
                "perturbations": {k: v for k, v in perturbations.items()},

                # NLI scores (base + all perturbs)
                "nli_scores": {
                    k: {"max_score": v["max_score"], "mean_score": v["mean_score"]}
                    for k, v in nli_scores_map.items()
                },

                # Retrieved doc IDs (truncated for storage)
                "doc_ids": {k: [d["id"] for d in v] for k, v in docs_map.items()},

                # UQ signals
                "UQ1_norm":          uq1["UQ1_norm"],
                "UQ1_raw":           uq1["UQ1_raw"],
                "score_preserve_mean": uq1["score_preserve_mean"],
                "score_destroy_mean":  uq1["score_destroy_mean"],

                "UQ2_norm":          uq2["UQ2_norm"],
                "mean_rbo":          uq2["mean_rbo"],
                "mean_jaccard":      uq2["mean_jaccard"],

                "UQ3_norm":          uq3_norm,
                "Signal_A_norm":     uq3["Signal_A_norm"] if uq3 else None,
                "Signal_B_norm":     uq3["Signal_B_norm"] if uq3 else None,

                # UQ4 diagnostic (per-answer, attached to all claims in that answer)
                "UQ4_diagnostic":    uq4_result["UQ4_diagnostic"] if uq4_result else None,
                "UQ4_interpretation": uq4_result["interpretation"] if uq4_result else None,

                # Composite
                "U_attr": composite["U_attr"],
                "level":  composite["level"],

                # Weights used (for reproducibility)
                "weights": {"w1": args.w1, "w2": args.w2, "w3": args.w3},
            }
            all_records.append(record)

            # Checkpoint: write to JSONL after each claim
            with open(JSONL_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 12: SAVE RESULTS & PRINT REPORT
    # ─────────────────────────────────────────────────────────────────────────
    if not all_records:
        print("\nNo records collected. Check API key and retry.")
        return

    # Flat CSV
    flat = [{
        "q_idx":       r["q_idx"],
        "question":    r["question"],
        "c_idx":       r["c_idx"],
        "claim":       r["claim"],
        "UQ1_norm":    r["UQ1_norm"],
        "UQ2_norm":    r["UQ2_norm"],
        "UQ3_norm":    r["UQ3_norm"],
        "U_attr":      r["U_attr"],
        "level":       r["level"],
        "mean_rbo":    r["mean_rbo"],
        "mean_jaccard":r["mean_jaccard"],
        "score_preserve_mean": r["score_preserve_mean"],
        "score_destroy_mean":  r["score_destroy_mean"],
        "UQ4_diagnostic": r["UQ4_diagnostic"],
    } for r in all_records]

    df = pd.DataFrame(flat)
    df.to_csv(RESULTS_DIR / "attribution_uncertainty.csv", index=False)

    # Per-question summary
    q_summary = []
    for q_key, grp in df.groupby("q_idx"):
        q_summary.append({
            "q_idx":        int(q_key),
            "question":     grp["question"].iloc[0],
            "num_claims":   len(grp),
            "mean_U_attr":  float(grp["U_attr"].mean()),
            "std_U_attr":   float(grp["U_attr"].std()),
            "max_U_attr":   float(grp["U_attr"].max()),
            "high_pct":     float((grp["level"] == "HIGH").mean() * 100),
            "stable_pct":   float((grp["level"] == "STABLE").mean() * 100),
            "mean_UQ1":     float(grp["UQ1_norm"].mean()),
            "mean_UQ2":     float(grp["UQ2_norm"].mean()),
            "mean_UQ3":     float(grp["UQ3_norm"].mean()),
        })
    with open(RESULTS_DIR / "question_summary.json", "w") as f:
        json.dump(q_summary, f, indent=2)

    # Terminal report
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Total claims : {len(df)}")
    print(f"  Mean U_attr  : {df['U_attr'].mean():.4f}")
    print(f"  Std  U_attr  : {df['U_attr'].std():.4f}")
    counts = df["level"].value_counts()
    for lvl in ["STABLE", "MODERATE", "HIGH"]:
        n = counts.get(lvl, 0)
        print(f"  {lvl:<10} : {n:3d} ({n / len(df) * 100:.1f}%)")
    print(f"\n  Mean UQ1 (Attribution) : {df['UQ1_norm'].mean():.4f}")
    print(f"  Mean UQ2 (Retrieval)   : {df['UQ2_norm'].mean():.4f}")
    print(f"  Mean UQ3 (Content)     : {df['UQ3_norm'].mean():.4f}")
    print(f"\n  Files saved to: {RESULTS_DIR.resolve()}/")
    for p in sorted(RESULTS_DIR.rglob("*")):
        if p.is_file():
            print(f"    {p.name}")
    print("=" * 65)


if __name__ == "__main__":
    main()
