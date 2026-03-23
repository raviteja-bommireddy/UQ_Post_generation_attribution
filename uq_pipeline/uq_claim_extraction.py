"""
uq_claim_extraction.py
======================
UQ₄ — Claim Extraction Uncertainty (DIAGNOSTIC ONLY)

Status: Open Problem.
    This module is a DIAGNOSTIC — it does NOT enter the composite U_attr score.
    It measures uncertainty in the measurement process itself (meta-uncertainty).
    Including it in U_attr would conflate pipeline reliability with attribution
    uncertainty — which are qualitatively different things.

What this measures:
    Research (HalluMeasure 2024; FactScore 2023) explicitly acknowledges that
    "the quality and consistency of claim decomposition directly impacts probe
    performance." The atomic decomposition step is a critical source of error:

    Known failure modes:
        - Over-splitting  : one fact split into two weak claims
        - Under-splitting : two facts fused — forces multi-hop attribution
        - Coreference loss: "He won the prize" loses its entity anchor
        - Boundary ambiguity: LLM draws claim boundaries differently per run

Method:
    Run claim extraction K=3 times at temperature T > 0.
    For each claim in run_1, find its best NLI match across run_2 and run_3.
    Low NLI agreement across runs → the decomposer is unstable → high UQ₄.

    Also track: variance in the NUMBER of claims extracted per run.
    High count variance = decomposition is non-deterministic.

Why NOT in composite U_attr:
    UQ₄ measures meta-uncertainty — uncertainty about whether the claims
    themselves are well-defined units. This is important context for
    INTERPRETING U_attr, not for computing it.

Inputs:
    llm    : built_llm_client() output
    answer : the LLM-generated answer string
    nli_model : loaded NLI cross-encoder

Output:
    dict with agreement_scores, count_variance, UQ4_diagnostic
"""

import numpy as np
from pipeline_utils import extract_atomic_claims, nli_score_claim


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UQ₄ FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_uq_claim_extraction(llm: dict, answer: str, nli_model,
                                 n_runs: int = 3,
                                 temperature: float = 0.7) -> dict:
    """
    Compute UQ₄ — Claim Extraction Uncertainty (diagnostic).

    Parameters
    ----------
    llm         : dict from build_llm_client()
    answer      : LLM-generated answer string
    nli_model   : loaded CrossEncoder from load_nli_model()
    n_runs      : number of independent extraction runs (default=3)
    temperature : LLM temperature for extraction runs (must be > 0 for variation)

    Returns
    -------
    dict:
        runs              : list of claim lists (one per run)
        claim_counts      : list of int — number of claims per run
        count_variance    : float — variance in claim count across runs
        agreement_scores  : list of float — per-claim cross-run NLI agreement
        mean_agreement    : float — mean NLI agreement across runs
        UQ4_diagnostic    : float — 1 - mean_agreement ∈ [0, 1]
        interpretation    : str
    """
    # --- Run claim extraction N times at T > 0 ---
    runs = []
    for i in range(n_runs):
        claims = extract_atomic_claims(llm, answer, temperature=temperature)
        runs.append(claims)

    claim_counts   = [len(r) for r in runs]
    count_variance = float(np.var(claim_counts))

    # --- Compute cross-run NLI agreement ---
    # For each claim in run_0, find its best NLI match in run_1 + run_2 combined.
    # Best match = highest entailment score across all claims in other runs.
    # We use a simple doc-as-sentence trick: treat each claim as a 1-doc corpus.

    agreement_scores = []
    base_run = runs[0] if runs else []
    other_claims = [c for r in runs[1:] for c in r]  # flatten all other runs

    if base_run and other_claims:
        for claim in base_run:
            # Create pseudo-docs from other claims
            pseudo_docs = [
                {"id": f"run_claim_{i}", "text": c, "score": 1.0}
                for i, c in enumerate(other_claims)
            ]
            result = nli_score_claim(claim, pseudo_docs, nli_model)
            # Best match = max entailment of this claim against any claim in other runs
            agreement_scores.append(result["max_score"])

    mean_agreement  = float(np.mean(agreement_scores)) if agreement_scores else 0.0
    UQ4_diagnostic  = 1.0 - mean_agreement  # high = low agreement = unstable decomposition

    # --- Build interpretation ---
    level = "HIGH" if UQ4_diagnostic >= 0.5 else ("MODERATE" if UQ4_diagnostic >= 0.2 else "STABLE")
    notes = []
    if count_variance > 2.0:
        notes.append(f"Claim count varies significantly across runs (var={count_variance:.2f}) — boundary instability.")
    if UQ4_diagnostic >= 0.5:
        notes.append("Low cross-run NLI agreement — decomposer is non-deterministic for this answer.")
    if not notes:
        notes.append("Decomposition appears consistent across runs.")

    interp = (
        f"UQ4_diagnostic={UQ4_diagnostic:.4f} [{level}] "
        f"| mean_agreement={mean_agreement:.3f} "
        f"| count_variance={count_variance:.2f} "
        f"| {' '.join(notes)}"
    )

    return {
        "runs":             runs,
        "claim_counts":     claim_counts,
        "count_variance":   count_variance,
        "agreement_scores": agreement_scores,
        "mean_agreement":   mean_agreement,
        "UQ4_diagnostic":   UQ4_diagnostic,   # NOT in composite U_attr
        "interpretation":   interp,
    }
