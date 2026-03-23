"""
uq_content.py
=============
UQ₃ — Hallucination / Content Uncertainty

What this measures:
    Did the LLM generate specific details that are NOT grounded in any
    retrievable corpus passage?

Two-signal approach (averaged):
    Signal A — Retrieval Anchoring Bias (Attribution Drop)
        Measures whether the original claim phrasing enjoys disproportionately
        high support compared to ALL its perturbation variants.
        Formula: Score_base − mean(all 12 perturbation scores)
        → Simple to compute; direct byproduct of NLI scoring step.
        → Captures phrase-specific corpus anchoring (indirect hallucination signal).

    Signal B — Specification Entailment Gap (SEG)
        Uses ONLY the C5 (depth/specification) perturbations:
        C5_causal_detail and C5_consequence_detail.
        If the base claim is well-entailed but its specification variants
        (added causal/consequence details) are NOT entailed → those specific
        details were likely hallucinated by the LLM.
        Formula: Score_base − mean(Score_C5_causal, Score_C5_consequence)
        → More targeted: directly probes whether added specifics are corpus-grounded.

Final UQ₃:
    UQ3_norm = avg(Signal_A_norm, Signal_B_norm)

    If w3=0 in main.py, this entire module is skipped.

Inputs (from main.py):
    nli_scores_map : dict mapping variant_key → nli_score_claim() output
"""

import numpy as np

# C5 keys (depth/specification perturbations) — used for Signal B
C5_KEYS = ["C5_causal_detail", "C5_consequence_detail"]


def _normalize_drop(drop_raw: float) -> float:
    """
    Normalize a signed drop value (range [-1, +1]) to [0, 1].
    drop_raw < 0 means perturbations score HIGHER than base → robust attribution.
    drop_raw > 0 means base score is INFLATED vs perturbations.
    """
    return max(0.0, min(1.0, (drop_raw - (-1.0)) / (1.0 - (-1.0))))


def compute_uq_content(nli_scores_map: dict) -> dict:
    """
    Compute UQ₃ — Hallucination / Content Uncertainty.

    Parameters
    ----------
    nli_scores_map : dict
        Keys: "base" + all 12 perturbation keys.
        Values: output of nli_score_claim() with "max_score" field.

    Returns
    -------
    dict:
        score_base          : float — NLI max score for base claim
        Signal_A_raw        : float — Attribution Drop (Score_base - mean_all_perturbs)
        Signal_A_norm       : float — Signal A normalized to [0, 1]
        Signal_B_raw        : float — SEG (Score_base - mean_C5_scores)
        Signal_B_norm       : float — Signal B normalized to [0, 1]
        UQ3_norm            : float — avg(Signal_A_norm, Signal_B_norm) ∈ [0, 1]
        c5_scores           : dict  — {key: score} for C5 perturbations
    """
    score_base = nli_scores_map["base"]["max_score"]

    # --- Signal A: Retrieval Anchoring Bias (Attribution Drop) ---
    # Uses ALL 12 perturbation scores
    all_pert_scores = [
        nli_scores_map[k]["max_score"]
        for k in nli_scores_map if k != "base"
    ]
    mean_all_perturbs = float(np.mean(all_pert_scores)) if all_pert_scores else score_base

    Signal_A_raw  = score_base - mean_all_perturbs  # positive = base is inflated
    Signal_A_norm = _normalize_drop(Signal_A_raw)

    # --- Signal B: Specification Entailment Gap (SEG) ---
    # Uses ONLY C5 (depth/specification) perturbations
    c5_scores = {}
    for k in C5_KEYS:
        if k in nli_scores_map:
            c5_scores[k] = nli_scores_map[k]["max_score"]

    if c5_scores:
        mean_c5       = float(np.mean(list(c5_scores.values())))
        Signal_B_raw  = score_base - mean_c5         # positive = specs not supported
        Signal_B_norm = _normalize_drop(Signal_B_raw)
    else:
        # Fallback: if C5 keys not present, use Signal_A
        Signal_B_raw  = Signal_A_raw
        Signal_B_norm = Signal_A_norm

    # --- Composite UQ₃ ---
    UQ3_norm = (Signal_A_norm + Signal_B_norm) / 2.0

    return {
        "score_base":    score_base,
        "Signal_A_raw":  Signal_A_raw,   # attribution drop (all perturbs)
        "Signal_A_norm": Signal_A_norm,  # normalized
        "Signal_B_raw":  Signal_B_raw,   # specification entailment gap (C5 only)
        "Signal_B_norm": Signal_B_norm,  # normalized
        "UQ3_norm":      UQ3_norm,       # composite → used in U_attr
        "c5_scores":     c5_scores,      # raw C5 scores for inspection
    }


def interpret_uq3(result: dict) -> str:
    """
    Return a human-readable interpretation of the UQ₃ result.
    """
    u  = result["UQ3_norm"]
    sa = result["Signal_A_norm"]
    sb = result["Signal_B_norm"]
    level = "HIGH" if u >= 0.5 else ("MODERATE" if u >= 0.2 else "STABLE")

    notes = []
    if sb > 0.6:
        notes.append("Specification details (C5) are poorly supported — possible hallucination.")
    if sa > 0.6 and sb < 0.3:
        notes.append("Signal A high but Signal B low — phrase anchoring bias, not content hallucination.")
    if not notes:
        notes.append("Content appears broadly grounded in corpus.")

    return (
        f"UQ3={u:.4f} [{level}] | Signal_A(Drop)={sa:.3f} | Signal_B(SEG)={sb:.3f} "
        f"| {' '.join(notes)}"
    )
