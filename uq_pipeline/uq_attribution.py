"""
uq_attribution.py
=================
UQ₁ — Attribution Uncertainty

What this measures:
    How stable is the factual support (NLI entailment) a claim receives
    from its retrieved documents — across all 12 perturbations + base claim?

Core idea:
    A well-attributed claim should show:
      - HIGH entailment for Preserve perturbations (C1: P1, P2)
      - LOW  entailment for Destroy perturbations  (C2: D1, D2, D3, D4)
      - VARIABLE for Specificity / Breadth / Depth perturbs

    If NLI scores are noisy (high variance) regardless of category,
    the claim has no stable evidence anchor → high attribution uncertainty.

Inputs (from main.py):
    nli_scores_map : dict mapping variant_key → nli_score_claim() output
                     e.g. {"base": {...}, "C1_paraphrase": {...}, ...}

Output:
    dict with UQ1_raw, UQ1_norm, and per-variant scores for inspection.
"""

import numpy as np


def compute_uq_attribution(nli_scores_map: dict) -> dict:
    """
    Compute UQ₁ — Attribution Uncertainty.

    Parameters
    ----------
    nli_scores_map : dict
        Keys: "base" + all 12 perturbation keys.
        Values: output of nli_score_claim() — dicts with "max_score" etc.

    Returns
    -------
    dict:
        scores_all   : list of float  — all 13 max_score values (base + 12 perturbs)
        UQ1_raw      : float          — raw variance of scores_all
        UQ1_norm     : float          — normalized to [0, 1]  (divides by 0.25)
        score_base   : float
        score_preserve_mean : float   — mean over C1 keys
        score_destroy_mean  : float   — mean over C2 keys
    """
    # C1 (preserve) and C2 (destroy) keys — used for interpretability breakdown
    C1_KEYS = ["C1_paraphrase", "C1_voice_change"]
    C2_KEYS = ["C2_negation", "C2_factual_inversion", "C2_year_shift", "C2_entity_swap_destroy"]

    # --- Collect scores ---
    score_base = nli_scores_map["base"]["max_score"]

    # All 13 scores: base + 12 perturbations
    all_keys   = ["base"] + [k for k in nli_scores_map if k != "base"]
    scores_all = [nli_scores_map[k]["max_score"] for k in all_keys]

    # --- Variance across all 13 scores ---
    # Max theoretical variance for a probability in [0, 1] is 0.25
    UQ1_raw  = float(np.var(scores_all))
    UQ1_norm = float(min(1.0, UQ1_raw / 0.25))  # normalize to [0, 1]

    # --- Per-category breakdown (for reporting / debugging) ---
    preserve_scores = [
        nli_scores_map[k]["max_score"]
        for k in C1_KEYS if k in nli_scores_map
    ]
    destroy_scores = [
        nli_scores_map[k]["max_score"]
        for k in C2_KEYS if k in nli_scores_map
    ]

    return {
        "scores_all":           scores_all,          # all 13 NLI scores
        "score_base":           score_base,
        "UQ1_raw":              UQ1_raw,              # raw variance
        "UQ1_norm":             UQ1_norm,             # normalized [0,1] → composite input
        "score_preserve_mean":  float(np.mean(preserve_scores)) if preserve_scores else 0.0,
        "score_destroy_mean":   float(np.mean(destroy_scores))  if destroy_scores  else 0.0,
    }


def interpret_uq1(result: dict) -> str:
    """
    Return a human-readable interpretation of the UQ₁ result.
    Useful for logging and the final report.
    """
    u = result["UQ1_norm"]
    preserve = result["score_preserve_mean"]
    destroy  = result["score_destroy_mean"]

    level = "HIGH" if u >= 0.5 else ("MODERATE" if u >= 0.2 else "STABLE")

    notes = []
    if preserve < 0.4:
        notes.append("Preserve perturbs have low NLI (retrieval may not support this claim at all).")
    if destroy > 0.4:
        notes.append("Destroy perturbs still score high NLI (corpus is not specific enough to distinguish).")

    note_str = " | ".join(notes) if notes else "Scores behave as expected across categories."
    return f"UQ1={u:.4f} [{level}] | preserve_mean={preserve:.3f} | destroy_mean={destroy:.3f} | {note_str}"
