"""
uq_retrieval.py
===============
UQ₂ — Retrieval Uncertainty

What this measures:
    Do perturbations retrieve the SAME documents as the original claim?
    A claim may always score high on NLI — but from completely different
    documents across queries. That is retrieval uncertainty: the claim
    has no stable, phrase-independent evidence anchor.

Two metrics:
    Jaccard Similarity  — set-level overlap (no rank awareness)
                          J(A,B) = |A∩B| / |A∪B|
                          → stored as DIAGNOSTIC only

    Rank-Biased Overlap — rank-aware overlap (Webber et al., ACM TOIS 2010)
                          RBO(S,T,p) = (1-p) * Σ p^(d-1) * |S[:d]∩T[:d]|/d
                          p=0.9 → top-10 positions carry ~86% of total weight
                          → PRIMARY signal for UQ₂

Why not Jaccard as primary:
    Jaccard treats rank-1 and rank-5 as equal. If the same doc moves from
    rank-1 to rank-5, Jaccard sees perfect overlap but attribution quality
    has changed significantly. RBO is rank-sensitive and more appropriate
    for retrieval evaluation.

Final signal:
    UQ2_norm (Instability) = 1 − mean(RBO over all 12 perturbation queries)

Inputs (from main.py):
    docs_map : dict mapping variant_key → list of doc dicts from bm25_retrieve
               e.g. {"base": [{id,text,score},...], "C1_paraphrase": [...], ...}

Output:
    dict with per-perturbation Jaccard and RBO scores, mean values, UQ2_norm.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# JACCARD SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

def jaccard_similarity(ids_a: list, ids_b: list) -> float:
    """
    Set-level Jaccard similarity between two lists of document IDs.

    J(A, B) = |A ∩ B| / |A ∪ B|
    Range: [0, 1].  1.0 = identical sets.  0.0 = no overlap.

    Stored as diagnostic. NOT used as primary UQ₂ signal.
    """
    a, b = set(ids_a), set(ids_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ─────────────────────────────────────────────────────────────────────────────
# RANK-BIASED OVERLAP
# ─────────────────────────────────────────────────────────────────────────────

def rank_biased_overlap(list_a: list, list_b: list, p: float = 0.9) -> float:
    """
    Rank-Biased Overlap (RBO) between two ranked lists of document IDs.
    Reference: Webber, Moffat & Zobel (2010), ACM TOIS.

    Formula:
        RBO(S, T, p) = (1-p) * Σ_{d=1}^{D} p^{d-1} * |S[:d] ∩ T[:d]| / d

    Parameters:
        list_a, list_b : ordered lists of document IDs (rank-1 first)
        p              : persistence parameter. p=0.9 (standard) gives
                         ~86% weight to top-10 positions.

    Range: [0, 1].  1.0 = identical ranked lists.  0.0 = completely disjoint.

    Key property: top-ranked agreement matters MORE than bottom-ranked.
    This is why RBO is chosen over Jaccard as the primary retrieval signal.
    """
    if not list_a or not list_b:
        return 0.0

    depth   = min(len(list_a), len(list_b), 20)  # cap depth at 20
    rbo_sum = 0.0

    for d in range(1, depth + 1):
        # Overlap at depth d = fraction of top-d in both lists that match
        overlap  = len(set(list_a[:d]) & set(list_b[:d]))
        rbo_sum += (p ** (d - 1)) * (overlap / d)

    return (1 - p) * rbo_sum


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UQ₂ FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_uq_retrieval(docs_map: dict) -> dict:
    """
    Compute UQ₂ — Retrieval Uncertainty.

    Parameters
    ----------
    docs_map : dict
        Keys: "base" + all 12 perturbation keys.
        Values: list of doc dicts [{"id": str, "text": str, "score": float}, ...]

    Returns
    -------
    dict:
        jaccard_scores  : dict  {pert_key → jaccard score}      — diagnostic
        rbo_scores      : dict  {pert_key → RBO score}          — primary
        mean_jaccard    : float — mean Jaccard across 12 perturbs
        mean_rbo        : float — mean RBO across 12 perturbs
        UQ2_norm        : float — Instability = 1 − mean_rbo ∈ [0, 1]
    """
    base_docs = docs_map["base"]
    base_ids  = [d["id"] for d in base_docs]

    jaccard_scores = {}
    rbo_scores     = {}

    for key, p_docs in docs_map.items():
        if key == "base":
            continue
        p_ids = [d["id"] for d in p_docs]
        jaccard_scores[key] = jaccard_similarity(base_ids, p_ids)
        rbo_scores[key]     = rank_biased_overlap(base_ids, p_ids, p=0.9)

    mean_jaccard = float(np.mean(list(jaccard_scores.values()))) if jaccard_scores else 1.0
    mean_rbo     = float(np.mean(list(rbo_scores.values())))     if rbo_scores     else 1.0

    return {
        "jaccard_scores": jaccard_scores,  # dict per perturbation key
        "rbo_scores":     rbo_scores,      # dict per perturbation key
        "mean_jaccard":   mean_jaccard,    # diagnostic summary
        "mean_rbo":       mean_rbo,        # used to compute UQ2_norm
        "UQ2_norm":       1.0 - mean_rbo,  # PRIMARY signal: Instability
    }


def interpret_uq2(result: dict) -> str:
    """
    Return a human-readable interpretation of the UQ₂ result.
    """
    u      = result["UQ2_norm"]
    rbo    = result["mean_rbo"]
    jac    = result["mean_jaccard"]
    level  = "HIGH" if u >= 0.5 else ("MODERATE" if u >= 0.2 else "STABLE")

    # Find the most and least stable perturbations
    rbo_items = sorted(result["rbo_scores"].items(), key=lambda x: x[1])
    worst_key = rbo_items[0][0]  if rbo_items else "N/A"
    worst_rbo = rbo_items[0][1]  if rbo_items else 0.0

    return (
        f"UQ2={u:.4f} [{level}] | mean_RBO={rbo:.3f} | mean_Jaccard={jac:.3f} "
        f"| most_unstable_perturb={worst_key} (RBO={worst_rbo:.3f})"
    )
