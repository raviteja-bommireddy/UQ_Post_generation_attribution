# UQ in Post-Generation Attribution

## Step-by-Step Flow

**Step 1 — Setup**
Install dependencies: `pip install -r requirements.txt`. Set `OPENROUTER_API_KEY` in `.env`.

**Step 2 — Load Data & Build Index**
Load 17 ASQA questions. Extract corpus from ASQA annotations + ALCE Wikipedia. Build BM25 index over all passages (`pipeline_utils.py`).

**Step 3 — Generate Answer**
LLM (`llama-3.1-70b`) generates a factual answer per question.

**Step 4 — Extract Atomic Claims**
LLM decomposes each answer into minimal, independently verifiable facts.

**Step 5 — Generate 12 Perturbations per Claim**
5 categories: C1 Preserve (×2), C2 Destroy (×4), C3 Specificity (×2), C4 Breadth (×2), C5 Depth (×2).

**Step 6 — Retrieve Docs & NLI Score**
For each claim + 12 perturbations, retrieve top-5 docs (BM25). Run NLI cross-encoder → entailment scores.

**Step 7 — Compute UQ₁ (Attribution)**  `uq_attribution.py` — NLI variance across 13 variants.

**Step 8 — Compute UQ₂ (Retrieval)** `uq_retrieval.py` — RBO instability across 12 perturbation doc sets.

**Step 9 — Compute UQ₃ (Content)** `uq_content.py` — avg(Attribution Drop, Specification Entailment Gap). Skipped if `w3=0`.

**Step 10 — Compute UQ₄ (Claim Extraction)** `uq_claim_extraction.py` — diagnostic only, not in composite.

**Step 11 — Composite Score**
`U_attr = w1*UQ1 + w2*UQ2 + w3*UQ3` with user-supplied weights at runtime.

## Run
```bash
python main.py --num_questions 3 --w1 0.4 --w2 0.35 --w3 0.25
```
