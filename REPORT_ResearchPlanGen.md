# Report 2: Jev on ResearchPlanGen (Meta), a sanity probe without ground truth

*2026-09-28 · model `jev-1.13.0` (TypeSafe AI) · exploratory, 20 examples / 200 rubric items*

## Scope
This report covers **only Meta's ResearchPlanGen dataset**. It was the first, cheapest check of whether a Jev-style
classifier behaves sensibly when asked "is this rubric item met by this text?". **The dataset has no ground-truth
labels**, so nothing here measures accuracy. It is a sanity probe, not an evaluation.

The PaperBench (OpenAI) evaluation, which does have human labels, is a separate report:
[REPORT_PaperBench_JudgeEval.md](REPORT_PaperBench_JudgeEval.md).

## Sources
| What | Link |
|---|---|
| Paper: *Training AI Co-Scientists Using Rubric Rewards* (Goel et al., Meta Superintelligence Labs, 2025) | [arXiv:2512.23707](https://arxiv.org/abs/2512.23707) |
| Dataset: ResearchPlanGen (`facebook/research-plan-gen`) | [huggingface.co/datasets/facebook/research-plan-gen](https://huggingface.co/datasets/facebook/research-plan-gen) |
| Split used (ML, test) | [ml/test/data.parquet](https://huggingface.co/datasets/facebook/research-plan-gen/blob/main/ml/test/data.parquet) |
| Jev / RLCD paper (TypeSafe AI) | [arXiv:2609.29429](https://arxiv.org/abs/2609.29429) |
| Jev API used here | [docs.typesafe.ai/api](https://docs.typesafe.ai/api) |

## What the dataset is
The Meta paper trains LLMs to write **research plans** with RL, rewarding them by how many **paper-specific rubric
items** the plan satisfies. Llama-4 extracted the goals, rubrics and reference solutions automatically from papers.
The released data has 22.5k tasks across three subsets: ML (7.6k), arXiv (8.1k) and PubMed (6.9k). It is CC-BY-NC.

| Column | Meaning |
|---|---|
| `Goal` | The research problem, phrased as an instruction (~90 words) |
| `Rubric` | **10 yes/no criteria** a good plan should meet, derived from what the paper did (~15 words each, unweighted) |
| `Reference solution` | Llama-4's ~550-word summary of how the authors solved the goal, written as a plan |
| `article_id`, `q_id`, `Identifier` | Paper ID, task ID, public paper ID (OpenReview / arXiv / PMID) |
| `Subdomain`, `Category` | Field and topic (filled in only for some subsets) |

**No ground truth:** nothing in the data says whether the reference solution meets each rubric item. The rubric and
reference come from the same paper, so the reference is *meant* to satisfy the rubric, but nobody verified it item by item.

Relevant findings from the paper itself:
- Their grader is a frozen Qwen3-30B. A rubric item counts as satisfied only if none of 7 general guidelines are violated.
- Human–LLM-jury agreement on this kind of rubric was only **Cohen's κ ≈ 0.30**.
- The grader was **exploited during RL**: self-grades kept rising while a stronger held-out judge disagreed after about 120 steps.

## Method
- **Examples:** the first 20 rows of the ML test split, 200 rubric items in total.
- **Input to Jev:** `state` = the reference solution; one `noul` question per rubric item:
  *"Does the plan satisfy the following rubric criterion? Answer yes only if the plan actually addresses it,
  not just mentions related words."* That is one API call per example, 20 in total.
- **Assumed label:** every item is treated as a **presumed "yes"**, so a low P(met) means Jev disagrees with that presumption.
- Code: [jev_rpg_probe.py](jev_rpg_probe.py). Output: [results/jev_items_ml_test_n20.csv](results/jev_items_ml_test_n20.csv).

## Results
| | |
|---|---|
| Mean / median P(met) | **0.88 / 0.94** |
| Items with P(met) ≥ 0.5 | 190 / 200 (95%) |
| Items with P(met) < 0.8 | 34 / 200 |
| Per-example expected score (sum of 10 P's) | min 7.58 · median 8.91 · max 9.87 |
| Latency | ~0.15 s per example (10 questions in one call) |

**Were Jev's "no"s real gaps?** We read the reference solutions behind the 5 lowest-scored items:

| P(met) | Rubric item | Reference solution | Verdict |
|---|---|---|---|
| 0.06 | Captions analysed for word count / frequency | Never mentions caption statistics | Real gap |
| 0.20 | Visualise domain specialisation **across layers** | Visualises per-expert routing, not across layers | Mostly a real gap |
| 0.28 | Evaluate quality of generated affordance maps | No evaluation of map quality | Real gap |
| 0.32 | Includes examples / case studies of dark patterns | No examples | Real gap |
| 0.39 | Scalable to **deeper search trees** | Only a generic "scalable" claim | Arguable |

In one example we checked by hand (Learning-to-Defer, `zl0HLZOJC9`), we had marked items 3, 4 and 9 as only
vaguely addressed. Jev gave items 3 and 4 its lowest scores for that example (0.87 and 0.80). It gave item 9
a score of 0.90.

## Two warnings this probe surfaced
1. **Word matching could explain some of the scores.** P(met) correlates with plain word overlap between the rubric
   item and the reference (r = 0.40): items with ≤ 50% overlap average 0.72, and items with > 85% overlap average 0.93.
2. **The references copy the rubric's wording.** Llama-4 wrote both in one pass, and references often restate rubric items
   nearly word for word (e.g. *"avoids the need for multiple attempts to fine-tune hyperparameters"*). A classifier trained on
   (reference, item) → yes can therefore score well by string matching, which would not transfer to real plans or trajectories.

## What this probe can and cannot tell us
- **Can:** Jev's behaviour is sensible and fast. Its low scores mostly point at real omissions in the references.
- **Cannot:** measure accuracy, false-positive rate or calibration. Every input is a presumed positive and there are no labels.

## If this dataset is used further
1. **Negatives:**
   - Easy: pair a rubric item with a reference from a different paper in the same `Category`.
   - Hard: have an LLM rewrite the reference to remove item k's *idea*, not just its wording.
2. **Honest test set:** have an open model write plans from `Goal` alone, without seeing the rubric, then hand-label a few hundred (plan, item) pairs.
3. **Split by `article_id`** to avoid leakage between train and test.
4. **Licensing:** CC-BY-NC. Per the dataset card, models trained on it must include "Llama" in their name.
