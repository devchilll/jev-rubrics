# Jev as a paper-reproduction rubric judge: first look on PaperBench JudgeEval

*2026-09-28 · model `jev-1.13.0` (TypeSafe AI) · exploratory, n = 195 rubric leaves*

## Question
Can a small calibrated "System One" decision model (Jev) grade paper-reproduction submissions against
fine-grained rubrics about as well as the frontier LLM judges used by OpenAI's PaperBench, at a fraction of the cost?

## Setup
- **Data:** [PaperBench JudgeEval](https://github.com/openai/frontier-evals/tree/main/project/paperbench/data/judge_eval).
  It has 5 reproduction submissions (code repo + `reproduce.log` + outputs, after `reproduce.sh` has run),
  with every rubric leaf graded pass/fail by humans (2,693 leaves in total).
- **Subset used:** the only 2 submissions whose text fits Jev's ~32k-token input:
  `semantic-self-consistency` (79 leaves, 94% pass) and `stay-on-topic-with-classifier-free-guidance`
  (116 leaves, 45% pass). The other 3 (`rice`, `all-in-one`, `pinn`) are 0.4M–1.6M tokens and were skipped.
- **Input to Jev:** `state` = all submission text files concatenated; one `noul` (yes/no) question per rubric
  leaf, with the parent requirement and a category hint. 40 leaves per call, **5 calls total**.
  **The paper text was not provided** (PaperBench's judge does get it).
- **Scoring:** identical to PaperBench's `evaluate.py`: pooled leaves, accuracy + **macro**-averaged
  precision/recall/F1, P(pass) thresholded at 0.5.

## Results

| Judge | Leaves | Accuracy | Macro-P | Macro-R | **Macro-F1** | Code Dev | Code Exec | Result Analysis | Cost |
|---|---|---|---|---|---|---|---|---|---|
| **Jev 1.13** (2 of 5 subs, t = 0.5) | 195 | 0.836 | 0.832 | 0.863 | **0.832** | 0.742 | **0.925** | 0.725 | 107k input tokens, 1.2 s total |
| o1 (high) | 2,693 | 0.844 | 0.842 | 0.841 | **0.842** | 0.740 | 0.840 | 0.876 | $830 / paper |
| o3-mini (high) | 2,693 | 0.827 | 0.827 | 0.831 | **0.827** | 0.720 | 0.822 | 0.945 | $66 / paper |
| o1-mini (high) | 2,693 | 0.807 | 0.848 | 0.761 | 0.775 | 0.669 | 0.741 | 0.798 | $72 / paper |
| gpt-4o | 2,693 | 0.740 | 0.744 | 0.725 | 0.728 | 0.681 | 0.707 | 0.825 | $120 / paper |
| gpt-4o-mini | 2,693 | 0.632 | 0.640 | 0.603 | 0.590 | 0.588 | 0.544 | 0.778 | $8 / paper |
| random | 2,693 | — | — | — | 0.504 | 0.469 | 0.495 | 0.487 | — |

The PaperBench rows come from `experiments/judge_eval/tables/` in openai/frontier-evals. Jev's AUROC is **0.961** overall
(0.942 and 0.962 per submission). Per submission at t = 0.5, macro-F1 is 0.742 and 0.766.

## What we see
1. **Jev ranks leaves almost perfectly (AUROC 0.96)** and its macro-F1 (0.83) is on par with o3-mini and o1.
   It gets there with no paper context, no prompt tuning and no training, in about a second.
2. **Jev is too strict, and that is its main error mode.** 8 of its 10 largest errors are human-pass items that
   Jev scored ≤ 0.12. An example is "code for acquiring dataset X has been implemented" when the submission loads X
   through the EleutherAI eval harness. Human graders accept that as an "equivalent" source, and Jev looks for
   explicit download code. Pass-class precision is 0.97 and pass-class recall is 0.77. Calibration error (ECE) is 0.18.
3. **Missing paper context causes the false passes.** Example: "benchmark run on the GPT-2 model family". The submission
   ran only 2 sizes, and you need the paper to know the family is larger.
4. **Threshold tuning helps but is unstable at this sample size.** Tuning on one submission and testing on the
   other gives a pooled macro-F1 of 0.878. However, the threshold tuned on `stay-on-topic` (t = 0.05) collapses
   to all-pass on the 94%-pass submission (macro-F1 0.48).

## Caveats (read before citing any number)
- **This is not the same test set.** PaperBench's numbers pool all 5 submissions, and 73% of their leaves come from
  `pinn`. Ours come from the 2 smallest submissions. Treat the comparison as indicative only.
- n = 195. Result Analysis has only 22 leaves, of which 2 are passes. One submission is 94% pass.
- Jev is closed-weight and was called via API (`jev-latest` resolved to `jev-1.13.0`).
- Jev never saw the human labels or explanations; those were used only for scoring.

## Next steps
1. **Same-subset baseline:** run an open LLM judge (PaperBench's prompt) on these 195 leaves for a like-for-like comparison.
2. **Full JudgeEval coverage:** retrieve only the relevant files per rubric leaf so `rice`, `all-in-one` and `pinn`
   fit in 32k tokens. This gives a number directly comparable to 0.827 and 0.842.
3. **Paper context:** add the relevant paper section per leaf.
4. **Calibration:** fix the strictness and ECE on held-out data. This motivates training an open RLCD-style judge.

## Also in this repo: ResearchPlanGen sanity probe
Jev was also run on 20 ResearchPlanGen ML-test goals (reference solution as state, 10 rubric items each).
Mean P(met) was 0.88, with 95% of items ≥ 0.5, which fits the fact that the references are meant to satisfy their rubrics.
The lowest-scored items were real gaps in the references. That dataset has no ground-truth labels,
so this run is a sanity check only (`results/jev_items_ml_test_n20.csv`).
