# Report 1: Jev as a paper-reproduction rubric judge on PaperBench JudgeEval

*2026-09-28 · model `jev-1.13.0` (TypeSafe AI) · exploratory, n = 195 human-graded rubric leaves*

## Scope
This report covers **only OpenAI's PaperBench**, specifically its **JudgeEval** subset, which is the part of PaperBench
with human ground-truth grades. It asks one question: how does Jev compare with the LLM judges that OpenAI
evaluated on the same benchmark?

The ResearchPlanGen (Meta) probe is a separate report: [REPORT_ResearchPlanGen.md](REPORT_ResearchPlanGen.md).

## Sources
| What | Link |
|---|---|
| PaperBench paper (Starace et al., OpenAI, 2025) | [arXiv:2504.01848](https://arxiv.org/abs/2504.01848) |
| PaperBench code + data (OpenAI GitHub; formerly `openai/preparedness`) | [openai/frontier-evals/project/paperbench](https://github.com/openai/frontier-evals/tree/main/project/paperbench) |
| JudgeEval data (5 submissions + human-graded rubric trees) | [data/judge_eval/](https://github.com/openai/frontier-evals/tree/main/project/paperbench/data/judge_eval) |
| JudgeEval description | [judge_eval/README.md](https://github.com/openai/frontier-evals/blob/main/project/paperbench/paperbench/judge/judge_eval/README.md) |
| PaperBench scoring code (macro metrics) | [evaluate.py](https://github.com/openai/frontier-evals/blob/main/project/paperbench/paperbench/judge/judge_eval/evaluate.py) · pooling across examples in [run_judge_eval.py](https://github.com/openai/frontier-evals/blob/main/project/paperbench/paperbench/scripts/run_judge_eval.py) |
| PaperBench published judge results (the baselines below) | [experiments/judge_eval/tables/](https://github.com/openai/frontier-evals/tree/main/project/paperbench/experiments/judge_eval/tables) |
| Jev / RLCD paper (TypeSafe AI) | [arXiv:2609.29429](https://arxiv.org/abs/2609.29429) |
| Jev API used here | [docs.typesafe.ai/api](https://docs.typesafe.ai/api) · input limits in [docs.typesafe.ai/models](https://docs.typesafe.ai/models) |

## What PaperBench and JudgeEval are
- **PaperBench** asks AI agents to reproduce 20 ICML 2024 papers from scratch. Each paper has an author-approved
  **rubric tree**: 8,316 weighted leaf requirements across the 20 papers. An LLM judge grades each leaf pass/fail
  against the agent's **submission**, meaning its code repo after `reproduce.sh` has been run (code + `reproduce.log` + outputs).
  Leaf scores roll up to a paper score through weighted averages.
- **JudgeEval** is PaperBench's test of the judge. It has **5 submissions whose every rubric leaf was graded by
  humans** (2,693 leaves in total). OpenAI reports their LLM judges' agreement with these human grades.
  Each example is `submission.tar` plus `grading/expected_result.json`, the rubric tree with a human `score` and
  sometimes an `explanation` on each leaf.

| JudgeEval example | What the submission is | Leaves | Human pass/fail | Text size | Used here |
|---|---|---|---|---|---|
| `semantic-self-consistency` | partial reproduction | 79 | 74 / 5 | ~11k tokens | ✅ |
| `stay-on-topic-with-classifier-free-guidance` | partial reproduction | 116 | 52 / 64 | ~13k tokens | ✅ |
| `rice` | authors' repo ([chengzelei/RICE](https://github.com/chengzelei/RICE)), never run | 361 | 97 / 264 | ~390k tokens | ❌ too large |
| `all-in-one` | authors' repo (Simformer), never run | 174 | 84 / 90 | ~440k tokens | ❌ too large |
| `pinn` | authors' repo + added `reproduce.sh`, run | 1,963 | 882 / 1,081 | ~1.6M tokens | ❌ too large |

## Worked example: one JudgeEval item, end to end
This walks through **`stay-on-topic-with-classifier-free-guidance`**, one of the 5 human-graded examples and one of
the 2 we sent to Jev. Everything below is committed under [examples/](examples/). Each piece links to both our copy
and the original in OpenAI's PaperBench repo.

### The paper being reproduced
**Stay on Topic with Classifier-Free Guidance** (Sanchez, Fan, Spangher, Levi, Ammanamanchi, Biderman; ICML 2024;
[arXiv:2306.17806](https://arxiv.org/abs/2306.17806)).

What it claims, per its abstract: *Classifier-Free Guidance (CFG)*, a trick from text-to-image diffusion, also works for
plain language models at inference time. It up-weights the model's prediction *given the prompt* against its
prediction *without the prompt*, with strength γ. The paper reports that this improves GPT-2, Pythia and LLaMA models
across Q&A, reasoning and code benchmarks (e.g. LLaMA-7B beats PaLM-540B on LAMBADA), and that the gain is roughly
equivalent to doubling model size.

### The pieces and where to find them
| Piece | What it is | This repo | Original (openai/frontier-evals) |
|---|---|---|---|
| **Paper** | The paper to reproduce, as Markdown (what judges read) and PDF | [paper.md](examples/stay-on-topic-paper/paper.md) · [paper.pdf](examples/stay-on-topic-paper/paper.pdf) | [paper.md](https://github.com/openai/frontier-evals/blob/main/project/paperbench/data/papers/stay-on-topic-with-classifier-free-guidance/paper.md) · [paper.pdf](https://github.com/openai/frontier-evals/blob/main/project/paperbench/data/papers/stay-on-topic-with-classifier-free-guidance/paper.pdf) |
| **Addendum** | Clarifications given to the reproducer (typos in the paper, default settings) | [addendum.md](examples/stay-on-topic-paper/addendum.md) | [addendum.md](https://github.com/openai/frontier-evals/blob/main/project/paperbench/data/papers/stay-on-topic-with-classifier-free-guidance/addendum.md) |
| **Judge addendum** | Extra notes given only to the judge (e.g. reference FLOPs code) | [judge.addendum.md](examples/stay-on-topic-paper/judge.addendum.md) | [judge.addendum.md](https://github.com/openai/frontier-evals/blob/main/project/paperbench/data/papers/stay-on-topic-with-classifier-free-guidance/judge.addendum.md) |
| **Blacklist** | URLs the reproducer may not use (the authors' own code) | [blacklist.txt](examples/stay-on-topic-paper/blacklist.txt) | [blacklist.txt](https://github.com/openai/frontier-evals/blob/main/project/paperbench/data/papers/stay-on-topic-with-classifier-free-guidance/blacklist.txt) |
| **Rubric** (ungraded) | The official rubric tree, written with the paper's authors | [rubric.json](examples/stay-on-topic-paper/rubric.json) | [rubric.json](https://github.com/openai/frontier-evals/blob/main/project/paperbench/data/papers/stay-on-topic-with-classifier-free-guidance/rubric.json) |
| **Submission** | The reproduction attempt being graded: code, run script, terminal output, result files | [stay-on-topic-submission/](examples/stay-on-topic-submission/) | [submission.tar](https://github.com/openai/frontier-evals/blob/main/project/paperbench/data/judge_eval/stay-on-topic-with-classifier-free-guidance/0/submission.tar) |
| **Ground truth** | The same rubric tree with a **human** pass/fail `score` (and sometimes an `explanation`) on every leaf | [stay-on-topic-human-grades.json](examples/stay-on-topic-human-grades.json) | [expected_result.json](https://github.com/openai/frontier-evals/blob/main/project/paperbench/data/judge_eval/stay-on-topic-with-classifier-free-guidance/0/grading/expected_result.json) |
| **What we sent to Jev** | The submission's text files concatenated into one `state`, plus the first 40 questions | [state.txt](results/sent_to_jev/stay-on-topic-with-classifier-free-guidance_state.txt) · [request1.json](results/sent_to_jev/stay-on-topic-with-classifier-free-guidance_request1.json) | (ours) |
| **Jev's answers** | One row per rubric leaf: human label next to Jev's P(pass) | [jev_items_judgeeval.csv](results/jev_items_judgeeval.csv) (rows with `paper` = this example) | (ours) |

The JudgeEval graded rubric can differ slightly from the official `rubric.json`, because OpenAI polished the rubrics
after running JudgeEval
([JudgeEval README](https://github.com/openai/frontier-evals/blob/main/project/paperbench/paperbench/judge/judge_eval/README.md)).

### Key facts that are easy to mix up
- **Models in the paper's experiments** are the *subjects* of the research, not the thing doing the reproduction.
  For Section 3.1 these are GPT-2 (4 sizes), Pythia (160M–12B) and LLaMA (7B–65B). Later sections use Guanaco-65B,
  WizardLM-30B, CodeGen and Falcon-7B.
- **Who produced the submission:** the PaperBench team. The PaperBench paper says JudgeEval replications were
  "created either from scratch or by modifying the original author's codebases". This one is a from-scratch
  **partial** reproduction. The paper does not say whether a human or an agent wrote it.
- **What the submission actually ran:** only Section 3.1, and only the `--short` setting: GPT-2 small and medium,
  Pythia-160M and 410M, 6 of the 9 benchmarks, each with γ = 1 (no CFG) and γ = 1.5 (CFG). The Section 3.2 script
  exists but is commented out in `reproduce.sh`.
- **What the submission contains:**
  - code: [cfg.py](examples/stay-on-topic-submission/cfg.py) (the method), [harness.py](examples/stay-on-topic-submission/harness.py) (plugs it into EleutherAI's eval harness), [run_3.1.py](examples/stay-on-topic-submission/run_3.1.py), [run_3.2.py](examples/stay-on-topic-submission/run_3.2.py) (the experiments);
  - the run script [reproduce.sh](examples/stay-on-topic-submission/reproduce.sh) and its terminal output [reproduce.log](examples/stay-on-topic-submission/reproduce.log);
  - result files [results/table5/*.csv](examples/stay-on-topic-submission/results/table5/): accuracy per model at γ = 1 and γ = 1.5.
- **It is a final repo state, not a trajectory.** There are no agent actions or step-by-step history, only what was left behind after `reproduce.sh` ran.
- **The rubric** is a tree of 155 nodes, of which 116 are gradable leaves. Parent scores are weighted averages of their children.
  The human grades roll up to **0.48** overall:
  ```
  [0.48] Reproduce the paper
   ├─ [0.90 w=4] Sec 3.1  CFG boosts zero-shot benchmarks          (mostly done)
   ├─ [0.58 w=4] Sec 3.2  CFG + chain-of-thought on GSM8K/AQuA     (code written, not run)
   ├─ [0.25 w=4] Sec 3.3  CFG on HumanEval code generation         (mostly not done)
   ├─ [0.00 w=1] Sec 4    FLOPs analysis                           (not done)
   ├─ [0.29 w=1] Sec 5    entropy / instruction-tuning analysis    (mostly not done)
   └─ [0.00 w=1] Sec 6    Table 4 toxicity / sentiment             (not done)
  ```

### Three different kinds of "score"
| Layer | What is being scored | Example number |
|---|---|---|
| ① The paper's experiment metric | How accurate GPT-2 is on a benchmark, with and without CFG | ARC-e accuracy 0.491 → 0.515 (in the result CSVs) |
| ② The rubric score | How completely and correctly the submission reproduced the paper | Human-graded 0.48 for this submission |
| ③ The judge metric (this project) | How well a judge's rubric grades agree with the human grades | Jev macro-F1 0.832, AUROC 0.961 |

The rubric judges the **whole submission folder**, using the paper as the reference. Each leaf looks only at the files
relevant to it: source code for Code Development, `reproduce.log` and outputs for Code Execution, and result files
compared with the paper's claims for Result Analysis.

### Three rubric leaves traced through the files
| Rubric leaf (category) | What the paper says | Evidence in the submission | Human | Jev P(pass) |
|---|---|---|---|---|
| "CFG inference is implemented as described in equation 7 … for the GPT2 family" (Code Development) | Eq. 7: log P = log P(uncond) + γ·(log P(cond) − log P(uncond)), [paper.md L106](https://github.com/devchilll/jev-rubrics/blob/main/examples/stay-on-topic-paper/paper.md?plain=1#L106) | [cfg.py L89](examples/stay-on-topic-submission/cfg.py#L89): `adjusted_logits = uncond_log_probs + guidance_scale * (cond_log_probs - uncond_log_probs)` | ✅ pass | 0.88 ✅ |
| "The zero-shot benchmark … has been run on the GPT2 model family with and without CFG (γ=1 and γ=1.5)" (Code Execution) | Evaluate the GPT-2 family, [paper.md L121](https://github.com/devchilll/jev-rubrics/blob/main/examples/stay-on-topic-paper/paper.md?plain=1#L121). Table 5 lists 4 sizes (G-s, G-m, G-l, G-xl), [paper.md L675–678](https://github.com/devchilll/jev-rubrics/blob/main/examples/stay-on-topic-paper/paper.md?plain=1#L675-L678) | Only `--short` was run, [reproduce.sh L20](examples/stay-on-topic-submission/reproduce.sh#L20). `MODELS_SHORT` holds only gpt2 and gpt2-medium, [run_3.1.py L13](examples/stay-on-topic-submission/run_3.1.py#L13), and the log confirms it, [reproduce.log L14](examples/stay-on-topic-submission/reproduce.log#L14) | ❌ fail ("only been run on gpt2 small and gpt medium") | 0.94 ❌ (Jev wrong: it never saw the paper, so it couldn't know the family has 4 sizes) |
| "The measured performance metrics … show that using CFG (γ=1.5) outperforms not using CFG (γ=1)" (Result Analysis) | Table 5: e.g. ARC-e, GPT-2 medium 43.6 → 47.6, [paper.md L676](https://github.com/devchilll/jev-rubrics/blob/main/examples/stay-on-topic-paper/paper.md?plain=1#L676) | [arc_easy_results.csv](examples/stay-on-topic-submission/results/table5/arc_easy_results.csv): gpt2-medium 0.491 → 0.515, also in [reproduce.log L130](examples/stay-on-topic-submission/reproduce.log#L130). The numbers differ from the paper, but the direction matches | ✅ pass | 0.59 ✅ (barely) |

## Method
- **Subset:** only the 2 submissions whose text fits Jev's ~32k-token input (195 leaves, 7% of JudgeEval).
- **Input to Jev:**
  - `state`: all submission text files concatenated. `.git` and lock files are excluded.
    The exact inputs are in [results/sent_to_jev/](results/sent_to_jev/).
  - Questions: one `noul` (yes/no) question per rubric leaf, with the parent requirement and a hint for its
    category (Code Development / Code Execution / Result Analysis).
  - Batching: 40 leaves per call, **5 API calls in total**.
- **Not given to Jev:** the paper text (PaperBench's judge does receive it), and the human grades or explanations,
  which were used only for scoring.
- **Scoring:** same as PaperBench's `evaluate.py`. Leaves are pooled, and we report accuracy plus **macro-averaged**
  precision, recall and F1 (the average over the pass and fail classes), with P(pass) thresholded at 0.5.
  Code: [jev_judgeeval_probe.py](jev_judgeeval_probe.py), [analyze_judgeeval.py](analyze_judgeeval.py).

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

- **Where the numbers come from:** all non-Jev rows are copied from PaperBench's published
  [judge tables](https://github.com/openai/frontier-evals/tree/main/project/paperbench/experiments/judge_eval/tables).
  The per-category columns are macro-F1, and the cost column is from `perf_cost_table.csv`. PaperBench publishes
  only OpenAI models, and no per-submission breakdown.
- **Other Jev metrics:** AUROC **0.961** overall (0.942 and 0.962 per submission). Per submission at t = 0.5,
  macro-F1 is 0.742 and 0.766. Brier score 0.134, ECE 0.178.

## What we see
1. **Jev ranks leaves almost perfectly (AUROC 0.96)** and its macro-F1 (0.83) is on par with o3-mini and o1.
   It gets there with no paper context, no prompt tuning and no training, in about a second.
2. **Jev is too strict, and that is its main error mode.** 8 of its 10 largest errors are human-pass items that
   Jev scored ≤ 0.12. An example is "code for acquiring dataset X has been implemented" when the submission loads X
   through the EleutherAI eval harness. Human graders accept that as an "equivalent" source, and Jev looks for
   explicit download code. Pass-class precision is 0.97 and pass-class recall is 0.77.
3. **Missing paper context causes the false passes.** Example: "benchmark run on the GPT-2 model family". The submission
   ran only 2 sizes, and you need the paper to know the family is larger.
4. **Threshold tuning helps but is unstable at this sample size.** Tuning on one submission and testing on the
   other gives a pooled macro-F1 of 0.878. However, the threshold tuned on `stay-on-topic` (t = 0.05) collapses
   to all-pass on the 94%-pass submission (macro-F1 0.48).

## Caveats (read before citing any number)
- **This is not the same test set.** PaperBench's numbers pool all 5 submissions, and 73% of their leaves come from
  `pinn`. Ours come from the 2 smallest submissions. Treat the comparison as indicative only.
- n = 195. Result Analysis has only 22 leaves, of which 2 are passes. One submission is 94% pass.
- JudgeEval submissions are **final repo states, not agent trajectories**. Nothing here tests judging step-by-step
  agent behaviour (e.g. reward hacking visible only in the actions).
- Jev is closed-weight and was called via API (`jev-latest` resolved to `jev-1.13.0`).

## Next steps
1. **Same-subset baseline:** run an open LLM judge (PaperBench's prompt) on these 195 leaves for a like-for-like comparison.
2. **Full JudgeEval coverage:** retrieve only the relevant files per rubric leaf so `rice`, `all-in-one` and `pinn`
   fit in 32k tokens. This gives a number directly comparable to 0.827 and 0.842.
3. **Paper context:** add the relevant paper section per leaf. PaperBench's `paper.md` is ~30k tokens, which is too large to send whole.
4. **Calibration:** fix the strictness and ECE on held-out data. This motivates training an open RLCD-style judge.
