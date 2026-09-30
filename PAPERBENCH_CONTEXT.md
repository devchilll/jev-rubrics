# PaperBench × Jev: full project context

*Last updated 2026-09-30. Repo: [github.com/devchilll/jev-rubrics](https://github.com/devchilll/jev-rubrics)*

This file is the handoff document for rebuilding and continuing the **PaperBench** part of this project on a fresh
machine (e.g. a cloud VM). It collects the goal, background, data, exact method, results, repo layout, tested setup
commands, the related-dataset search, and open next steps.

Scope: **PaperBench only.** The ResearchPlanGen (Meta) sanity probe also lives in this repo
([REPORT_ResearchPlanGen.md](REPORT_ResearchPlanGen.md)) but is out of scope here. The one exception is that
`jev_rpg_probe.py` provides the `call_jev()` helper that the PaperBench probe imports.

---

## 1. Project goal

- **Main question:** can a small, fast classifier in the style of TypeSafe's **Jev** grade AI paper-reproduction work
  against fine-grained rubrics with high accuracy?
- **How it would work:**
  - Given a reproduction attempt and a list of rubric items, the model outputs P(item is met) for each item.
  - Rubric score = Σ P(met) × item points (PaperBench does this as a weighted roll-up of a rubric tree).
- **Later steps in the original plan:**
  - Curate (attempt, rubric, ground-truth) data.
  - Synthetically perturb attempts to flip labels, e.g. inject reward hacking where there was none.
  - Train or convert an open model into a Jev-style classifier.
  - Compare it against Jev, DiffusionGemma-based decision models (djev) and LLM judges.
- **Long-term use:** as a **reward model** in RL for paper reproduction or any rubric-judged task. In standard RL
  infrastructure the reward model is trained first and then **frozen**, so it must resist being exploited by the policy.
- **Constraints:** public data and open models wherever possible. The intent is to publish a paper.

## 2. Background

### Jev (TypeSafe AI)
- A "System One" decision model, released September 2026. Weights are **closed**; it is API-only.
- **Input and output:** it takes a `state` (the content to judge) plus a map of typed questions:
  - `noul` (yes/no), `choice` or `score`;
  - it returns **probabilities**, never text;
  - the state is read once and shared across all questions, which makes many questions per call cheap.
- **Training:** "RLCD" (Reinforcement Learning for Calibrated Decisions). The reward depends on how calibrated the
  stated probabilities are; people outside TypeSafe believe it is a proper scoring rule such as Brier or log score.
  TypeSafe says the training data is 100% synthetic. Architecture, size and base model are undisclosed.
- **Input limit:** about **32k tokens** for the state plus the longest question, and 64k for the state plus all questions.
- **API:** `POST https://api.typesafe.ai/v1/systemone`, header `Authorization: Bearer $TYPESAFE_API_KEY`.
  A `noul` answer comes back as `answers[<id>].noul` ∈ [0, 1].
- **Links:** [Jev/RLCD paper arXiv:2609.29429](https://arxiv.org/abs/2609.29429) ·
  [API docs](https://docs.typesafe.ai/api) · [models / limits](https://docs.typesafe.ai/models)
- **Open analogues:** OpenJev/Verdict ([heman10x/rlcd-modernbert-151m](https://huggingface.co/heman10x/rlcd-modernbert-151m))
  and djev on DiffusionGemma ([write-up](https://glaforge.dev/posts/2026/09/24/multimodal-decision-models-with-diffusiongemma-jev-and-langchain4j/)).

### PaperBench (OpenAI)
- **Paper:** [PaperBench, arXiv:2504.01848](https://arxiv.org/abs/2504.01848) (Starace et al., 2025).
- **Code + data:** [openai/frontier-evals/project/paperbench](https://github.com/openai/frontier-evals/tree/main/project/paperbench).
  The repo was formerly `openai/preparedness`, and old links redirect.
- **The task:** AI agents reproduce **20 ICML 2024 papers** from scratch. Each paper has an **author-approved rubric
  tree**, 8,316 weighted leaf requirements across the 20 papers.
- **What an agent produces:** a `submission/` repo containing code plus `reproduce.sh`. PaperBench runs
  `reproduce.sh` in a fresh container, which yields `reproduce.log` and output files.
- **The judge:** an LLM (o3-mini in the paper) grades every leaf pass/fail. It receives the paper Markdown, the judge
  addendum, the rubric and the relevant submission files. Leaf scores roll up into a paper score through weighted averages.
- **Leaf categories:**
  - **Code Development:** is it implemented correctly?
  - **Code Execution:** was it actually run?
  - **Result Analysis:** do the outputs support the paper's claim?

#### Per-paper files: `data/papers/<paper-id>/` ([browse](https://github.com/openai/frontier-evals/tree/main/project/paperbench/data/papers))
| File | Purpose |
|---|---|
| `paper.md`, `paper.pdf`, `assets/` | The paper. `paper.md` is ~30k tokens for the CFG paper, which is too big to send to Jev whole |
| `addendum.md` | Clarifications for the reproducing agent |
| `judge.addendum.md` | Extra notes for the judge only |
| `blacklist.txt` | URLs the agent may not use (the authors' code) |
| `rubric.json` | Ungraded rubric tree. Each node has `id`, `requirements`, `weight`, `sub_tasks`, `task_category` |
| `config.yaml` | id + title |

No submissions ship for the 20 main papers. They only exist once someone runs agents on the tasks.

#### JudgeEval: `data/judge_eval/<paper-id>/0/` ([browse](https://github.com/openai/frontier-evals/tree/main/project/paperbench/data/judge_eval), [README](https://github.com/openai/frontier-evals/blob/main/project/paperbench/paperbench/judge/judge_eval/README.md))
- **What it is:** PaperBench's test of the *judge*. It has 5 submissions where **humans graded every rubric leaf**.
- **Files per example:**
  - `submission.tar`, stored in Git LFS: the repo in its state **after** `reproduce.sh` ran;
  - `grading/expected_result.json`: the rubric tree with a human `score` (0/1) on each leaf and sometimes an `explanation`.
- **Who made the submissions:** the PaperBench team, "created either from scratch or by modifying the original
  author's codebases". The paper does not say whether the from-scratch ones were written by a human or an agent.
- **They are final repo states, not agent trajectories.** No step-by-step actions are included.
- **Rubric versions:** the JudgeEval rubric can differ slightly from the official `rubric.json`, which OpenAI polished later.
- **Missing tar:** `rice` has no tar because it is not redistributed. It is cloned from
  [chengzelei/RICE](https://github.com/chengzelei/RICE) at commit `4962aeb46d6000ebdf5ecbc5b46fff0d0505da59`.

| Example | Submission | Leaves | Human pass / fail | Human score | Text size | Fits Jev? |
|---|---|---|---|---|---|---|
| `semantic-self-consistency` | partial reproduction, 440 KB | 79 | 74 / 5 | 0.91 | ~11k tokens | ✅ |
| `stay-on-topic-with-classifier-free-guidance` | partial reproduction, 412 KB | 116 | 52 / 64 | 0.48 | ~13k tokens | ✅ |
| `rice` | authors' repo, not run, 29 MB | 361 | 97 / 264 | 0.19 | ~390k tokens | ❌ |
| `all-in-one` | authors' repo (Simformer), not run, 212 MB | 174 | 84 / 90 | 0.71 | ~440k tokens | ❌ |
| `pinn` | authors' repo + added `reproduce.sh`, run, 104 MB | 1,963 | 882 / 1,081 | 0.83 | ~1.6M tokens | ❌ |
| **Total** | | **2,693** | 1,189 / 1,504 | | | |

#### How PaperBench scores judges
- **Code:** [evaluate.py](https://github.com/openai/frontier-evals/blob/main/project/paperbench/paperbench/judge/judge_eval/evaluate.py)
  and [run_judge_eval.py](https://github.com/openai/frontier-evals/blob/main/project/paperbench/paperbench/scripts/run_judge_eval.py).
- **Method:** pool all leaves from all 5 examples, then compute sklearn accuracy plus **macro-averaged**
  precision, recall and F1 (the average over the pass and fail classes), overall and per category.
- **Consequence:** `pinn` supplies 73% of the pooled leaves, so it dominates their numbers.

Published results ([tables](https://github.com/openai/frontier-evals/tree/main/project/paperbench/experiments/judge_eval/tables)):

| Judge | Accuracy | Macro-P | Macro-R | Macro-F1 | Code Dev F1 | Code Exec F1 | Result Analysis F1 | $/paper |
|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini | 0.632 | 0.640 | 0.603 | 0.590 | 0.588 | 0.544 | 0.778 | 8.01 |
| gpt-4o | 0.740 | 0.744 | 0.725 | 0.728 | 0.681 | 0.707 | 0.825 | 119.87 |
| o1-mini (high) | 0.807 | 0.848 | 0.761 | 0.775 | 0.669 | 0.741 | 0.798 | 71.72 |
| o1 (high) | 0.844 | 0.842 | 0.841 | 0.842 | 0.740 | 0.840 | 0.876 | 830.16 |
| o3-mini (high) | 0.827 | 0.827 | 0.831 | 0.827 | 0.720 | 0.822 | 0.945 | 66.04 |
| random | — | — | — | 0.504 | 0.469 | 0.495 | 0.487 | — |

These are OpenAI models only, and there is no per-submission breakdown.

### Three kinds of "score" (the most common source of confusion)
| Layer | What is scored | Example |
|---|---|---|
| ① The paper's experiment metric | How well the paper's *subject models* do on a task | GPT-2 medium ARC-e accuracy 0.491 → 0.515 with CFG |
| ② The rubric score | How completely and correctly the submission reproduced the paper | Humans gave the CFG submission 0.48 |
| ③ The judge metric (this project) | How well a judge's rubric grades agree with the human grades | Jev macro-F1 0.832 |

The rubric grades the whole submission folder against the paper. Each leaf looks only at the relevant files:
code for Code Development, `reproduce.log` and outputs for Code Execution, and result files versus the paper's
claims for Result Analysis.

### Worked example
[REPORT_PaperBench_JudgeEval.md § Worked example](REPORT_PaperBench_JudgeEval.md#worked-example-one-judgeeval-item-end-to-end)
walks through `stay-on-topic-with-classifier-free-guidance`:
- the paper *Stay on Topic with Classifier-Free Guidance* ([arXiv:2306.17806](https://arxiv.org/abs/2306.17806));
- links to every file both in this repo (`examples/`) and in the original PaperBench repo;
- the rubric tree with its human scores;
- three rubric leaves traced through exact lines in the paper and the submission, with the human label and Jev's P(pass) for each.

---

## 3. What was done: chronological log of requests and actions

1. **Found PaperBench** while looking for paper-reproduction data with ground truth.
   ResearchPlanGen, the initial candidate, has no labels and isn't reproduction.
2. **Downloaded JudgeEval** and measured each submission against Jev's 32k-token limit. Only the 2 small ones fit.
3. **Wrote [jev_judgeeval_probe.py](jev_judgeeval_probe.py)** and ran Jev on those 2 submissions (195 leaves) in **5 API calls**.
   Model `jev-latest` resolved to `jev-1.13.0`. Total: 107,388 input tokens, 3,480 output tokens, 1.21 s of API time.
4. **Error analysis:** read the human explanations for Jev's largest disagreements (see §5).
5. **Wrote [analyze_judgeeval.py](analyze_judgeeval.py)** to score Jev exactly as PaperBench scores its judges
   (pooled leaves, macro metrics), plus a cross-fit threshold check. Transcribed PaperBench's published tables into it.
6. **Wrote [REPORT_PaperBench_JudgeEval.md](REPORT_PaperBench_JudgeEval.md):** scope, sources, results table against
   PaperBench's judges, findings, caveats and next steps.
7. **Created the GitHub repo** `devchilll/jev-rubrics` and pushed. It was later made **public** by the owner.
8. **Added explicit scope and source links** to the report, and split the ResearchPlanGen material into its own report.
9. **Added a worked example:** the `examples/` folder holds the CFG paper files, the extracted submission and the human
   grades, and the report gained a walkthrough section linking both our copies and the PaperBench originals.
10. **Searched for similar datasets** (§7).
11. **Pending:** invite `vsubhashini@google.com` as a collaborator. GitHub needs a **username**, not an email.
    Candidates found ([vsubhashini](https://github.com/vsubhashini), [vsubhashini-rgb](https://github.com/vsubhashini-rgb))
    are unconfirmed. Once confirmed:
    `gh api -X PUT repos/devchilll/jev-rubrics/collaborators/USERNAME -f permission=push`

## 4. Method details (exact, for reproduction)

**State construction** (`build_state` in `jev_judgeeval_probe.py`):
- **Included:** every file in `submission/` with suffix `.py .sh .md .txt .csv .log .json .jsonl .yaml .yml .toml .cfg .ini .ipynb`.
- **Skipped:** anything under `.git/`, plus `uv.lock`, `poetry.lock`, `package-lock.json`, `.DS_Store`.
- **Order:** `README.md` first, `reproduce.sh` second, other files alphabetically, `reproduce.log` last.
- **Format:** each file under a header `===== FILE: <path> =====`. Any file over 30,000 chars is truncated in the middle.
- **Size guard:** states over 100,000 chars (~32k tokens) are refused.
- **Result:** 37,797 chars for `semantic-self-consistency` and 45,496 for `stay-on-topic`. The exact states are in [results/sent_to_jev/](results/sent_to_jev/).

**Questions:**
- One `noul` question per rubric leaf, 40 leaves per request, with question keys `q0…q39`.
- Text template:
  ```
  The state is a code submission attempting to reproduce a research paper: its source files, plus reproduce.log
  from running reproduce.sh, plus any output files. Grade one rubric requirement.
  {category hint}

  Parent requirement: {immediate parent node's requirement}
  Requirement: {leaf requirement}
  ```
- Category hints:
  - Code Development: *"Pass if the submission's code contains a correct implementation of the requirement. It does not need to have been run."*
  - Code Execution: *"Pass only if the code was actually run (see reproduce.sh, reproduce.log and output files) and that execution carried out the requirement."*
  - Result Analysis: *"Pass only if the produced outputs/logs show the stated result."*

**Not given to Jev:** the paper (`paper.md`), the addenda, the human grades and the human explanations.

**Scoring:**
- Verdict = `p_pass >= 0.5`, compared with the `human` label (1 = pass).
- Macro metrics follow PaperBench. We also report AUROC, Brier score and ECE (10 bins).
- Tree score = weighted roll-up of leaf values, using either P(pass) or thresholded verdicts.
- Cross-fit: choose the threshold that maximizes macro-F1 on one submission and test it on the other.

**Output CSV** [results/jev_items_judgeeval.csv](results/jev_items_judgeeval.csv), one row per leaf:
`paper, leaf_id, category, weight, parent, requirement, human, human_explanation, p_pass, model`.
Jev's answer is `p_pass`, and the ground truth is `human`.

## 5. Results

| Slice | n | Pass rate | Accuracy | Macro-P | Macro-R | Macro-F1 | AUROC |
|---|---|---|---|---|---|---|---|
| **Overall** | 195 | 0.646 | 0.836 | 0.832 | 0.863 | **0.832** | **0.961** |
| Code Development | 120 | 0.808 | 0.783 | 0.735 | 0.866 | 0.742 | 0.984 |
| Code Execution | 53 | 0.509 | 0.925 | 0.925 | 0.925 | 0.925 | 0.939 |
| Result Analysis | 22 | 0.091 | 0.909 | 0.725 | 0.725 | 0.725 | 0.925 |
| semantic-self-consistency | 79 | 0.937 | 0.911 | 0.693 | 0.859 | 0.742 | 0.942 |
| stay-on-topic | 116 | 0.448 | 0.784 | 0.832 | 0.763 | 0.766 | 0.962 |

- **Confusion matrix at 0.5:** TP 97, FP 3, FN 29, TN 66. Pass-class precision is 0.97 and recall 0.77.
  Pass-class F1 is 0.858; **don't compare that with PaperBench's number, which is macro-F1.**
- **Calibration:** Brier 0.134, ECE 0.178. Jev's probabilities run **low**.
- **Cross-fit threshold:** pooled macro-F1 0.878, but unstable. Tuning on `stay-on-topic` picks t = 0.05, which
  collapses to all-pass on the 94%-pass submission (macro-F1 0.48).
- **Rubric-tree score, human vs. Jev expected score:** 0.910 vs 0.799 (`semantic-self-consistency`) and
  0.482 vs 0.325 (`stay-on-topic`). Jev consistently underestimates.
- **Error pattern 1, too strict (8 of the 10 largest errors):** "Code for acquiring dataset X" items that humans passed
  because the submission loads X through the EleutherAI eval harness, which humans accept as an "equivalent" source.
  Jev looked for explicit download code and scored them 0.07–0.21.
- **Error pattern 2, missing paper context:** "Run on the GPT-2 model family" got P = 0.94, but humans failed it.
  Only 2 of the 4 sizes listed in the paper's Table 5 were run, and Jev never saw the paper.
- **Headline:** Jev ranks leaves almost perfectly and matches o3-mini and o1 on macro-F1, with no paper, no tuning and
  no training, in about a second. Its errors are calibration (strictness) and missing context, both fixable.
- **Main caveat:** the comparison uses a different subset (2 of 5 submissions, 195 of 2,693 leaves), so it is indicative only.

## 6. Repo map (PaperBench-relevant) and rebuilding on a VM

| Path | What |
|---|---|
| [PAPERBENCH_CONTEXT.md](PAPERBENCH_CONTEXT.md) | This file |
| [REPORT_PaperBench_JudgeEval.md](REPORT_PaperBench_JudgeEval.md) | The report, including the worked example |
| [jev_judgeeval_probe.py](jev_judgeeval_probe.py) | Runs Jev on JudgeEval and writes the raw JSONL and per-leaf CSV |
| [analyze_judgeeval.py](analyze_judgeeval.py) | PaperBench-style metrics and published baselines. Needs no API key |
| [jev_rpg_probe.py](jev_rpg_probe.py) | Provides `call_jev()` (with retry on 429/529), which the probe imports |
| [download_data.sh](download_data.sh) | Fetches all 5 JudgeEval submissions and gradings (+ one ResearchPlanGen parquet) into `data/` |
| `data/judge_eval/*/0/grading/expected_result.json` | Human ground truth (committed) |
| `data/judge_eval/*/0/submission/` | Submissions (**git-ignored**, fetched by `download_data.sh`, ~350 MB) |
| [results/jev_items_judgeeval.csv](results/jev_items_judgeeval.csv) | Per-leaf human label + Jev P(pass) |
| [results/jev_raw_judgeeval.jsonl](results/jev_raw_judgeeval.jsonl) | Raw Jev responses. Also the probe's resume cache |
| [results/sent_to_jev/](results/sent_to_jev/) | Exact states and first requests sent to Jev |
| [examples/](examples/) | Worked example: CFG paper files, rubric, extracted submission, human grades |

**Rebuild on a fresh machine.** This was tested on 2026-09-30 from a clean clone with Python 3.14. It needs `git`,
`curl`, `tar` and Python ≥ 3.10.
```bash
git clone https://github.com/devchilll/jev-rubrics.git && cd jev-rubrics
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # pandas, pyarrow
./download_data.sh                                 # ~350 MB, about 10 s on a fast link
.venv/bin/python analyze_judgeeval.py              # recomputes all metrics from the committed CSV; no key needed
.venv/bin/python jev_judgeeval_probe.py --dry-run  # prints state sizes and the first question; no key needed
```
In the clean-clone test, the rebuilt states were **byte-identical** to `results/sent_to_jev/`.

**To re-query Jev** instead of reusing the committed answers:
```bash
export TYPESAFE_API_KEY=...        # never commit it
mv results/jev_raw_judgeeval.jsonl results/jev_raw_judgeeval.prev.jsonl   # the probe resumes from this file, so move it aside
.venv/bin/python jev_judgeeval_probe.py --model jev-1.13.0   # pin the version used in the report
.venv/bin/python analyze_judgeeval.py
```
Useful probe flags:
- `--papers <ids…>` picks which submissions to run.
- `--batch-size` defaults to 40.
- `--n-context` sets how many parent requirements to include (default 1).
- `--max-state-chars` defaults to 100000.

## 7. Related datasets search

The goal was datasets where a model's output (ideally an **agent trajectory**) is graded **per rubric item with human
labels**, the same shape as JudgeEval.

| Dataset | Domain | What's judged | Task-specific rubric? | Human labels | Size | Notes |
|---|---|---|---|---|---|---|
| **PaperBench JudgeEval** | Paper reproduction | Final repo + logs | ✅ weighted tree | ✅ | 5 subs / 2,693 leaves | Current test set |
| **DevAI** ([Agent-as-a-Judge, arXiv:2410.10934](https://arxiv.org/abs/2410.10934), ICML 2025; [GitHub](https://github.com/metauto-ai/agent-as-a-judge), [HF](https://huggingface.co/datasets/DEVAI-benchmark/DEVAI)) | AI/ML dev tasks | Workspaces from MetaGPT, GPT-Pilot, OpenHands | ✅ 365 requirements in a dependency graph | ✅ per requirement per agent | 55 tasks × 3 agents | **Closest cousin.** MIT. Human grades in `benchmark/judgment/<agent>/human_as_a_judge/*.json` (`requirements[].satisfied`). The repo has a trajectories folder, but only one OpenHands file |
| **HealthBench meta-eval** ([OpenAI blog](https://openai.com/index/healthbench/), [paper](https://cdn.openai.com/pdf/bd7a39d5-9e9f-47b3-903c-8b847ca650c7/healthbench_paper.pdf), [code](https://github.com/openai/simple-evals), [data](https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_meta_eval.jsonl)) | Medical chat | Model response | ✅ one criterion per row | ✅ ~2 physicians per row | 29,511 rows, 60,896 labels, 3,671 conversations | Fields: `prompt, completion, rubric, binary_labels, anonymized_physician_ids, category`. GPT-4.1 grader macro-F1 0.71 ≈ physician–physician agreement. Generic "consensus" criteria; noisy labels |
| **ProfBench** ([arXiv:2510.18941](https://arxiv.org/abs/2510.18941), ICLR 2026) | Chemistry, physics, finance, consulting | Model reports | ✅ expert rubrics | ✅ PhD/MBA experts | 7,347 response–criterion pairs | Has LLM-judge baselines |
| **AgentRewardBench** ([arXiv:2504.08942](https://arxiv.org/abs/2504.08942)) | Web agents | **Full trajectories** | ❌ fixed questions: success, side effects, repetition | ✅ experts | 1,302 trajectories | Real trajectories with human labels |
| **BaitBench** ([arXiv:2608.30724](https://arxiv.org/abs/2608.30724)) | ML tasks with planted shortcuts | Agent transcripts | ❌ "reward-hacked?" | ✅ annotated | 7 agents; 57% of runs hacked | Reward hacking in ML work |
| **ImpossibleBench** ([ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/ca688eb14e29701a11bdba6633186328-Paper-Conference.pdf)) | Coding tasks that can't be solved honestly | Transcripts | ❌ "cheated?" | ✅ by construction | many | Clean reward-hacking labels |
| **Terminal Wrench** ([arXiv:2604.17596](https://arxiv.org/pdf/2604.17596)) | Terminal agents | Trajectories | ❌ hack vs. legitimate | ✅ | 3,632 hack + 2,352 legit | Reward-hacking trajectories at scale |

Also checked and ruled out: **openai/simple-evals** (MMLU, MATH, GPQA, DROP, MGSM, HumanEval, SimpleQA, BrowseComp,
HealthBench). It has no paper reproduction and no trajectories, and it stopped being updated in July 2025. Its only
relevant part is the HealthBench meta-eval above.

**Gap:** no public dataset has **paper-reproduction trajectories with per-rubric human labels**. It would have to be
built by running open agents on the 20 PaperBench papers, logging every action, and labeling the results. That is a
likely data contribution for the paper.

**Planned use:**
- **Main evaluation:** JudgeEval + DevAI.
- **Generality at scale:** HealthBench meta-eval + ProfBench.
- **Reward-hacking questions:** BaitBench, ImpossibleBench, Terminal Wrench.
- **Trajectory-level judging:** AgentRewardBench.

## 8. Next steps (in priority order)

1. **Same-subset baseline:** run an open LLM judge (e.g. Qwen3 or DeepSeek with PaperBench's judge prompt) on the same
   195 leaves. This gives a like-for-like comparison.
2. **Full JudgeEval:** retrieve only the files relevant to each leaf so `rice`, `all-in-one` and `pinn` fit in 32k
   tokens. That yields a number directly comparable to o3-mini's 0.827.
3. **Paper context:** add the relevant `paper.md` section per leaf. The whole paper is ~30k tokens, too big to send.
4. **Calibration:** a proper held-out threshold or temperature fix. More data is needed than 2 submissions.
5. **DevAI run:** adapt the probe (state = agent workspace, questions = requirements) and compare with its published
   LLM-as-judge and Agent-as-a-Judge numbers.
6. **HealthBench meta-eval run** as a large out-of-domain check against GPT-4.1's 0.71.
7. **Train an open Jev-style judge:**
   - Base model: a decoder LLM with a probability head (long context), rather than an 8k encoder.
   - Data: synthetic perturbations (label by construction), LLM-jury soft labels, and human gold for evaluation only.
   - Key ablation: RLCD-style RL with proper-scoring rewards versus supervised log-loss plus temperature scaling.
8. **Reward-model experiment:** use the frozen judge as the RL reward. Measure how quickly the policy exploits it
   compared with an LLM-judge reward, and whether calibrated confidence flags the exploitation.

**Paper framing we discussed:**
- **Science:** is RLCD more than supervised calibration?
- **Application:** calibrated rubric judges as cheap reward models, since rubric rewards score every item for every rollout.
- **Robustness:** are calibrated judges harder to exploit?
- **Release:** an open judge plus data.

**Candidate venues:** ICLR 2027 workshops; NeurIPS 2027 Datasets & Benchmarks or main track; ACL/EMNLP via ARR; COLM 2027.

## 9. Gotchas

- PaperBench's "F1" is **macro-F1**. Jev's comparable number is 0.832, not the pass-class 0.858.
- PaperBench pools leaves across examples, so `pinn` (1,963 leaves) dominates its published numbers.
- JudgeEval submissions are **end states, not trajectories**.
- `submission.tar`, `paper.md` and `paper.pdf` are Git LFS files. On GitHub they show as pointers; download them from
  `media.githubusercontent.com/media/openai/frontier-evals/main/...`, as `download_data.sh` does.
- The probe **resumes** from `results/jev_raw_judgeeval.jsonl`. Move that file aside to force fresh API calls.
- `jev-latest` can change over time. Pin `--model jev-1.13.0` to reproduce the report.
- **Licensing:** the public repo includes copies of PaperBench files (the CFG paper and submission in `examples/`,
  states in `results/sent_to_jev/`). No LICENSE file was found at the root of `openai/frontier-evals`. Remove them if
  redistribution becomes a concern; `download_data.sh` can regenerate them.
