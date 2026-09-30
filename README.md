# jev-rubrics

Exploring calibrated "System One" decision models (e.g. TypeSafe AI's Jev) as fast rubric judges for
paper-reproduction tasks, and possibly as reward models in RL.

**Rebuilding or continuing the PaperBench work? Start with [PAPERBENCH_CONTEXT.md](PAPERBENCH_CONTEXT.md).**
It holds the full context, exact method, results, tested setup commands and the related-dataset search.

## Reports
1. **[REPORT_PaperBench_JudgeEval.md](REPORT_PaperBench_JudgeEval.md)** covers Jev on OpenAI's PaperBench JudgeEval
   ([paper](https://arxiv.org/abs/2504.01848), [data](https://github.com/openai/frontier-evals/tree/main/project/paperbench)),
   scored against human labels and compared with PaperBench's published LLM judges.
2. **[REPORT_ResearchPlanGen.md](REPORT_ResearchPlanGen.md)** covers the Jev sanity probe on Meta's ResearchPlanGen
   ([paper](https://arxiv.org/abs/2512.23707), [data](https://huggingface.co/datasets/facebook/research-plan-gen)).
   That dataset has no ground truth.

## Layout
| Path | What |
|---|---|
| `jev_judgeeval_probe.py` | Sends JudgeEval submissions to Jev (one yes/no question per rubric leaf) and scores against human labels |
| `analyze_judgeeval.py` | PaperBench-style macro metrics, per-category breakdown, cross-fit threshold, published baselines |
| `jev_rpg_probe.py` | Sends ResearchPlanGen reference solutions + rubrics to Jev (sanity probe, no ground truth) |
| `download_data.sh` | Fetches the raw datasets into `data/` |
| `data/judge_eval/*/0/grading/expected_result.json` | Human-graded rubric trees (ground truth), from PaperBench |
| `results/jev_items_judgeeval.csv` | One row per rubric leaf: requirement, human label, Jev P(pass) |
| `results/jev_raw_*.jsonl` | Raw Jev API responses |
| `results/sent_to_jev/` | The exact state text and first request sent to Jev per submission |
| `results/jev_items_ml_test_n20.csv` | ResearchPlanGen probe output |
| `examples/` | One full JudgeEval example (CFG paper, rubric, submission, human grades), walked through in the PaperBench report |

The submissions (up to 212 MB each) and the ResearchPlanGen parquet files are not committed. Run `download_data.sh`.

## Reproduce
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./download_data.sh
export TYPESAFE_API_KEY=...
.venv/bin/python jev_judgeeval_probe.py      # ~5 API calls
.venv/bin/python analyze_judgeeval.py
.venv/bin/python jev_rpg_probe.py            # optional, 20 API calls
```

## Data sources and licenses
- **PaperBench / JudgeEval:** [openai/frontier-evals](https://github.com/openai/frontier-evals/tree/main/project/paperbench).
  `rice` is cloned from its authors' repo by `download_data.sh`, as in PaperBench's own `download_data.py`.
- **ResearchPlanGen:** [facebook/research-plan-gen](https://huggingface.co/datasets/facebook/research-plan-gen),
  CC-BY-NC, generated with Llama 4 (see the dataset card for Llama naming terms).
