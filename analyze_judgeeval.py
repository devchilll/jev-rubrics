"""Score Jev's JudgeEval results the same way PaperBench scores its LLM judges.

PaperBench (paperbench/judge/judge_eval/evaluate.py) pools all leaf nodes across examples and reports
sklearn accuracy + MACRO-averaged precision/recall/F1 (average over the pass and fail classes),
overall and stratified by task_category. We do the same on Jev's P(pass) thresholded at 0.5.

Also reports a cross-fit threshold: pick the threshold maximizing macro-F1 on one submission,
apply it to the other, so the tuned number is not fit on its own test data.

    python analyze_judgeeval.py   # reads results/jev_items_judgeeval.csv
"""

import numpy as np
import pandas as pd

from jev_judgeeval_probe import auroc

CATEGORIES = ["Code Development", "Code Execution", "Result Analysis"]

# From openai/frontier-evals project/paperbench/experiments/judge_eval/tables/ (all 5 JudgeEval examples).
PAPERBENCH = pd.DataFrame([
    ("gpt-4o-mini", 0.632, 0.640, 0.603, 0.590, 0.588, 0.544, 0.778, 8.01),
    ("gpt-4o", 0.740, 0.744, 0.725, 0.728, 0.681, 0.707, 0.825, 119.87),
    ("o1-mini (high)", 0.807, 0.848, 0.761, 0.775, 0.669, 0.741, 0.798, 71.72),
    ("o1 (high)", 0.844, 0.842, 0.841, 0.842, 0.740, 0.840, 0.876, 830.16),
    ("o3-mini (high)", 0.827, 0.827, 0.831, 0.827, 0.720, 0.822, 0.945, 66.04),
    ("random", None, None, None, 0.504, 0.469, 0.495, 0.487, 0.0),
], columns=["judge", "accuracy", "precision", "recall", "macro_f1", *CATEGORIES, "usd_per_paper"])


def macro_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    out = {"n": len(y), "accuracy": (pred == y).mean()}
    ps, rs, fs = [], [], []
    for cls in (0, 1):
        tp = ((pred == cls) & (y == cls)).sum()
        fp = ((pred == cls) & (y != cls)).sum()
        fn = ((pred != cls) & (y == cls)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        ps.append(p); rs.append(r); fs.append(2 * p * r / (p + r) if p + r else 0.0)
    out.update(precision=np.mean(ps), recall=np.mean(rs), macro_f1=np.mean(fs))
    return out


def best_threshold(y, p):
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
    return max(grid, key=lambda t: macro_metrics(y, (p >= t).astype(int))["macro_f1"])


def main():
    df = pd.read_csv("results/jev_items_judgeeval.csv")
    y, p = df.human.to_numpy(), df.p_pass.to_numpy()

    rows = []
    for name, g in [("Overall", df)] + [(c, df[df.category == c]) for c in CATEGORIES] + list(df.groupby("paper")):
        m = macro_metrics(g.human.to_numpy(), (g.p_pass >= 0.5).astype(int).to_numpy())
        rows.append({"slice": name, "n": m["n"], "pass_rate": g.human.mean(), "accuracy": m["accuracy"],
                     "precision": m["precision"], "recall": m["recall"], "macro_f1": m["macro_f1"],
                     "auroc": auroc(g.human, g.p_pass)})
    print("Jev (threshold 0.5), PaperBench-style macro metrics:")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\nCross-fit threshold (tune on one submission, test on the other):")
    papers = df.paper.unique()
    pooled_pred = np.zeros(len(df), dtype=int)
    for tune, test in [(papers[0], papers[1]), (papers[1], papers[0])]:
        a, b = df[df.paper == tune], df[df.paper == test]
        t = best_threshold(a.human.to_numpy(), a.p_pass.to_numpy())
        pooled_pred[(df.paper == test).to_numpy()] = (b.p_pass >= t).astype(int).to_numpy()
        m = macro_metrics(b.human.to_numpy(), (b.p_pass >= t).astype(int).to_numpy())
        print(f"  tuned on {tune[:28]:28s} -> t={t:.2f}; on {test[:28]:28s} macro-F1={m['macro_f1']:.3f} acc={m['accuracy']:.3f}")
    m = macro_metrics(y, pooled_pred)
    print(f"  pooled cross-fit: macro-F1={m['macro_f1']:.3f} acc={m['accuracy']:.3f}")

    print("\nPaperBench published judges (all 5 JudgeEval examples, 2,693 leaves):")
    print(PAPERBENCH.to_string(index=False))


if __name__ == "__main__":
    main()
