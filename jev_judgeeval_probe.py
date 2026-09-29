"""Probe Jev on PaperBench JudgeEval: grade reproduction submissions against human-labeled rubric leaves.

Each JudgeEval example is a submission folder (code + reproduce.log + outputs) plus a rubric tree
whose leaf nodes were graded pass/fail by humans. We concatenate the submission's text files into
the Jev `state`, ask one `noul` question per leaf, and compare Jev's P(pass) to the human labels.

Only submissions whose text fits in Jev's ~32k-token state budget can be sent whole. By default
that's the two small ones (semantic-self-consistency, stay-on-topic-with-classifier-free-guidance).

Setup (data lives in data/judge_eval/<paper>/0/{submission/, grading/expected_result.json}):
    export TYPESAFE_API_KEY=...
    python jev_judgeeval_probe.py
    python jev_judgeeval_probe.py --dry-run          # show state size + first questions, no API call
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from jev_rpg_probe import call_jev

DEFAULT_PAPERS = ["semantic-self-consistency", "stay-on-topic-with-classifier-free-guidance"]
TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".csv", ".log", ".json", ".jsonl", ".yaml", ".yml",
                 ".toml", ".cfg", ".ini", ".ipynb"}
SKIP_NAMES = {"uv.lock", "poetry.lock", "package-lock.json", ".DS_Store"}
# Show setup first, then code, then execution evidence.
FILE_ORDER = {"README.md": 0, "reproduce.sh": 1, "reproduce.log": 9}

CATEGORY_HINTS = {
    "Code Development": "Pass if the submission's code contains a correct implementation of the requirement. "
                        "It does not need to have been run.",
    "Code Execution": "Pass only if the code was actually run (see reproduce.sh, reproduce.log and output files) "
                      "and that execution carried out the requirement.",
    "Result Analysis": "Pass only if the produced outputs/logs show the stated result.",
}
QUESTION_TEMPLATE = (
    "The state is a code submission attempting to reproduce a research paper: its source files, plus "
    "reproduce.log from running reproduce.sh, plus any output files. Grade one rubric requirement.\n"
    "{hint}\n\n"
    "{context}"
    "Requirement: {req}"
)


def build_state(sub_dir: Path, max_file_chars: int) -> str:
    files = [f for f in sub_dir.rglob("*")
             if f.is_file() and ".git" not in f.parts and f.name not in SKIP_NAMES
             and f.suffix.lower() in TEXT_SUFFIXES]
    files.sort(key=lambda f: (FILE_ORDER.get(f.name, 5), str(f)))
    parts = []
    for f in files:
        text = f.read_text(errors="replace")
        if len(text) > max_file_chars:
            half = max_file_chars // 2
            text = text[:half] + f"\n[... {len(text) - max_file_chars} chars truncated ...]\n" + text[-half:]
        parts.append(f"===== FILE: {f.relative_to(sub_dir)} =====\n{text}")
    return "\n\n".join(parts)


def leaves(node: dict, ancestors: tuple = ()):
    if not node.get("sub_tasks"):
        yield node, ancestors
    for child in node.get("sub_tasks", []):
        yield from leaves(child, ancestors + (node["requirements"],))


def tree_score(node: dict, leaf_value: dict) -> float:
    """Weighted-average roll-up, same as PaperBench: leaf value in [0,1], parent = weighted mean of children."""
    if not node.get("sub_tasks"):
        return leaf_value[node["id"]]
    kids = node["sub_tasks"]
    total_w = sum(k["weight"] for k in kids)
    return sum(k["weight"] * tree_score(k, leaf_value) for k in kids) / total_w


def build_question(node: dict, ancestors: tuple, n_context: int) -> dict:
    ctx = ancestors[1:][-n_context:] if n_context else ()  # drop the root ("core contributions reproduced")
    context = "".join(f"Parent requirement: {a}\n" for a in ctx)
    hint = CATEGORY_HINTS.get(node.get("task_category"), "")
    return {"type": "noul",
            "instructions": QUESTION_TEMPLATE.format(hint=hint, context=context, req=node["requirements"])}


def auroc(y, p):
    df = pd.DataFrame({"y": y, "p": p})
    n_pos, n_neg = (df.y == 1).sum(), (df.y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = df.p.rank()
    return (ranks[df.y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def metrics(y: pd.Series, p: pd.Series) -> dict:
    pred = (p >= 0.5).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
    f1_pos = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    f1_neg = 2 * tn / (2 * tn + fn + fp) if tn else 0.0
    bins = pd.cut(p, [i / 10 for i in range(11)], include_lowest=True)
    ece = sum(len(g) / len(p) * abs(g.p.mean() - g.y.mean())
              for _, g in pd.DataFrame({"y": y, "p": p, "b": bins}).groupby("b", observed=True))
    return {"n": len(y), "human_pass_rate": y.mean(), "accuracy": (pred == y).mean(),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "f1_pass": f1_pos, "macro_f1": (f1_pos + f1_neg) / 2,
            "auroc": auroc(y, p), "brier": ((p - y) ** 2).mean(), "ece": ece}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", nargs="+", default=DEFAULT_PAPERS)
    ap.add_argument("--data-dir", type=Path, default=Path("data/judge_eval"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--batch-size", type=int, default=40, help="rubric leaves per Jev request")
    ap.add_argument("--n-context", type=int, default=1, help="how many parent requirements to include as context")
    ap.add_argument("--max-file-chars", type=int, default=30_000, help="truncate any single file beyond this")
    ap.add_argument("--max-state-chars", type=int, default=100_000, help="~32k tokens; refuse bigger states")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not args.dry_run and not api_key:
        raise SystemExit("Set TYPESAFE_API_KEY first.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "jev_raw_judgeeval.jsonl"

    # Resume: answers already collected, keyed by (paper, leaf id).
    done = {}
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            rec = json.loads(line)
            for leaf_id, ans in rec["answers"].items():
                done[(rec["paper"], leaf_id)] = ans

    rows, tree_rows = [], []
    for paper in args.papers:
        ex_dir = args.data_dir / paper / "0"
        tree = json.load(open(ex_dir / "grading/expected_result.json"))
        state = build_state(ex_dir / "submission", args.max_file_chars)
        leaf_list = list(leaves(tree))
        print(f"== {paper}: {len(leaf_list)} leaves, state {len(state):,} chars (~{len(state) / 3.5 / 1e3:.1f}k tokens)")
        if len(state) > args.max_state_chars:
            print(f"   SKIP: state exceeds --max-state-chars={args.max_state_chars:,}")
            continue

        if args.dry_run:
            node, anc = leaf_list[0]
            print(json.dumps(build_question(node, anc, args.n_context), indent=2))
            print(state[:1500], "\n...")
            continue

        todo = [(n, a) for n, a in leaf_list if (paper, n["id"]) not in done]
        with raw_path.open("a") as f:
            for b in range(0, len(todo), args.batch_size):
                batch = todo[b:b + args.batch_size]
                qmap = {f"q{i}": build_question(n, a, args.n_context) for i, (n, a) in enumerate(batch)}
                t0 = time.time()
                resp = call_jev({"model": args.model, "state": state, "questions": qmap}, api_key)
                latency = time.time() - t0
                answers = {n["id"]: {"p": resp["answers"][f"q{i}"]["noul"], "model": resp.get("model"),
                                     "latency_s": round(latency, 2), "batch_n": len(batch)}
                           for i, (n, _) in enumerate(batch)}
                f.write(json.dumps({"paper": paper, "answers": answers, "usage": resp.get("usage")}) + "\n")
                f.flush()
                for k, v in answers.items():
                    done[(paper, k)] = v
                print(f"   batch {b // args.batch_size + 1}: {len(batch)} leaves in {latency:.2f}s")

        for node, anc in leaf_list:
            ans = done[(paper, node["id"])]
            rows.append({"paper": paper, "leaf_id": node["id"], "category": node.get("task_category"),
                         "weight": node["weight"], "parent": anc[-1] if anc else "",
                         "requirement": node["requirements"], "human": int(node["score"]),
                         "human_explanation": node.get("explanation", ""), "p_pass": ans["p"],
                         "model": ans["model"]})
        p_map = {r["leaf_id"]: r["p_pass"] for r in rows if r["paper"] == paper}
        h_map = {r["leaf_id"]: r["human"] for r in rows if r["paper"] == paper}
        tree_rows.append({"paper": paper, "human_score": tree_score(tree, h_map),
                          "jev_expected_score": tree_score(tree, p_map),
                          "jev_thresholded_score": tree_score(tree, {k: float(v >= 0.5) for k, v in p_map.items()})})

    if args.dry_run or not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "jev_items_judgeeval.csv", index=False)
    pd.set_option("display.width", 200)

    print(f"\nModel: {df.model.iloc[0]}   (PaperBench reference: o3-mini-high F1 0.83, o1-high 0.84, "
          "gpt-4o 0.73, random 0.50 — on all 5 submissions)")
    table = [{"slice": "ALL", **metrics(df.human, df.p_pass)}]
    table += [{"slice": p, **metrics(g.human, g.p_pass)} for p, g in df.groupby("paper")]
    table += [{"slice": c, **metrics(g.human, g.p_pass)} for c, g in df.groupby("category")]
    print(pd.DataFrame(table).round(3).to_string(index=False))

    print("\nRubric-tree score (weighted roll-up, i.e. sum of P x points):")
    print(pd.DataFrame(tree_rows).round(3).to_string(index=False))

    df["err"] = (df.p_pass - df.human).abs()
    print("\nBiggest disagreements with human graders:")
    for r in df.nlargest(10, "err").itertuples():
        print(f"  human={r.human} P={r.p_pass:.2f} [{r.category}] {r.requirement[:130]}")
    print(f"\nSaved: {raw_path}\n       {args.out_dir / 'jev_items_judgeeval.csv'}")


if __name__ == "__main__":
    main()
