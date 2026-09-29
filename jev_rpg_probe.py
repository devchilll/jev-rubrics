"""Probe Jev on ResearchPlanGen: does Jev think the reference solution meets each rubric item?

For the first N examples of a subset/split, send the reference solution as the Jev `state`
and each rubric item as a `noul` (yes/no) question. All items for one example go in one call.

Usage:
    export TYPESAFE_API_KEY=...
    python jev_rpg_probe.py                      # ml/test, first 20 examples
    python jev_rpg_probe.py --include-goal       # also give Jev the research goal as context
    python jev_rpg_probe.py --dry-run            # print the first request, don't call the API
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

HF_URL = "https://huggingface.co/datasets/facebook/research-plan-gen/resolve/main/{subset}/{split}/data.parquet"
JEV_URL = "https://api.typesafe.ai/v1/systemone"

QUESTION_TEMPLATE = (
    "The state is a research plan. Does the plan satisfy the following rubric criterion? "
    "Answer yes only if the plan actually addresses it, not just mentions related words.\n\n"
    "Criterion: {item}"
)


def load_split(subset: str, split: str, cache_dir: Path) -> pd.DataFrame:
    path = cache_dir / f"{subset}_{split}.parquet"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {subset}/{split} -> {path}")
        urllib.request.urlretrieve(HF_URL.format(subset=subset, split=split), path)
    return pd.read_parquet(path)


def build_request(row: pd.Series, model: str, include_goal: bool) -> dict:
    solution = row["Reference solution"]
    state = {"research_goal": row["Goal"], "research_plan": solution} if include_goal else solution
    questions = {
        f"r{i}": {"type": "noul", "instructions": QUESTION_TEMPLATE.format(item=item)}
        for i, item in enumerate(row["Rubric"])
    }
    return {"model": model, "state": state, "questions": questions}


def call_jev(payload: dict, api_key: str, max_retries: int = 5) -> dict:
    body = json.dumps(payload).encode()
    for attempt in range(max_retries):
        req = urllib.request.Request(
            JEV_URL,
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            if e.code in (429, 529) and attempt < max_retries - 1:
                wait = 2**attempt
                print(f"  HTTP {e.code}, retrying in {wait}s")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Jev API error {e.code}: {detail}") from e
    raise RuntimeError("unreachable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="ml", choices=["ml", "arxiv", "pubmed"])
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--include-goal", action="store_true", help="pass the Goal alongside the plan in the state")
    ap.add_argument("--dry-run", action="store_true", help="print the first request and exit")
    ap.add_argument("--cache-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    df = load_split(args.subset, args.split, args.cache_dir).head(args.n)

    if args.dry_run:
        print(json.dumps(build_request(df.iloc[0], args.model, args.include_goal), indent=2)[:4000])
        return

    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise SystemExit("Set TYPESAFE_API_KEY first.")

    tag = f"{args.subset}_{args.split}_n{args.n}{'_goal' if args.include_goal else ''}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / f"jev_raw_{tag}.jsonl"

    # Resume: skip examples already in the raw log.
    done = {}
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            rec = json.loads(line)
            done[rec["q_id"]] = rec

    with raw_path.open("a") as f:
        for i, (_, row) in enumerate(df.iterrows()):
            if row["q_id"] in done:
                continue
            print(f"[{i + 1}/{len(df)}] {row['q_id']} ({len(row['Rubric'])} rubric items)")
            t0 = time.time()
            resp = call_jev(build_request(row, args.model, args.include_goal), api_key)
            rec = {"q_id": row["q_id"], "latency_s": round(time.time() - t0, 2), "response": resp}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[row["q_id"]] = rec

    # Flatten to one row per (example, rubric item).
    rows = []
    for _, row in df.iterrows():
        rec = done[row["q_id"]]
        answers = rec["response"]["answers"]
        for j, item in enumerate(row["Rubric"]):
            rows.append({
                "q_id": row["q_id"],
                "article_id": row["article_id"],
                "item_idx": j,
                "rubric_item": item,
                "p_met": answers[f"r{j}"]["noul"],
                "model": rec["response"].get("model"),
                "latency_s": rec["latency_s"],
            })
    items = pd.DataFrame(rows)
    items_path = args.out_dir / f"jev_items_{tag}.csv"
    items.to_csv(items_path, index=False)

    # Summary. Every item is an assumed "yes", so low p_met = Jev disagrees with that assumption.
    per_ex = items.groupby("q_id").agg(n_items=("p_met", "size"), score=("p_met", "sum"), min_p=("p_met", "min"))
    print(f"\nModel: {items['model'].iloc[0]}  |  examples: {len(per_ex)}  |  rubric items: {len(items)}")
    print(f"Mean P(met): {items['p_met'].mean():.3f}   Median: {items['p_met'].median():.3f}")
    print(f"Items with P(met) >= 0.5: {(items['p_met'] >= 0.5).mean():.1%}")
    print(f"Mean latency per example: {items.groupby('q_id')['latency_s'].first().mean():.2f}s")
    print("\nPer-example expected score (sum of P(met)):")
    print(per_ex.round(3).to_string())
    print("\nLowest-confidence items (candidates where the assumed 'yes' may be wrong):")
    for _, r in items.nsmallest(10, "p_met").iterrows():
        print(f"  {r['p_met']:.3f}  [{r['q_id']} #{r['item_idx']}] {r['rubric_item'][:120]}")
    print(f"\nSaved: {raw_path}\n       {items_path}")


if __name__ == "__main__":
    main()
