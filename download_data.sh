#!/usr/bin/env bash
# Fetch raw data into data/: PaperBench JudgeEval submissions + grading, and ResearchPlanGen ML test split.
set -euo pipefail
cd "$(dirname "$0")"

RAW=https://raw.githubusercontent.com/openai/frontier-evals/main/project/paperbench/data/judge_eval
LFS=https://media.githubusercontent.com/media/openai/frontier-evals/main/project/paperbench/data/judge_eval

for p in all-in-one pinn rice semantic-self-consistency stay-on-topic-with-classifier-free-guidance; do
  d=data/judge_eval/$p/0
  mkdir -p "$d/grading"
  [ -f "$d/grading/expected_result.json" ] || curl -sfL "$RAW/$p/0/grading/expected_result.json" -o "$d/grading/expected_result.json"
  [ -d "$d/submission" ] && continue
  if [ "$p" = rice ]; then
    # Not redistributed by PaperBench; clone the authors' repo at the pinned commit.
    git clone -q https://github.com/chengzelei/RICE "$d/submission"
    git -C "$d/submission" checkout -q 4962aeb46d6000ebdf5ecbc5b46fff0d0505da59
    rm -rf "$d/submission/.git"
  else
    curl -sfL "$LFS/$p/0/submission.tar" -o "$d/submission.tar"
    tar -xf "$d/submission.tar" -C "$d"
  fi
  echo "fetched $p"
done

mkdir -p data
[ -f data/ml_test.parquet ] || curl -sfL \
  https://huggingface.co/datasets/facebook/research-plan-gen/resolve/main/ml/test/data.parquet -o data/ml_test.parquet
echo done
