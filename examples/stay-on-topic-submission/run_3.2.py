import os
from pathlib import Path

from harness import evaluate
from utils import save_task_results

GUIDANCE_SCALES = [
    1,
    1.1,
    1.25,
    1.5,
    1.75,
    2,
]


MODELS = [
    "TheBloke/WizardLM-30B-fp16",
    "timdettmers/guanaco-65b",
]

TASKS = [
    "gsm8k_cot_self_consistency",
    "agieval_aqua_rat",
]
task_to_metric = {
    "gsm8k_cot_self_consistency": "exact_match",
    "agieval_aqua_rat": "acc",
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Run the Eleuther evaluation harness of CFG enabled HF LMs for section 3.1"
    )

    args = parser.parse_args()

    task_results = evaluate(MODELS, TASKS, GUIDANCE_SCALES, task_to_metric)

    results_dir = Path("results")
    save_task_results(task_results, results_dir / "3.2/")
