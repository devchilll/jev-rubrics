import os
from pathlib import Path

from harness import evaluate
from utils import save_task_results

# ours, base
GUIDANCE_SCALES = [
    1.5,
    1,
]

MODELS_SHORT = [
    "gpt2",  # G-s: GPT-2 Small (~117M parameters)
    "gpt2-medium",  # G-m: GPT-2 Medium (~345M parameters)
    "EleutherAI/pythia-160m",  # P-160M
    "EleutherAI/pythia-410m",  # P-410M
]

MODELS = MODELS_SHORT + [
    "gpt2-large",  # G-l: GPT-2 Large (~774M parameters)
    "gpt2-xl",  # G-xl: GPT-2 Extra Large (~1.5B parameters)
    "EleutherAI/pythia-1b",  # P-1B
    "EleutherAI/pythia-1.4b",  # P-1.4B
    "EleutherAI/pythia-2.8b",  # P-2.8B
]

TASKS_SHORT = [
    "arc_challenge",
    "arc_easy",
    "boolq",
    "sciq",
    "winogrande",
    "lambada_openai",
]
TASKS = TASKS_SHORT + [
    "hellaswag",
    "piqa",
    "triviaqa",
]
task_to_metric = {
    "arc_challenge": "acc",
    "arc_easy": "acc",
    "boolq": "acc",
    "hellaswag": "acc",
    "piqa": "acc",
    "sciq": "acc",
    "triviaqa": "exact_match",
    "winogrande": "acc",
    "lambada_openai": "acc",
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Run the Eleuther evaluation harness of CFG enabled HF LMs for section 3.1"
    )

    parser.add_argument(
        "--short",
        action="store_true",
        help="Run a shorter version evaluation harness with a subset of tasks and models",
        default=False,
    )

    args = parser.parse_args()

    if args.short:
        task_results = evaluate(MODELS_SHORT, TASKS_SHORT, GUIDANCE_SCALES, task_to_metric)
    else:
        task_results = evaluate(MODELS, TASKS, GUIDANCE_SCALES, task_to_metric)

    results_dir = Path("results")
    save_task_results(task_results, results_dir / "table5/")
