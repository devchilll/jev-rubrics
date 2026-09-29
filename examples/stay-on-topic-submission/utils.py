from pathlib import Path

import pandas as pd


def save_task_results(task_results: dict, save_dir: Path):
    for task, results in task_results.items():
        # Convert results into a DataFrame
        task_df = pd.DataFrame(results)

        # Pivot the DataFrame to create rows as models and columns as "base" and "ours"
        table = task_df.pivot(index="Model", columns="Guidance Scale", values="Score")
        table = table.sort_index()

        save_dir.mkdir(parents=True, exist_ok=True)
        # Save the table as a CSV file
        csv_filename = f"{save_dir}/{task}_results.csv"
        table.to_csv(csv_filename)

        # Print confirmation of the saved file
        print(f"Results for {task} saved to {csv_filename}.")
        print(table)
        print("-----")
