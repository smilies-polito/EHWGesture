import argparse
import json
import numpy as np
import scienceplots
import matplotlib.pyplot as plt
# Load the JSON file
import pandas as pd
import os
plt.style.use(['science', 'ieee', 'vibrant', 'no-latex'])
# Increase global font size for better visibility
plt.rcParams.update({'font.size': 22})

def produce_statistics(output_folder):
    # File paths
    print(os.getcwd())
    file_paths = {
        "sw=3": os.path.join('TriggerDetection','output', 'metrics_sw3.json'),
        "sw=5": os.path.join('TriggerDetection', 'output', 'metrics_sw5.json'),
        "sw=7": os.path.join('TriggerDetection', 'output', 'metrics_sw7.json')
    }
    os.makedirs(output_folder, exist_ok=True)
    # Define categories and speed groups
    categories = {
        "FT": ["FTS", "FTN", "FTF"],
        "OC": ["OCS", "OCN", "OCF"],
        "PS": ["PSS", "PSN", "PSF"],
        "NOSE": ["NOSE"]
    }
    speed_groups = ["S", "N", "F"]

    # Load data
    data = {key: json.load(open(path, "r")) for key, path in file_paths.items()}

    # Initialize dictionaries for storing statistics
    mae_values_by_task = {task: [] for task in categories.keys()}
    stats_tasks = {key: {} for key in file_paths}
    stats_speed = {key: {} for key in file_paths}

    for key, dataset in data.items():
        # Compute task-based statistics and collect MAE values
        for task, subcategories in categories.items():
            mae_values = []
            correct_detections, ground_truths, wrong_detections = [], [], []

            for cat in subcategories:
                mae_values.extend(sum(dataset["mae"].get(cat, []), []))  # Flatten list
                correct_detections.extend(dataset["correct_detections"].get(cat, []))
                ground_truths.extend(dataset["ground_truths"].get(cat, []))
                wrong_detections.extend(dataset["spurious_detections"].get(cat, []))

            correct_detections = np.array(correct_detections)
            ground_truths = np.array(ground_truths)
            wrong_detections = np.array(wrong_detections)
            total_detections = correct_detections + wrong_detections
            mae_values = np.array(mae_values)

            stats_tasks[key][task] = {
                "mean_mae": np.mean(mae_values) if len(mae_values) > 0 else None,
                "std_mae": np.std(mae_values) if len(mae_values) > 0 else None,
                "accuracy": np.mean(correct_detections / ground_truths) if len(ground_truths) > 0 else None,
                "std_accuracy": np.std(correct_detections / ground_truths) if len(ground_truths) > 0 else None,
                "FDR": np.mean(wrong_detections / total_detections) if len(total_detections) > 0 else None,
                "std_FDR": np.std(wrong_detections / total_detections) if len(total_detections) > 0 else None,
            }
            mae_values_by_task[task].extend(mae_values)

        # Compute speed-based statistics
        for speed in speed_groups:
            wrong_detections, correct_detections = [], []

            for task, subcategories in categories.items():
                for cat in subcategories:
                    if cat.endswith(speed):
                        wrong_detections.extend(dataset["spurious_detections"].get(cat, []))
                        correct_detections.extend(dataset["correct_detections"].get(cat, []))

            wrong_detections = np.array(wrong_detections)
            correct_detections = np.array(correct_detections)
            total_detections = correct_detections + wrong_detections

            stats_speed[key][speed] = {
                "mean_wrong_det_rate": np.mean((wrong_detections / total_detections) * 100),
                "std_wrong_det_rate": np.std((wrong_detections / total_detections) * 100)
            }

    # -------- Figure 1: Boxplots for MAE by Task --------
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    # Color-blind friendly palette (Color Universal Design)
    box_colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442"]

    # Create boxplots with specified colors
    box = ax1.boxplot(mae_values_by_task.values(), labels=mae_values_by_task.keys(), patch_artist=True)
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)

    # Overlay individual data points (with jitter)
    for i, task in enumerate(mae_values_by_task.keys()):
        y = mae_values_by_task[task]
        x = np.random.normal(i + 1, 0.05, size=len(y))
        ax1.scatter(x, y, alpha=0.3, color="black")

    ax1.set_ylabel("Absolute Error")
    ax1.grid(True)
    plt.savefig(os.path.join(output_folder, "AE_boxplots.png"), dpi=300)
    # -------- Figure 2: Line plot for Wrong Detection Rate by Speed --------
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    smoothing_windows = list(file_paths.keys())
    # Color-blind friendly colors and distinct markers
    line_colors = ["#0072B2", "#D55E00", "#CC79A7"]  # Blue, Red-Orange, Magenta
    markers = ["o", "s", "D"]
    labels={'S':'SLOW', 'N':'NORMAL', 'F':'FAST'}
    for (speed, color, marker) in zip(speed_groups, line_colors, markers):
        means = [stats_speed[window][speed]["mean_wrong_det_rate"] for window in smoothing_windows if
                 speed in stats_speed[window]]
        if len(means) > 0:
            ax2.plot(smoothing_windows[:len(means)], means, marker=marker, color=color, label=labels[speed], markersize=10, linewidth=3)

    ax2.set_xlabel("Smoothing Window")
    ax2.set_ylabel("FDR (%)")

    # Create a denser adaptive grid (major and minor ticks)
    ax2.grid(which="major", linestyle="--", linewidth=0.8)
    ax2.minorticks_on()
    ax2.grid(which="minor", linestyle=":", linewidth=0.5)
    ax2.legend()
    plt.savefig(os.path.join(output_folder,"FDR.png"), dpi=300)
    plt.show()

    # -------- Generate Tables for Reporting --------

    # Build task-based statistics table
    task_stats_rows = []
    for window, tasks in stats_tasks.items():
        for task, stats in tasks.items():
            row = {"Smoothing Window": window, "Task": task}
            row.update(stats)
            task_stats_rows.append(row)
    df_tasks = pd.DataFrame(task_stats_rows)

    # Build speed-based statistics table
    speed_stats_rows = []
    for window, speeds in stats_speed.items():
        for speed, stats in speeds.items():
            row = {"Smoothing Window": window, "Speed": speed}
            row.update(stats)
            speed_stats_rows.append(row)
    df_speeds = pd.DataFrame(speed_stats_rows)

    # Convert DataFrames to Markdown format for easy inclusion in your paper
    # For each task, choose the row with the highest correct detection rate (detection rate is already a fraction)
    best_rows = df_tasks.loc[df_tasks.groupby("Task")["accuracy"].idxmax()].copy()

    # Format the numbers: convert detection rates and wrong detection ratio to percentages (mean ± std) and MAE as is
    best_rows["Acc (\%)"] = best_rows.apply(
        lambda row: f"{row['accuracy']*100:.2f} ± {row['std_accuracy']*100:.2f}", axis=1
    )
    best_rows["FDR (\%)"] = best_rows.apply(
        lambda row: f"{row['FDR']*100:.2f} ± {row['std_FDR']*100:.2f}", axis=1
    )
    best_rows["MAE"] = best_rows["mean_mae"].map(lambda x: f"{x:.5f}")

    # Select the columns of interest and rename for clarity
    table_df = best_rows[["Smoothing Window", "Task", "MAE", "Acc (\%)", "FDR (\%)"]]

    # Output the markdown table
    print(table_df.to_markdown(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes statistics on trigger detection results")
    parser.add_argument("--output_folder", default='output', type=str,
                        help="Path to output folder to save images.")

    args = parser.parse_args()
    output_folder = args.output_folder
    produce_statistics(output_folder)