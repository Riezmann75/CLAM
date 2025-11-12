from matplotlib import pyplot as plt
import numpy as np
import json
import os
import argparse


def plot_training_curves(current_path, result_logs, y_lim=(2, 5)):
    number_of_configs = len(result_logs)
    num_cols = 2
    num_rows = number_of_configs // num_cols + number_of_configs % num_cols

    fig, axes = plt.subplots(
        figsize=(16, 8),
        ncols=num_cols,
        nrows=num_rows,
    )
    for i in range(num_rows):
        for j in range(num_cols):
            avg_losses = result_logs[i * num_cols + j].get("avg_losses")
            val_losses = result_logs[i * num_cols + j].get("val_losses")
            test_c_index = result_logs[i * num_cols + j].get("test_c_index")
            train_c_index = result_logs[i * num_cols + j].get("train_c_index")
            config = result_logs[i * num_cols + j].get("config")
            optimizer = (
                "Adam" if "adam" in config.get("optimizer").get("name") else "SGD"
            )
            lr = config.get("lr")
            weight_decay = config.get("weight_decay")
            num_epoch = config.get("num_epoch")
            axes[i, j].plot(range(1, num_epoch + 1), avg_losses, marker="o")
            axes[i, j].plot(range(1, num_epoch + 1), val_losses, marker="o")
            axes[i, j].set_title(
                f"Optimizer: {optimizer}, LR: {lr:.4f}, weight_decay: {weight_decay}, Test C-index: {test_c_index:.4f}, Train C-index: {train_c_index:.4f}",
                size=10,
                pad=10,
            )
            axes[i, j].set_xlabel("Epoch")
            axes[i, j].set_ylim(y_lim)
            axes[i, j].set_yticks(np.arange(y_lim[0], y_lim[1] + 0.1, 0.5))
            axes[i, j].set_ylabel("Average Loss")
            axes[i, j].grid()
            axes[i, j].legend(["Train", "Validation"])
    fig.tight_layout()
    plt.savefig(os.path.join(current_path, "training_curves.png"))


def plot_top_configs(experiment_path: str, y_lim=(2, 5)):
    log_path = os.path.join(experiment_path, "result_logs.jsonl")
    with open(log_path, "r") as f:
        result_logs = [json.loads(line) for line in f.readlines()]
        result_logs = [
            log for log in result_logs if log.get("test_c_index") is not None
        ]
        result_logs = sorted(result_logs, key=lambda x: x["test_c_index"], reverse=True)
        # select top 4 results
        top_4 = result_logs[:4]
        plot_training_curves(experiment_path, top_4, y_lim=y_lim)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Plot top configurations training curves"
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        default=os.path.join(os.getcwd(), "experiments"),
        help="path to the experiments directory",
    )
    parser.add_argument(
        "--y_lim",
        type=float,
        nargs=2,
        default=(2, 5),
        help="y-axis limits for the loss plots",
    )
    parser = parser.parse_args()

    plot_top_configs(
        experiment_path=parser.experiment_path,
        y_lim=parser.y_lim
    )

# example command to run:
# python lib/plot.py --experiment_path ./experiments --y_lim 2 5
