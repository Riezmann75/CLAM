from matplotlib import pyplot as plt
import numpy as np
import json
import os
import argparse


def plot_training_curves(current_path, result_logs, y_lim=None, num_cols=2):
    number_of_configs = len(result_logs)
    num_rows = number_of_configs // num_cols + number_of_configs % num_cols

    fig, axes = plt.subplots(
        figsize=(16, 4 * num_rows),
        ncols=num_cols,
        nrows=num_rows,
    )
    for i in range(num_rows):
        for j in range(num_cols):
            if i * num_cols + j >= number_of_configs:
                break
            avg_losses = result_logs[i * num_cols + j].get("avg_losses")
            val_losses = result_logs[i * num_cols + j].get("val_losses")
            test_c_index = result_logs[i * num_cols + j].get("test_c_index")
            train_c_index = result_logs[i * num_cols + j].get("train_c_index")
            config = result_logs[i * num_cols + j].get("config")
            optimizer = (
                "Adam" if "adam" in config.get("optimizer").get("name") else "SGD"
            )
            lr = config.get("lr")
            # max num of digits for lr is 5
            lr = float(f"{lr:.5g}")
            weight_decay = config.get("weight_decay")
            num_epoch = config.get("num_epoch")
            if type(axes) is np.ndarray:
                # if axes is 2D array
                if axes.ndim == 2:
                    axes[i, j].plot(
                        range(1, len(avg_losses) + 1), avg_losses, marker="o"
                    )
                    axes[i, j].plot(
                        range(1, len(val_losses) + 1), val_losses, marker="o"
                    )
                    axes[i, j].set_title(
                        f"Optimizer: {optimizer}, LR: {lr}, weight_decay: {weight_decay}, Test C-index: {test_c_index:.4f}, Train C-index: {train_c_index:.4f}",
                        size=10,
                        pad=10,
                    )
                    axes[i, j].set_xlabel("Epoch")
                    if y_lim is not None:
                        axes[i, j].set_ylim(y_lim)
                        axes[i, j].set_yticks(
                            np.arange(
                                y_lim[0], y_lim[1] + 0.1, (y_lim[1] - y_lim[0]) / 10
                            )
                        )
                    axes[i, j].set_ylabel("Average Loss")
                    axes[i, j].grid()
                    axes[i, j].legend(["Train", "Validation"])
                elif axes.ndim == 1:  # if axes is 1D array
                    axes[j].plot(range(1, len(avg_losses) + 1), avg_losses, marker="o")
                    axes[j].plot(range(1, len(val_losses) + 1), val_losses, marker="o")
                    axes[j].set_title(
                        f"Optimizer: {optimizer}, LR: {lr:.4f}, weight_decay: {weight_decay}, Test C-index: {test_c_index:.4f}, Train C-index: {train_c_index:.4f}",
                        size=10,
                        pad=10,
                    )
                    axes[j].set_xlabel("Epoch")
                    if y_lim is not None:
                        axes[j].set_ylim(y_lim)
                        axes[j].set_yticks(
                            np.arange(
                                y_lim[0], y_lim[1] + 0.1, (y_lim[1] - y_lim[0]) / 10
                            )
                        )
                    axes[j].set_ylabel("Average Loss")
                    axes[j].grid()
                    axes[j].legend(["Train", "Validation"])
            else:  # if there's only one plot
                axes.plot(range(1, len(avg_losses) + 1), avg_losses, marker="o")
                axes.plot(range(1, len(val_losses) + 1), val_losses, marker="o")
                axes.set_title(
                    f"Optimizer: {optimizer}, LR: {lr:.4f}, weight_decay: {weight_decay}, Test C-index: {test_c_index:.4f}, Train C-index: {train_c_index:.4f}",
                    size=10,
                    pad=10,
                )
                axes.set_xlabel("Epoch")
                if y_lim is not None:
                    axes.set_ylim(y_lim)
                    axes.set_yticks(
                        np.arange(y_lim[0], y_lim[1] + 0.1, (y_lim[1] - y_lim[0]) / 10)
                    )
                axes.set_ylabel("Average Loss")
                axes.grid()
                axes.legend(["Train", "Validation"])
    fig.tight_layout()
    plt.savefig(os.path.join(current_path, "training_curves.png"))


def plot_top_configs(
    experiment_path: str,
    log_file: str = "result_logs.jsonl",
    y_lim=None,
    top_k: int = 2,
):
    log_path = os.path.join(experiment_path, log_file)
    with open(log_path, "r") as f:
        result_logs = [json.loads(line) for line in f.readlines()]
        result_logs = [
            log for log in result_logs if log.get("test_c_index") is not None
        ]
        result_logs = sorted(result_logs, key=lambda x: x["test_c_index"], reverse=True)
        # select top 2 results
        top_k_results = result_logs[:top_k]
        plot_training_curves(experiment_path, top_k_results, y_lim=y_lim)


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
        default=None,
        help="y-axis limits for the loss plots",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="number of top configurations to plot",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="result_logs.jsonl",
        help="name of the log file containing results",
    )
    parser = parser.parse_args()

    plot_top_configs(
        experiment_path=parser.experiment_path,
        log_file=parser.log_file,
        y_lim=parser.y_lim,
        top_k=parser.top_k,
    )

# example command to run:
# python lib/plot.py --experiment_path ./experiments --log_file result_logs.jsonl --y_lim 2 5 --top_k 2
