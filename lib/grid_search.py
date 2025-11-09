import json
import os
from pydantic import BaseModel
from torch import nn
import torch

from lib.exception import StopTrainingError
from lib.utils import parse_optimizer

from tqdm import tqdm


class SearchSpace(BaseModel):
    learning_rates: list
    optimizers: list
    weight_decays: list
    num_epochs: list


class GridSearch:
    def __init__(self, search_space: SearchSpace, device=None):
        self.learning_rates = search_space.learning_rates
        self.optimizers = search_space.optimizers
        self.weight_decays = search_space.weight_decays
        self.num_epochs = search_space.num_epochs
        self.device = device

    def __call__(self, Model: nn.Module, train_fn, model_init_args=None, **kwargs):
        for optimizer in self.optimizers:
            for lr in self.learning_rates:
                for weight_decay in self.weight_decays:
                    for num_epoch in self.num_epochs:
                        # average losses each epoch
                        model = Model(**model_init_args) if model_init_args else Model()
                        model.to(self.device) if self.device else model.to("cpu")
                        configured_optimizer = optimizer(
                            lr=lr,
                            weight_decay=weight_decay,
                            model=model,
                        )
                        if "sgd" in parse_optimizer(str(configured_optimizer)).get(
                            "name"
                        ):
                            configured_optimizer.param_groups[0]["momentum"] = 0.9
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            configured_optimizer,
                            T_max=num_epoch,
                        )
                        try:
                            (
                                avg_losses,
                                val_losses,
                                c_index_value,
                                train_c_index_value,
                            ) = train_fn(
                                model=model,
                                optimizer=configured_optimizer,
                                scheduler=scheduler,
                                num_epoch=num_epoch,
                                device=self.device,
                                **kwargs,
                            )
                            self.write_training_log(
                                {
                                    "config": {
                                        "optimizer": parse_optimizer(
                                            str(configured_optimizer),
                                            lr=lr,
                                            weight_decay=weight_decay,
                                        ),
                                        "lr": lr,
                                        "weight_decay": weight_decay,
                                        "num_epoch": num_epoch,
                                    },
                                    "avg_losses": avg_losses,
                                    "val_losses": val_losses,
                                    "test_c_index": c_index_value,
                                    "train_c_index": train_c_index_value,
                                }
                            )
                        except StopTrainingError as e:
                            # print(
                            #     f"Config: {lr}, {parse_optimizer(str(configured_optimizer)).get('name')}, {weight_decay}, {num_epoch} stopped training due to {e}\n"
                            # )
                            self.write_training_log(
                                {
                                    "config": {
                                        "optimizer": parse_optimizer(
                                            str(configured_optimizer),
                                            lr=lr,
                                            weight_decay=weight_decay,
                                        ),
                                        "lr": lr,
                                        "weight_decay": weight_decay,
                                        "num_epoch": num_epoch,
                                    },
                                    "avg_losses": None,
                                    "val_losses": None,
                                    "c_index": None,
                                    "train_c_index": train_c_index_value,
                                    "error": str(e),
                                }
                            )
                            continue

    def write_training_log(self, log):
        current_path = os.getcwd()
        os.makedirs(os.path.join(current_path, "experiments"), exist_ok=True)
        log_path = os.path.join(current_path, "experiments/result_logs.jsonl")
        if not os.path.exists(log_path):
            open(log_path, "x").close()
        with open(log_path, "a") as f:
            f.write(json.dumps(log) + "\n")
