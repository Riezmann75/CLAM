import os

import numpy as np
import torch

from lib.grid_search import GridSearch, SearchSpace
from lib.grid_search import SearchSpace
from lib.models import NLL, GenomicEncoder, PathologicalEncoder, SurvivalModel
from lib.pre_process import load_dataset
from lib.train import train_model_with_config
from lib.utils import decorate_optimizer


h5_dir = "wsi_patches/BLCA/patches/"
h5_files = os.listdir(h5_dir)

case_ids = [h5_file.split("-01Z")[0] for h5_file in h5_files]

processed_data = load_dataset(h5_dir, h5_files)

path_enc = PathologicalEncoder(hidden_dim=128)
geno_enc = GenomicEncoder(
    df=processed_data["filtered_df"],
    categorical_cols=processed_data["categorical_cols"],
    numeric_cols=processed_data["numeric_cols"],
    hidden_dim=128,
)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

lr = 0.0005
loss = NLL()
num_epochs = 10

search_space = SearchSpace.model_validate(
    {
        "learning_rates": [0.0001],
        "weight_decays": [1e-4],
        "optimizers": [
            decorate_optimizer(torch.optim.Adam),
        ],
        "num_epochs": [10],
    }
)

grid_searcher = GridSearch(search_space, device=device)
grid_searcher(
    Model=SurvivalModel,
    model_init_args={
        "path_encoder": path_enc,
        "geno_encoder": geno_enc,
        "hidden_dim": 128,
    },
    train_fn=train_model_with_config,
    loss_fn=NLL(),
    train_loader=processed_data["train_loader"],
    validation_loader=processed_data["val_loader"],
    test_loader=processed_data["test_loader"],
)
