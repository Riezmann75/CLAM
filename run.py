import os

import numpy as np
import torch

from lib.grid_search import GridSearch, SearchSpace
from lib.grid_search import SearchSpace
from lib.models import NLL, GenomicEncoder, PathologicalEncoder, SurvivalModel
from lib.pre_process import load_dataset
from lib.train import train_model_with_config
from lib.utils import decorate_optimizer
from lib.plot import plot_top_configs

h5_dir = "wsi_patches/BLCA/patches/"
clean_csv_path = "dataset_csv/tcga_blca_all_clean.csv"
h5_files = os.listdir(h5_dir)

processed_data = load_dataset(clean_csv_path, h5_dir, h5_files)

path_enc = PathologicalEncoder(hidden_dim=128)
geno_enc = GenomicEncoder(
    df=processed_data["filtered_df"],
    categorical_cols=processed_data["categorical_cols"],
    numeric_cols=processed_data["numeric_cols"],
    hidden_dim=128,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss = NLL()

search_space = SearchSpace.model_validate(
    {
        "learning_rates": np.arange(1e-4, 1e-3, step=2e-4).tolist(),
        "weight_decays": [1e-4, 1e-3, 1e-2],
        "optimizers": [
            decorate_optimizer(torch.optim.Adam),
            decorate_optimizer(torch.optim.SGD),
        ],
        "num_epochs": [50, 100],
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
    validation_loader=processed_data["validate_loader"],
    test_loader=processed_data["test_loader"],
)

plot_top_configs(experiment_path=os.path.join(os.getcwd(), "experiments"))
