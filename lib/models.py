import pandas as pd
import torch
from torch import nn


class PathologicalEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super(PathologicalEncoder, self).__init__()
        self.fc = nn.LazyLinear(hidden_dim)

    def forward(self, x):
        # x shape: 
        x = self.fc(x)
        return x

class BaseGenomicEncoder(nn.Module):
    def __init__(
        self, df: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str]
    ):
        super(BaseGenomicEncoder, self).__init__()
        self.categorical_cols = categorical_cols
        self.numerical_cols = numeric_cols
        self.features = self.numerical_cols + self.categorical_cols
        self.embeddings = nn.ModuleDict()
        for col in self.categorical_cols:
            num_unique_values = int(df[col].nunique())
            embedding_size = 4
            self.embeddings[col] = nn.Embedding(num_unique_values, embedding_size)

    def categorical_name_to_index(self, col):
        return self.categorical_cols.index(col)

    def numerical_name_to_index(self, col):
        return self.numerical_cols.index(col)

    def categorical_index_to_name(self, index):
        return self.categorical_cols[index]

    def numerical_index_to_name(self, index):
        return self.numerical_cols[index]

    def embed(self, x):
        embedded_cols = []
        for col in self.categorical_cols:
            # ndarray = np.array()
            embedded_col = self.embeddings[col](
                x[:, self.categorical_name_to_index(col)].long()
            )
            # print(embedded_col.shape)
            embedded_cols.append(embedded_col)
        numerical_data = torch.stack(
            [x[:, self.numerical_name_to_index(col)] for col in self.numerical_cols],
            dim=1,
        ).float()
        x = torch.cat(embedded_cols + [numerical_data], dim=1)
        return x

    def embed_with_time(self, x, t):
        time_data = torch.reshape(t, (x.shape[0], 1)).float()
        x = self.embed(x)
        x = torch.cat((x, time_data), dim=1)
        return x

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented!")


class GenomicEncoder(BaseGenomicEncoder):
    def __init__(
        self,
        df: pd.DataFrame,
        categorical_cols: list[str],
        numeric_cols: list[str],
        bias=True,
        hidden_dim: int = 128,
    ):
        super().__init__(df, categorical_cols, numeric_cols)

        self.net = nn.Sequential(
            nn.LazyLinear(128),
            nn.LeakyReLU(0.1),
            nn.LazyLinear(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.LazyLinear(128),
        )
        self.fc = nn.LazyLinear(hidden_dim)

        # self.bias = nn.Sequential(nn.LazyLinear(32), nn.ReLU(), nn.LazyLinear(1))

    def forward(self, x):
        x = self.embed(x)
        x = self.net(x)
        x = self.fc(x)
        return x.squeeze()


# loss
class NLL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, failure_times, is_observed):
        # Number of observed events
        return (
            1
            / len(preds)
            * torch.sum(torch.exp(preds) * failure_times - is_observed * preds, dim=0)
        )


class SurvivalModel(nn.Module):
    def __init__(
        self,
        path_encoder: PathologicalEncoder,
        geno_encoder: GenomicEncoder,
        hidden_dim: int = 128,
    ):
        super(SurvivalModel, self).__init__()
        self.path_encoder = path_encoder
        self.geno_encoder = geno_encoder
        self.fc = nn.LazyLinear(1)
        self.hidden_dim = hidden_dim
        self.path_msa = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        self.geno_msa = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )

    def forward(self, path_x, geno_x, mask):
        # path_x shape: Batch size x #patches x Feature dim
        # flatten:
        B, N, D = path_x.shape
        path_x = path_x.view(-1, D)  # shape: (Batch size * Num patches) x Feature dim
        # geno_x shape: Batch size x Num genomic features
        path_features = self.path_encoder(
            path_x
        )  # shape: (Batch size * Num patches) x Feature dim
        path_features = path_features.view(
            B, N, -1
        )  # shape: Batch size x Num patches x Feature dim
        geno_features = self.geno_encoder(geno_x)  # shape: Batch size x Feature dim

        # Self Attention Mechanism
        path_attended, _ = self.path_msa(
            path_features, path_features, path_features, key_padding_mask=mask
        )  # shape: Batch size x Num patches x Feature dim
        path_representation = torch.mean(
            path_attended, dim=1
        )  # shape: Batch size x Feature dim
        geno_features = geno_features.view(B, -1)  # shape: Batch size x Feature dim
        # concat path and genomic features
        combined_features = torch.cat(
            (path_representation, geno_features), dim=1
        )  # shape: Batch size x (2 * Feature dim)
        preds = self.fc(combined_features)  # shape: Batch size x 1
        return preds.squeeze()  # shape: Batch size
