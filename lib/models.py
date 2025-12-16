import math
from typing import Optional
import pandas as pd
import torch
from torch import nn
from torchvision import models
from transformers import ViTModel


class ResnetEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super(ResnetEncoder, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(
            *list(self.resnet.children())[6:-1]
        )  # remove last fc layer
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.LazyLinear(hidden_dim)

    def forward(self, x):
        # x shape: Batch size x 512 x 28 x 28
        x = self.resnet(x)  # shape: Batch size x 2048 x 1 x 1
        x = torch.flatten(x, start_dim=1)  # shape: Batch size x 2048
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super(ImageEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(hidden_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class ViTHeadEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        **kwargs
    ):
        super(ViTHeadEncoder, self).__init__()
        self.model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        original_layer = self.model.encoder.layer[-1]

        # 2. Reconstruct the architecture EXACTLY as it is in Phikon
        self.norm = nn.LayerNorm(input_dim)

        # The "Intermediate" (Expansion) part
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()

        # The "Output" (Contraction) part
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(0.0)  # Phikon default is 0.0

        self.output_dim = output_dim if output_dim is not None else hidden_dim
        if self.output_dim != input_dim:
            self.final_fc = nn.Linear(input_dim, self.output_dim)
        else:
            self.final_fc = None

        with torch.no_grad():
            self.norm.weight.copy_(original_layer.layernorm_after.weight)
            self.norm.bias.copy_(original_layer.layernorm_after.bias)
            self.fc1.weight.copy_(original_layer.intermediate.dense.weight)
            self.fc1.bias.copy_(original_layer.intermediate.dense.bias)
            self.fc2.weight.copy_(original_layer.output.dense.weight)
            self.fc2.bias.copy_(original_layer.output.dense.bias)

    def forward(self, x):
        # shape x: batch size * seq len * input_dim
        x_norm = self.norm(x)
        x_intermediate = self.fc1(x_norm)
        x_activated = self.act(x_intermediate)
        x_output = self.fc2(x_activated)
        x_dropped = self.dropout(x_output)
        output = x_dropped + x
        if self.final_fc is not None:
            x = self.final_fc(output)
            return x
        return output[
            :, 0, :
        ]  # return cls token representation batch size * output_dim


class ViTTransformers(nn.Module):
    def __init__(self, input_dim: int, output_dim: Optional[int] = None, **kwargs):
        super(ViTTransformers, self).__init__()
        full_model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pretrained_layer = full_model.encoder.layer[-1]
        self.final_norm = full_model.layernorm
        self.fc = nn.Linear(768, output_dim)

        if self.output_dim != self.input_dim:
            self.final_fc = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.final_fc = None

        del full_model

    def forward(self, x):
        layer_output = self.pretrained_layer(x)
        normalized_output = self.final_norm(layer_output)
        cls_token = normalized_output[:, 0, :]  # shape batch size * 768
        if self.final_fc:
            return self.final_fc(cls_token)
        return cls_token


class GatedAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super(GatedAttentionPooling, self).__init__()
        self.attention_V = nn.Linear(hidden_dim, hidden_dim)
        self.attention_U = nn.Linear(hidden_dim, hidden_dim)
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: Batch size x Num patches x Feature dim
        # mask shape: Batch size x Num patches (True for padding positions)
        A_V = torch.tanh(
            self.attention_V(x)
        )  # shape: Batch size x Num patches x Feature dim
        A_U = torch.sigmoid(
            self.attention_U(x)
        )  # shape: Batch size x Num patches x Feature dim
        A = A_V * A_U  # shape: Batch size x Num patches x Feature dim
        A = self.attention_weights(A).squeeze(-1)  # shape: Batch size x Num patches

        A = torch.softmax(A, dim=1)  # shape: Batch size x Num patches

        A = A.unsqueeze(-1)  # shape: Batch size x Num patches x 1
        M = torch.sum(A * x, dim=1)  # shape: Batch size x Feature dim
        return M, A.squeeze(-1)  # return pooled features and attention scores


class PositionalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_width: int = 100000,
        max_height: int = 100000,
        temperature: float = 10000.0,
    ):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.max_width = max_width
        self.max_height = max_height
        self.temperature = temperature
        self.dim_x = self.d_model // 2
        self.dim_y = self.d_model - self.dim_x
        self.scale = 2 * math.pi

    def forward(self, coordinates):
        # coordinates shape:  #batch * #patches * 2
        x = coordinates[:, :, 0]  # shape:  #batch * #patches
        y = coordinates[:, :, 1]  # shape:  #batch * #patches
        x_norm = x / self.max_width * self.scale  # normalize to [0, 2*pi]
        y_norm = y / self.max_height * self.scale  # normalize to [0, 2*pi]
        # div term has shape (dim/2,)
        div_term_x = self.temperature ** (
            2 * torch.arange(0, self.dim_x, 2, device=coordinates.device) / self.dim_x
        )
        div_term_y = self.temperature ** (
            2 * torch.arange(0, self.dim_y, 2, device=coordinates.device) / self.dim_y
        )

        pe_x = x_norm.unsqueeze(-1) / div_term_x  # shape: #batch * #patches * (dim_x/2)
        pe_y = y_norm.unsqueeze(-1) / div_term_y  # shape: #batch * #patches * (dim_y/2)

        pe_x = torch.concat(
            (torch.sin(pe_x), torch.cos(pe_x)), dim=-1
        )  # shape: #batch * #patches * dim_x
        pe_y = torch.concat(
            (torch.sin(pe_y), torch.cos(pe_y)), dim=-1
        )  # shape: #batch * #patches * dim_y

        pe = torch.concat((pe_x, pe_y), dim=-1)  # shape: #batch * #patches * d_model
        return pe


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
        if len(preds.shape) == 0:
            return torch.sum(
                torch.exp(preds) * failure_times - is_observed * preds, dim=0
            )
        return (
            1
            / len(preds)
            * torch.sum(torch.exp(preds) * failure_times - is_observed * preds, dim=0)
        )


class SurvivalModel(nn.Module):
    def __init__(
        self,
        path_encoder: ResnetEncoder,
        geno_encoder: GenomicEncoder,
        hidden_dim: int = 128,
        use_positional_encoding: bool = True,
        use_gated_attention: bool = True,
        use_transformer: bool = False,
        unimodal: bool = True,
        num_transformer_layers: int = 1,
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
        self.positional_encoder = PositionalEncoder(
            d_model=hidden_dim, max_width=1000, max_height=1000
        )
        self.attention_pooling = GatedAttentionPooling(hidden_dim=hidden_dim)
        self.path_mlp = nn.Sequential(
            nn.LazyLinear(hidden_dim * 2),
            nn.LeakyReLU(0.1),
            nn.LazyLinear(hidden_dim),
        )
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=num_transformer_layers,
            enable_nested_tensor=False,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.use_positional_encoding = use_positional_encoding
        self.use_gated_attention = use_gated_attention
        self.use_transformer = use_transformer
        assert (
            self.use_gated_attention + self.use_transformer <= 1
        ), "Cannot use both gated attention and transformer cls token pooling."
        self.unimodal = unimodal

    def forward(self, path_x, geno_x, coordinates, mask):
        # path_x shape: Batch size x #patches x Feature dim
        # coordinates shape: Batch size x #patches x 2

        # flatten:
        B, N, *hidden_dims = path_x.shape
        # reshape to B*N x Feature dim
        path_x = path_x.view(B * N, *hidden_dims)
        # geno_x shape: Batch size x Num genomic features
        path_features = self.path_encoder(
            path_x
        )  # shape: (Batch size * Num patches) x Feature dim
        path_features = path_features.view(
            B, N, -1
        )  # shape: Batch size x Num patches x Feature dim

        if self.use_positional_encoding:
            coordinates = self.positional_encoder(
                coordinates
            )  # shape: Batch size x Num patches x Feature dim
            path_features = path_features + coordinates  # add positional encoding
        # Self Attention Mechanism
        if self.use_transformer:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # shape: Batch size x 1 x Feature dim
            path_features = torch.cat(
                (cls_tokens, path_features), dim=1
            )  # shape: Batch size x (1 + Num patches) x Feature dim
            # update mask to account for cls token
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=mask.device)
            mask = torch.cat(
                (cls_mask, mask), dim=1
            )  # shape: Batch size x (1 + Num patches)
            path_attended = self.transformer_encoder(
                path_features, src_key_padding_mask=mask
            )
            # extract cls token representation
        else:
            path_attended, _ = self.path_msa(
                path_features, path_features, path_features, key_padding_mask=mask
            )  # shape: Batch size x Num patches x Feature dim

        if self.use_gated_attention:
            path_representation, _ = self.attention_pooling(path_attended)
        elif self.use_transformer:
            path_representation = path_attended[
                :, 0, :
            ]  # shape: Batch size x Feature dim
        else:
            path_representation = path_attended.mean(dim=1)

        if self.unimodal:
            preds = self.fc(path_representation)  # shape: Batch size x 1
            return preds.squeeze()  # shape: Batch size
        geno_features = self.geno_encoder(geno_x)  # shape: Batch size x Feature dim
        geno_features = geno_features.view(B, -1)  # shape: Batch size x Feature dim
        # concat path and genomic features
        combined_features = torch.cat(
            (path_representation, geno_features), dim=1
        )  # shape: Batch size x (2 * Feature dim)
        preds = self.fc(combined_features)  # shape: Batch size x 1
        return preds.squeeze()  # shape: Batch size
 