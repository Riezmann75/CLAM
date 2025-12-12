import torch.nn as nn
import torch


class PPEG(nn.Module):
    def __init__(self, hidden_dim):
        super(PPEG, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)

    def forward(self, x):
        # shape of x: (B, N + 1, D)
        B, N, D = x.shape
        assert D == self.hidden_dim, "Input feature dimension must match hidden_dim"
        cls_token = x[:, 0, :]  # (B, 1, D)
        x_patch = x[:, 1:, :]  # (B, N, D)
        H = W = int(N**0.5)
        x_patch = x_patch.transpose(1, 2).reshape(B, D, H, W)  # (B, D, H, W)
        x_conv = self.conv1(x_patch) + self.conv2(x_patch) + self.conv3(x_patch)  # (B, D, H, W)
        x_conv = x_conv.flatten(2).transpose(1, 2)  # (B, N, D)
        x = x + x_conv
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)  # (B, N + 1, D)
        return x
