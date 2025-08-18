import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.device = device
        self.weight = nn.Parameter(
            torch.empty(
                (out_features, in_features), device=self.device, dtype=self.dtype
            ),
            requires_grad=True,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty((out_features,), device=self.device, dtype=self.dtype),
                requires_grad=True,
            )
        else:
            self.register_parameter("bias", None)

        self._initialize()

    def _initialize(self):
        init_std = math.sqrt(2.0 / (self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(
            self.weight, 0.0, std=init_std, a=-3 * init_std, b=3 * init_std
        )
        if self.bias:
            torch.nn.init.trunc_normal_(
                self.bias, mean=0, std=init_std, a=-3 * init_std, b=3 * init_std
            )

    def forward(self, input):  # input is [batch_dims, dim]
        if self.bias:
            return input @ self.weight.T + self.bias
        return input @ self.weight.T


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(
            torch.empty(
                (num_embeddings, embedding_dim), device=self.device, dtype=self.dtype
            ),
            requires_grad=True,
        )

    def _initialize(self):
        torch.nn.init.trunc_normal_(self.weight, 0.0, std=1.0, a=-3, b=3)

    def forward(self, token_ids):
        return self.weight[token_ids]
