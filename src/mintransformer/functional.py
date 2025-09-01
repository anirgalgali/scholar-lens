import torch
import math
from typing import Optional
import torch.nn as nn
from einops import einsum

# Scaled Dot-Product attention (The star of the show)


def scaled_dot_product_attention(Q: torch.Tensor, 
                                 K: torch.Tensor, 
                                 V: torch.Tensor, 
                                 mask: torch.Tensor = None,
                                 dropout : Optional[nn.Module] = None
):

    # Q is the output of W_{q} @ x - # batch x n_seq x d_k
    # K is the output of W_{k} @ x - # batch x n_seq x d_k
    # V is the output of W_{v} @ x - # batch x n_seq x d_v

    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... q d, ... k d -> ... q k")
    scores /= d_k**0.5  # raw scores
    # mask out with -inf pre softmax for implementing causal mechanism
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn = softmax(scores, dim=-1)

    if dropout is not None:
        attn = dropout(attn)

    output = einsum(attn, V, "... q k, ... k d -> ... q d")
    return output


# State-less activation function implementation


def gelu(x: torch.Tensor, approximate: str = "none"):

    if approximate == "none":
        cdf = lambda x: 0.5 * (1 + torch.erf(x / math.sqrt(2)))  # gaussian cdf function
        return x * cdf(x)
    elif approximate == "tanh":
        # This uses the approximation used in the original GeLu paper (Hendrycks & Gimpel, 2016)
        c = math.sqrt(2 / math.pi)
        return 0.5 * x * (1 + torch.tanh(c * (x + 0.044715 * (x**3))))


def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int):
    x_ = x - torch.amax(x, dim=dim, keepdim=True)
    numerator = torch.exp(x_)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    return numerator / denominator


# wrapping a nn.Module around the activation function to make each of them into a module


class SiLU(nn.Module):
    """Applies the Sigmoid Linear unit activation function (silu)
    This is a nn.Module wrapper around the function implementation"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return silu(x)


class GeLU(nn.Module):
    """Applies the Gaussian error Linear unit activation function (gelu)
    This is a nn.Module wrapper around the function implementation"""

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor):
        return gelu(x, self.approximate)


class Softmax(nn.Module):
    """Applies a softmax activation
    This is a nn.Module wrapper around the function implementation"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return softmax(x, dim=self.dim)
