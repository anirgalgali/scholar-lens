import torch
import torch.nn as nn
import math
from typing import Optional
from einops import rearrange, einsum
from src.mintransformer.functional import scaled_dot_product_attention, SiLU, GeLU
from collections import OrderedDict


#### LINEAR AND GATED-LINEAR LAYERS
class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.device = device
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=self.device, dtype=self.dtype),
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
        torch.nn.init.trunc_normal_(self.weight, 0.0, std=init_std, a=-3 * init_std, b=3 * init_std)
        if self.bias is not None:
            torch.nn.init.trunc_normal_(self.bias, mean=0, std=init_std, a=-3 * init_std, b=3 * init_std)

    def forward(self, input):  # input is [batch_dims, dim]
        if self.bias is not None:
            return input @ self.weight.T + self.bias
        return input @ self.weight.T


class GatedLinearUnit(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.device = device
        self.dtype = dtype
        self.bias = bias
        self.proj_up = Linear(self.in_features, self.out_features, bias=self.bias, device=self.device, dtype=self.dtype)
        self.proj_gate = Linear(
            self.in_features, self.out_features, bias=self.bias, device=self.device, dtype=self.dtype
        )
        self.activation = activation

    def forward(self, input):
        out_proj = self.proj_up(input)
        out_gate = self.proj_gate(input)
        return self.activation(out_proj) * out_gate


#### EMBEDDING AND UNEMBEDDING LAYERS
class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=self.device, dtype=self.dtype),
            requires_grad=True,
        )
        self._initialize()

    def _initialize(self):
        torch.nn.init.trunc_normal_(self.weight, 0.0, std=1.0, a=-3, b=3)

    def forward(self, ids):
        # ids could either be position or token ids
        return self.weight[ids]


#### NORMALIZATION LAYERS (LAYERNORM & RMSNORM)
class LayerNorm(nn.Module):

    def __init__(
        self,
        d_hidden: int,
        eps: float = 1e-5,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.device = device
        self.dtype = dtype
        self.eps = eps
        # calling what's typically gamma as 'weight' to be consistent with PyTorch definitions
        self.weight = nn.Parameter(
            torch.empty((d_hidden,), device=self.device, dtype=self.dtype),
            requires_grad=True,
        )
        # calling what's typically beta as 'bias' to be consistent with PyTorch definitions
        if bias:
            self.bias = nn.Parameter(
                torch.empty((d_hidden,), device=self.device, dtype=self.dtype),
                requires_grad=True,
            )
        else:
            self.register_parameter("bias", None)

        self._initialize()

    def _initialize(self):
        torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input):  # input is of size batch x n_seq x d_hidden
        in_dtype = input.dtype  # IN case you are using mixed precisiion
        input = input.to(torch.float32)
        input_mean = torch.mean(input, dim=-1, keepdim=True)
        centered_input = input - input_mean
        input_var = torch.mean(centered_input**2, dim=-1, keepdim=True) + self.eps
        scaled_centered_input = centered_input / torch.sqrt(input_var)
        output = scaled_centered_input * self.weight[None, None, :]
        if self.bias is not None:
            output = output + self.bias[None, None, :]

        return output.to(in_dtype)


class RMSNorm(nn.Module):

    def __init__(self, d_hidden: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_hidden = d_hidden
        self.device = device
        self.dtype = dtype
        self.eps = eps
        self.weight = nn.Parameter(
            torch.empty((d_hidden,), device=self.device, dtype=self.dtype),
            requires_grad=True,
        )
        self._initialize()

    def _initialize(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, input):  # input is of size batch x n_seq x d_hidden
        in_dtype = input.dtype
        input.to(torch.float32)
        rms = torch.sqrt(torch.mean(input**2, dim=-1, keepdim=True) + self.eps)
        output = (input / rms) * self.weight[None, None, :]
        return output.to(in_dtype)


#### POSITION-WISE FEED-FORWARD NETWORK
class PositionWiseFeedForward(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        activation_type: str,
        is_gated: bool,
        ff_dim: Optional[int] = None,
        bias: bool = False,
        device=None,
        dtype=None,
    ):

        super().__init__()
        all_activations = {
            "silu": SiLU,
            "gelu": GeLU,
            "relu": nn.ReLU,  # You can use PyTorch's built-ins
        }

        if activation_type not in all_activations:
            raise ValueError(f" Unknown activation: {activation_type}")

        self.d_model = embedding_dim
        self.d_ff = ff_dim
        self.is_gated = is_gated
        self.bias = bias
        self.activation = all_activations[activation_type]()
        self.device = device
        self.dtype = dtype
        network_dict = OrderedDict()

        if self.is_gated:

            if self.d_ff is None:
                d_ff = int(self.d_model * 8 / 3)
                self.d_ff = d_ff = (d_ff + 63) & -64

            network_dict["glu"] = GatedLinearUnit(
                in_features=self.d_model,
                out_features=self.d_ff,
                activation=self.activation,
                bias=self.bias,
                device=self.device,
                dtype=self.dtype,
            )

            network_dict["fc_out"] = Linear(
                in_features=self.d_ff, out_features=self.d_model, bias=self.bias, device=device, dtype=dtype
            )

        else:

            if self.d_ff is None:
                self.d_ff = 4 * self.d_model

            network_dict["fc_in"] = Linear(
                in_features=self.d_model, out_features=self.d_ff, bias=self.bias, device=device, dtype=dtype
            )
            network_dict["actn"] = self.activation
            network_dict["fc_out"] = Linear(
                in_features=self.d_ff, out_features=self.d_model, bias=self.bias, device=device, dtype=dtype
            )

        self.net = nn.Sequential(network_dict)

    def forward(self, input):
        return self.net(input)


#### MULTI-HEAD ATTENTION LAYER
class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        max_seq_len: int,
        use_causal_mask: bool = True,
        rope_module: nn.Module = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.d_model = embedding_dim
        self.n_heads = n_heads
        self.use_causal = use_causal_mask
        self.rope = rope_module
        self.d_head = self.d_model // self.n_heads
        self.device = device
        self.dtype = dtype

        if self.use_causal:
            mask = torch.triu(torch.ones((max_seq_len, max_seq_len), dtype=bool), diagonal=1)
            self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len))

        self.qkv_proj = Linear(
            in_features=self.d_model,
            out_features=3 * self.d_model,
            device=self.device,
            dtype=self.dtype,
        )

        self.out_proj = Linear(
            in_features=self.d_model,
            out_features=self.d_model,
            device=self.device,
            dtype=self.dtype,
        )

    def forward(self, input: torch.Tensor, pos_ids: torch.Tensor = None):  # input is #batch x nseq x d_model

        seq_len = input.shape[1]
        # qkv is of size batch_size (b) x n_seq (s) x (p*d_model), where p = 3 (query, key, val)
        qkv = self.qkv_proj(input)
        # the third dimensions of qkv can be split as (p* d_model) -> (p h, d), where h*d = d_model
        q, k, v = rearrange(qkv, "b s (p h d) -> p b h s d", p=3, h=self.n_heads)

        if self.rope is not None:
            q = self.rope(q, pos_ids)
            k = self.rope(k, pos_ids)

        # truncating mask to input sequence length
        mask = self.causal_mask[:, :, :seq_len, :seq_len] if self.use_causal else None
        attn = scaled_dot_product_attention(q, k, v, mask)
        attn = rearrange(attn, " b h s d -> b s (h d)", h=self.n_heads)
        return self.out_proj(attn)


#### POSITIONAL EMBEDDING LAYERS
class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, thetaN: float, d_k: int, max_seq_len: int):
        super().__init__()
        self.thetaN = thetaN
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        pos_array = torch.arange(0, self.max_seq_len, dtype=torch.float32)
        dim_array = torch.arange(1, self.d_k // 2 + 1, dtype=torch.float32)
        freq_array = thetaN ** (-2.0 * (dim_array - 1) / self.d_k)
        freq_array = freq_array[:, None].repeat(1, 2).reshape(-1)
        all_thetas = einsum(pos_array, freq_array, "s, d -> s d")
        self.register_buffer(
            "cos_theta",
            torch.cos(all_thetas).view(1, self.max_seq_len, self.d_k),
            persistent=False,
        )
        self.register_buffer(
            "sin_theta",
            torch.sin(all_thetas).view(1, self.max_seq_len, self.d_k),
            persistent=False,
        )

        flip_matrix = torch.diagflat(
            -1
            * torch.ones(
                1,
            ),
            offset=1,
        )
        flip_matrix += torch.diagflat(
            1
            * torch.ones(
                1,
            ),
            offset=-1,
        )
        self.register_buffer(
            "flip_matrix", torch.block_diag(*flip_matrix[None, :, :].expand(self.d_k // 2, -1, -1)), persistent=False
        )

    def forward(self, input: torch.tensor, token_positions: torch.tensor):

        input_rot = (self.cos_theta[..., token_positions, :] * input) + self.sin_theta[..., token_positions, :] * (
            einsum(self.flip_matrix, input, "d1 d2, ... s d2 -> ... s d1")
        )
        return input_rot
