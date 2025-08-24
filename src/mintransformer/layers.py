import torch
import torch.nn as nn
import math
from einops import rearrange
from src.mintransformer.functional import scaled_dot_product_attention


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
        if self.bias is not None:
            torch.nn.init.trunc_normal_(
                self.bias, mean=0, std=init_std, a=-3 * init_std, b=3 * init_std
            )

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
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.device = device
        self.dtype = dtype
        self.proj_up = Linear(
            self.in_features, self.out_features, device=self.device, dtype=self.dtype
        )
        self.proj_gate = Linear(
            self.in_features, self.out_features, device=self.device, dtype=self.dtype
        )
        self.activation = activation

    def forward(self, input):
        out_proj = self.proj_up(input)
        out_gate = self.proj_gate(input)
        return self.activation(out_proj) * out_gate


#### EMBEDDING AND UNEMBEDDING LAYERS
class Embedding(nn.Module):

    def __init__(
        self, num_embeddings: int, embedding_dim: int, device=None, dtype=None
    ):
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
        self._initialize()

    def _initialize(self):
        torch.nn.init.trunc_normal_(self.weight, 0.0, std=1.0, a=-3, b=3)

    def forward(self, token_ids):
        return self.weight[token_ids]


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
        ff_dim: int,
        activation=nn.Module,
        device=None,
        dtype=None,
    ):

        super().__init__()
        self.d_model = embedding_dim
        self.d_ff = ff_dim
        self.activation = activation
        self.device = device
        self.dtype = dtype
        self.glu = GatedLinearUnit(
            in_features=self.d_model,
            out_features=self.d_ff,
            activation=self.activation,
            device=self.device,
            dtype=self.dtype,
        )
        self.output = Linear(
            in_features=self.d_ff, out_features=self.d_model, device=device, dtype=dtype
        )

    def forward(self, input):
        hidden = self.glu(input)
        return self.output(hidden)


#### MULTI-HEAD ATTENTION LAYER


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        use_causal_mask: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.d_model = embedding_dim
        self.n_heads = n_heads
        self.use_causal = use_causal_mask
        self.d_head = self.d_model // self.n_heads
        self.device = device
        self.dtype = dtype

        self.query_proj = Linear(
            in_features=self.d_model,
            out_features=self.d_model,
            device=self.device,
            dtype=self.dtype,
        )
        self.key_proj = Linear(
            in_features=self.d_model,
            out_features=self.d_model,
            device=self.device,
            dtype=self.dtype,
        )
        self.value_proj = Linear(
            in_features=self.d_model,
            out_features=self.d_model,
            device=self.device,
            dtype=self.dtype,
        )
        self.out_proj = Linear(
            in_features=self.d_model,
            out_features=self.d_model,
            device=self.device,
            dtype=self.dtype,
        )

    def forward(self, input: torch.Tensor):  # input is #batch x nseq x d_model

        q = rearrange(self.query_proj(input), "b s (h d) -> b h s d", h = self.n_heads)
        k = rearrange(self.key_proj(input), "b s (h d) -> b h s d", h = self.n_heads)
        v = rearrange(self.value_proj(input), "b s (h d) -> b h s d", h = self.n_heads)
        batch_dim, head_dim, seq_dim, _ = k.shape
        mask = self._construct_mask((batch_dim, head_dim, seq_dim, seq_dim))
        attn = scaled_dot_product_attention(q, k, v, mask)
        attn = rearrange(attn, " b h s d -> b s (h d)", h = self.n_heads)
        return self.out_proj(attn)

    def _construct_mask(self, mask_shape: tuple):

        mask = torch.ones(*mask_shape, dtype=bool)
        if self.use_causal:
            mask = rearrange(torch.triu(mask), "b h k q -> b h q k")

        return mask
