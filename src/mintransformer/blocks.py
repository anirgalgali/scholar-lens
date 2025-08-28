from typing import Optional
import torch
import torch.nn as nn
from . import layers


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        norm_position: str,
        norm_type: str,
        causal_attn: bool,
        ffn_activation: str,
        ffn_type: str,
        ff_dim: Optional[int] = None,
        bias: bool = False,  # whether to include bias term in Feedforward network and layernorm
        rope_module: nn.Module = None,
        device=None,
        dtype=None,
    ):

        super().__init__()
        valid_norm_positions = ["pre", "post"]

        if ffn_type.lower() == "gated":
            is_gated = True
        elif ffn_type.lower() == "standard":
            is_gated = False
        else:
            raise ValueError(f"Unidentifiable feed-forward network architecture type: {ffn_type}")

        if not norm_position.lower() in valid_norm_positions:
            raise ValueError(f"Unidentifiable norm position type: {norm_position}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads
        self.max_seq_len = max_seq_len
        self.causal_attn = causal_attn
        self.norm_position = norm_position
        self.norm_type = norm_type
        self.rope = rope_module
        self.ffn_activation = ffn_activation
        self.ffn_type = ffn_type
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.attn = layers.MultiHeadSelfAttention(
            embedding_dim=self.d_model,
            n_heads=self.n_heads,
            max_seq_len=self.max_seq_len,
            use_causal_mask=self.causal_attn,
            rope_module=rope_module,
            device=self.device,
            dtype=self.dtype,
        )

        # The number of hidden neurons in the feed-foward MLP gets set internally since
        # are not passing it in as an explicit parameter. If using a gated architecture,
        # it gets set to (8/3)*d_model rounded off to the closest multiple of 64
        # (for maximal GPU efficiency). If using a standard architecture,
        self.ff = layers.PositionWiseFeedForward(
            embedding_dim=self.d_model,
            activation_type=ffn_activation,
            is_gated=is_gated,
            bias=self.bias,
            ff_dim=ff_dim,
            device=self.device,
            dtype=self.dtype,
        )

        if self.norm_type == "rms":
            # normalization for attention
            self.norm_attn = layers.RMSNorm(d_hidden=self.d_model, device=self.device, dtype=self.dtype)
            # normalization for ffn
            self.norm_ffn = layers.RMSNorm(d_hidden=self.d_model, device=self.device, dtype=self.dtype)

        elif self.norm_type == "layer":
            # normalization for attention
            self.norm_attn = layers.LayerNorm(d_hidden=self.d_model, bias=self.bias, device=device, dtype=dtype)
            # normalization for ffn
            self.norm_ffn = layers.RMSNorm(d_hidden=self.d_model, bias=self.bias, device=self.device, dtype=self.dtype)
        else:

            raise ValueError(f"Unknown normalization llayer type: {self.norm_type}")

    def _attention_computation(self, input: torch.Tensor, pos_ids: torch.tensor = None):

        if self.norm_position == "pre":
            return input + self.attn(self.norm_attn(input), pos_ids=pos_ids)
        elif self.norm_position == "post":
            return self.norm_attn(input + self.attn(input, pos_ids=pos_ids))

    def _feedforward_computation(self, input: torch.Tensor):

        if self.norm_position == "pre":
            return input + self.ff(self.norm_ffn(input))
        elif self.norm_position == "post":
            return self.norm_ffn(input + self.ff(input))

    def forward(self, input: torch.Tensor, position_ids: torch.Tensor = None):

        attn_out = self._attention_computation(input, position_ids)
        return self._feedforward_computation(attn_out)
