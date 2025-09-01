from typing import Optional
import torch
import torch.nn as nn
from . import layers
from . import config


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        context_length: int,
        block_config: config.TransformerConfig,
        dropout: Optional[float] = None,
        rope_module: nn.Module = None,
        device=None,
        dtype=None,
    ):

        super().__init__()
        valid_norm_positions = ["pre", "post"]

        if not block_config.norm_position.lower() in valid_norm_positions:
            raise ValueError(f"Unidentifiable norm position type: {block_config.norm_position}")

        self.d_model = d_model
        self.norm_position = block_config.norm_position
        self.norm_type = block_config.norm_type

        self.attn = layers.MultiHeadSelfAttention(
            d_model=self.d_model,
            context_length=context_length,
            attention_config=block_config.attn,
            use_causal_mask=block_config.use_causal_mask,
            rope_module=rope_module,
            device=device,
            dtype=dtype,
        )

        # The number of hidden neurons in the feed-foward MLP gets set internally since
        # are not passing it in as an explicit parameter. If using a gated architecture,
        # it gets set to (8/3)*d_model rounded off to the closest multiple of 64
        # (for maximal GPU efficiency). If using a standard architecture, it uses a multiple
        # of 4. See FFNConfig for details

        self.ff = layers.PositionWiseFeedForward(
            d_model=self.d_model,
            ffn_config=block_config.ffn,
            dropout = dropout,
            device=device,
            dtype=dtype,
        )

        if self.norm_type == "rms":
            # normalization for attention
            self.norm_attn = layers.RMSNorm(d_model=self.d_model, device=device, dtype=dtype)
            # normalization for ffn
            self.norm_ffn = layers.RMSNorm(d_model=self.d_model, device=device, dtype=dtype)

        elif self.norm_type == "layer":
            # normalization for attention
            self.norm_attn = layers.LayerNorm(d_model=self.d_model, bias=block_config.ln_bias, 
                                              device=device, dtype=dtype)
            # normalization for ffn
            self.norm_ffn = layers.LayerNorm(d_model=self.d_model, bias=block_config.ln_bias, 
                                              device=device, dtype=dtype)
        else:

            raise ValueError(f"Unknown normalization layer type: {self.norm_type}")

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


class Decoder(nn.Module):

    def __init__(self, config: config.ArchitectureConfig, 
                 rope_module: nn.Module = None,
                 device = None, dtype = None):
        super().__init__()
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model=self.d_model, 
                              context_length=config.context_length, 
                              block_config=config.transformer,
                              dropout = config.dropout, 
                              rope_module=rope_module,
                              device=device,
                              dtype = dtype) for _ in range(self.n_layers)])
        if config.transformer.norm_type == "rms":
            self.final_norm = layers.RMSNorm(d_model=self.d_model, device=device, dtype=dtype)
        elif config.transformer.norm_type == "layer":
            self.final_norm = layers.LayerNorm(d_model=self.d_model,bias=config.transformer.ln_bias,
                                                device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown normalization layer type: {config.transformer.norm_type}")

    def forward(self, input: torch.Tensor, position_ids: torch.Tensor):

        for layer in self.layers:
            input = layer(input, position_ids)

        return self.final_norm(input)
    
class Encoder(nn.Module):

    def __init__(self, config: config.ArchitectureConfig, 
                 rope_module: nn.Module = None,
                 device = None, dtype = None):
        
        super().__init__()
        if config.transformer.use_causal_mask:
            raise ValueError( "use_causal_mask should be False for an encoder")
        
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model=self.d_model, 
                              context_length=config.context_length, 
                              block_config=config.transformer,
                              dropout = config.dropout, 
                              rope_module=rope_module,
                              device=device,
                              dtype = dtype) for _ in range(self.n_layers)])
        


    def forward(self, input: torch.Tensor, position_ids: torch.Tensor):

        for layer in self.layers:
            input = layer(input, position_ids)

        return input
