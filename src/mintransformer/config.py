from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SelfAttentionConfig:
    n_heads: int = 4
    bias: bool = False


@dataclass
class FFNConfig:
    is_gated: bool = True
    activation: str = "silu"
    bias: bool = False
    hidden_dim_multiplier: Optional[float] = None
    # when none (set to defaults below):
    #   hidden_dim_multiplier = 8/3 (is_gated = True)
    #   hidden_dim_multiplier = 4 (is_gated = False)


@dataclass
class TransformerConfig:
    norm_position: str = "pre"  # whether to have normalizaiton before or after computations
    norm_type: str = "layer"  # rms or layernorm
    use_causal_mask: bool = True
    ln_bias: bool = False  # bias for layer norm layers
    ffn: FFNConfig = field(default_factory=FFNConfig)
    attn: SelfAttentionConfig = field(default_factory=SelfAttentionConfig)


@dataclass
class ArchitectureConfig:
    # By default this config supports creation of decoder only models, since the
    # default value of num_encoder_layers = 0 and use_causal = True by default
    d_model: int
    num_decoder_layers: int
    vocab_size: int
    theta: Optional[float] = None # parameter for RoPE
    num_encoder_layers: Optional[int] = None
    share_embed_lmhead_wts: bool = False
    context_length: int = 512
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
