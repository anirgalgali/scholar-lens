from dataclasses import dataclass, field
from typing import Optional, Literal

PositionEmbeddingType = Literal["rope","absolute","none"]
NormPositionType = Literal["pre", "post"]
NormType = Literal["rms", "layer"]
ActivationType = Literal["gelu", "relu","silu"]

@dataclass
class SelfAttentionConfig:
    n_heads: int = 4
    bias: bool = False
    dropout_attn: float = 0.0

@dataclass
class FFNConfig:
    is_gated: bool = True
    activation: ActivationType = "silu"
    bias: bool = False
    hidden_dim_multiplier: Optional[float] = None
    # when none (set to defaults below):
    #   hidden_dim_multiplier = 8/3 (is_gated = True)
    #   hidden_dim_multiplier = 4 (is_gated = False)


@dataclass
class TransformerConfig:
    norm_position: NormPositionType = "pre" # whether to have normalizaiton before or after computations
    norm_type: NormType = "layer"  # rms or layernorm
    use_causal_mask: bool = True
    ln_bias: bool = False  # bias for layer norm layers
    ffn: FFNConfig = field(default_factory=FFNConfig)
    attn: SelfAttentionConfig = field(default_factory=SelfAttentionConfig)


@dataclass
class ArchitectureConfig:
    d_model: int
    vocab_size: int
    context_length: int

@dataclass
class DistilBERTConfig(ArchitectureConfig):
    d_model: int = 768
    n_layers: int = 6
    vocab_size: int = 30522
    context_length: int = 512
    pos_embedding_type: PositionEmbeddingType = "absolute"
    dropout: float = 0.1
    share_embed_lmhead_wts: bool = False
    transformer: Optional[TransformerConfig] = None 

    def __post_init__(self):
        if self.transformer is None:

            self.transformer = TransformerConfig(norm_position="post",
                                                 norm_type = "layer",
                                                 use_causal_mask=False,
                                                 ln_bias = True,
                                                 ffn = FFNConfig(
                                                 is_gated=False,
                                                 activation = "gelu",
                                                 bias = True),
                                                 attn = SelfAttentionConfig(
                                                 n_heads=12,
                                                 bias = True))

@dataclass
class DecoderLMConfigTest(ArchitectureConfig):
    n_layers: int
    theta: float
    transformer: TransformerConfig
    pos_embedding_type: PositionEmbeddingType = "rope"
    dropout: float = 0.0
    share_embed_lmhead_wts: bool = False
    # def __post_init__(self):
    #     if self.pos_embedding_type != 'rope' and self.theta is not None:
    #         raise ValueError(
    #             f"rope_theta is set to {self.theta} but pos_embedding_type "
    #             f"is '{self.pos_embedding_type}'. theta is only applicable when "
    #             "pos_embedding_type is 'rope'."
    #         )
        

