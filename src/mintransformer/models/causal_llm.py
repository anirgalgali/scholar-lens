import torch
import torch.nn as nn
from ..layers import Embedding, Linear, RotaryPositionalEmbedding
from ..blocks import Decoder
from ..config import ArchitectureConfig

class TransformerLM(nn.Module):
    def __init__(self, config: ArchitectureConfig,
                 device = None, dtype = None):
        super().__init__()
        self.d_model = config.d_model
        self.context_length = config.context_length
        self.vocab_size = config.vocab_size
        self.num_decoder_layers = config.num_decoder_layers
        self.num_encoder_layers = config.num_encoder_layers
        self.token_embed = Embedding(self.vocab_size, self.d_model, device = device, dtype = dtype)
        self.rope_module = RotaryPositionalEmbedding(theta = config.theta,
                                                     d_head=config.d_model//config.transformer.attn.n_heads,
                                                     context_length=self.context_length)
        
        self.decoder = Decoder(config, rope_module = self.rope_module, device= device, dtype = dtype)
        
        self.lm_head = Linear(in_features = self.d_model,
                                  out_features=self.vocab_size,
                                  device = device, dtype = dtype)
        
        if config.share_embed_lmhead_wts:
            self.lm_head.weight = self.token_embed.weight

    def forward(self, input: torch.Tensor):
        _, seq_len = input.shape
        position_ids = torch.arange(0,seq_len).view(1, - 1)
        embedded = self.token_embed(input)
        embedded = self.decoder(embedded, position_ids)
        return self.lm_head(embedded)
    

    

        