from typing import Optional
import torch
import torch.nn as nn
from ..layers import Embedding, Linear, LayerNorm
from ..functional import GeLU, SiLU
from ..blocks import Encoder
from ..config import DistilBERTConfig


class DistilBERT(nn.Module):
    def __init__(self, config: DistilBERTConfig,
                 device = None, dtype = None):
        
        super().__init__()
        self.word_embeddings= Embedding(config.vocab_size, config.d_model, device = device, dtype = dtype)
        # absolute learned position embeddings
        self.pos_embeddings = Embedding(config.context_length, config.d_model, device = device, dtype = dtype)
        self.embedding_norm = LayerNorm(config.d_model, bias = True)
        self.embedding_dropout = nn.Dropout(p = config.dropout)
        self.transformer = Encoder(config, device= device, dtype = dtype)
        
    def forward(self, input: torch.Tensor):
        _, seq_len = input.shape
        position_ids = torch.arange(0,seq_len).view(1, - 1)
        embedded = self.word_embeddings(input) + self.pos_embeddings(position_ids)
        embedded = self.embedding_dropout(self.embedding_norm(embedded))
        return self.transformer(embedded)


class DistilBERTwithClassifierHead(nn.Module):
    def __init__(self, config: DistilBERTConfig,
                 num_output_classes: int,
                 classifier_dropout: Optional[float] = None,
                 classifier_activation: str = "relu",
                 device = None, dtype = None):
        
        super().__init__()
        
        self.distilbert = DistilBERT(config, device= device, dtype = dtype)
        self.pre_classifier = Linear(in_features = config.d_model,
                                  out_features=config.d_model, bias=True,
                                  device = device, dtype = dtype)
        
        self.classifier = Linear(in_features=config.d_model,
                                 out_features= num_output_classes, bias=True,
                                 device = device, dtype = dtype)
        
        if classifier_activation == "relu":
            self.classifier_activation = nn.ReLU()
        elif classifier_activation == "gelu":
            self.classifier_activation = GeLU()
        elif classifier_activation == "silu":
            self.classifier_activation = SiLU()

        self.classifier_dropout = nn.Dropout(classifier_dropout)

    def forward(self, input: torch.Tensor):
        
        embedded = self.distilbert(input)[:,:,0]
        classifier_out = self.pre_classifier(embedded)
        classifier_out = self.classifier_activation(classifier_out)
        classifier_out = self.classifier_dropout(classifier_out)
        logits = self.classifier(classifier_out)
        return logits