from einops import rearrange
import numpy
import torch
import torch.nn.functional as F
from mintransformer.layers import Linear
from mintransformer.layers import Embedding


def test_linear(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    w1_weight = ts_state_dict[0]["layers.0.ffn.w1.weight"]
    linear_layer = Linear(d_model, d_ff, bias=False, device="cpu", dtype=torch.float32)
    linear_layer.load_state_dict({"weight": w1_weight})
    output = linear_layer(in_embeddings)
    numpy_snapshot.assert_match(output)


def test_embedding(numpy_snapshot, ts_state_dict, in_indices, vocab_size, d_model):
    embedding_weight = ts_state_dict[0]["token_embeddings.weight"]
    embedding_layer = Embedding(vocab_size, d_model, device="cpu", dtype=torch.float32)
    embedding_layer.load_state_dict({"weight": embedding_weight})
    output = embedding_layer(in_indices)
    numpy_snapshot.assert_match(output)


