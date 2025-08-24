from einops import rearrange
import numpy
import torch
import torch.nn.functional as F
from mintransformer.layers import (
    Linear,
    Embedding,
    RMSNorm,
    LayerNorm,
    PositionWiseFeedForward,
)
from src.mintransformer.functional import SiLU


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


def test_rmsnorm(numpy_snapshot, ts_state_dict, in_embeddings):
    state_dict, _ = ts_state_dict
    reference_weights = state_dict["layers.1.ln1.weight"]
    d_model = reference_weights.shape[0]
    rms_norm_layer = RMSNorm(
        d_hidden=d_model, eps=1e-5, device="cpu", dtype=torch.float32
    )
    rms_norm_layer.load_state_dict({"weight": reference_weights})
    actual_output = rms_norm_layer(in_embeddings)
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_layernorm(in_embeddings):
    d_model = in_embeddings.shape[-1]
    my_layernorm = LayerNorm(d_hidden=d_model, device="cpu", dtype=torch.float32)
    reference_layernorm = torch.nn.LayerNorm(normalized_shape=d_model)
    my_layernorm.load_state_dict(
        {"weight": reference_layernorm.weight, "bias": reference_layernorm.bias}
    )
    my_output = my_layernorm(in_embeddings)
    reference_output = reference_layernorm(in_embeddings)
    assert torch.allclose(
        my_output, reference_output, atol=1e-6
    ), "Forward pass outputs do not match!"

def test_swiglu(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    w1_weight, w2_weight, w3_weight = [
        ts_state_dict[0][f"layers.0.ffn.{k}.weight"] for k in ["w1", "w2", "w3"]
    ]

    swiglu = PositionWiseFeedForward(
        embedding_dim=d_model,
        ff_dim=d_ff,
        activation=SiLU(),
        device="cpu",
        dtype=torch.float32,
    )
    swiglu.load_state_dict(
        {
            "glu.proj_up.weight": w1_weight,
            "glu.proj_gate.weight": w3_weight,
            "output.weight": w2_weight,
        }
    )
    actual_output = swiglu(in_embeddings)
    numpy_snapshot.assert_match(actual_output, atol=1e-5)
