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
    MultiHeadSelfAttention,
    RotaryPositionalEmbedding,
)
from src.mintransformer.functional import SiLU, scaled_dot_product_attention


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
    rms_norm_layer = RMSNorm(d_hidden=d_model, eps=1e-5, device="cpu", dtype=torch.float32)
    rms_norm_layer.load_state_dict({"weight": reference_weights})
    actual_output = rms_norm_layer(in_embeddings)
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_layernorm(in_embeddings):
    d_model = in_embeddings.shape[-1]
    my_layernorm = LayerNorm(d_hidden=d_model, device="cpu", dtype=torch.float32)
    reference_layernorm = torch.nn.LayerNorm(normalized_shape=d_model)
    my_layernorm.load_state_dict({"weight": reference_layernorm.weight, "bias": reference_layernorm.bias})
    my_output = my_layernorm(in_embeddings)
    reference_output = reference_layernorm(in_embeddings)
    assert torch.allclose(my_output, reference_output, atol=1e-6), "Forward pass outputs do not match!"


def test_swiglu(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    w1_weight, w2_weight, w3_weight = [ts_state_dict[0][f"layers.0.ffn.{k}.weight"] for k in ["w1", "w2", "w3"]]

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


def test_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    actual_output = scaled_dot_product_attention(Q=q, K=k, V=v, mask=~mask)
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_4d_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    # Shape: (batch_size, num_heads, seq_len, d_k)
    q, k, v = (rearrange(x, "(batch head) seq d -> batch head seq d", head=2) for x in (q, k, v))
    mask = rearrange(mask, "(batch head) query key -> batch head query key", head=2)

    actual_output = scaled_dot_product_attention(Q=q, K=k, V=v, mask=~mask)
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_multihead_self_attention(numpy_snapshot, in_embeddings, d_model, n_heads, ts_state_dict):
    d, _ = ts_state_dict
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]

    max_seq_len = 2 * in_embeddings.shape[1]
    attention_layer = MultiHeadSelfAttention(
        embedding_dim=d_model,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        use_causal_mask=True,
        device="cpu",
        dtype=torch.float32,
    )

    qkv_weights = torch.concatenate((q_proj_weight, k_proj_weight, v_proj_weight), dim=0)

    attention_layer.load_state_dict({"qkv_proj.weight": qkv_weights, "out_proj.weight": o_proj_weight}, strict=False)

    actual_output = attention_layer(in_embeddings)
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_rope(numpy_snapshot, in_embeddings, d_model, theta, n_queries, pos_ids):

    rope = RotaryPositionalEmbedding(thetaN=theta, d_k=d_model, max_seq_len=n_queries)
    output = rope(input=in_embeddings, token_positions=pos_ids)
    numpy_snapshot.assert_match(output, atol=1e-6)


def test_multihead_self_attention_with_rope(
    numpy_snapshot,
    in_embeddings,
    d_model,
    n_heads,
    ts_state_dict,
    n_keys,
    theta,
    pos_ids,
):
    d, _ = ts_state_dict
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]

    pos_ids = rearrange(pos_ids, "seq -> 1 seq")
    max_seq_len = 2 * in_embeddings.shape[1]
    rope = RotaryPositionalEmbedding(thetaN=theta, d_k=n_keys, max_seq_len=max_seq_len)

    attention_layer_with_rope = MultiHeadSelfAttention(
        embedding_dim=d_model,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        rope_module=rope,
        use_causal_mask=True,
        device="cpu",
        dtype=torch.float32,
    )
    qkv_weights = torch.concatenate((q_proj_weight, k_proj_weight, v_proj_weight), dim=0)
    attention_layer_with_rope.load_state_dict(
        {"qkv_proj.weight": qkv_weights, "out_proj.weight": o_proj_weight}, strict=False
    )
    actual_output = attention_layer_with_rope(in_embeddings, pos_ids)
    numpy_snapshot.assert_match(actual_output, atol=1e-6)
