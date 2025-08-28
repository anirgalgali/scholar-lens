import torch
import torch.nn.functional as F
from src.mintransformer.blocks import TransformerBlock
from src.mintransformer.layers import RotaryPositionalEmbedding
from src.mintransformer.utils import state_dict_mapper_tests_transformerblock


def test_transformer_block(numpy_snapshot, ts_state_dict, in_embeddings, d_model, n_heads, d_ff, n_keys, theta):

    seq_len = in_embeddings.shape[1]
    rope_module = RotaryPositionalEmbedding(thetaN=theta, d_k=n_keys, max_seq_len=seq_len)
    transformer = TransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        max_seq_len=seq_len,
        norm_position="pre",
        norm_type="rms",
        causal_attn=True,
        ffn_activation="silu",
        ffn_type="gated",
        ff_dim=d_ff,
        bias=False,
        rope_module=rope_module,
        device="cpu",
    )

    _ = state_dict_mapper_tests_transformerblock(ts_state_dict[0], transformer)
    pos_ids = torch.arange(0, seq_len).view(1, -1)
    actual_output = transformer(in_embeddings, pos_ids)
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )
