import torch
import torch.nn.functional as F
from src.mintransformer.blocks import TransformerBlock
from src.mintransformer.layers import RotaryPositionalEmbedding
from src.mintransformer.utils import (
    state_dict_mapper_tests_transformerblock,
    state_dict_mapper_tests_transformerlm
)
from src.mintransformer.config import TransformerConfig, DecoderLMConfigTest, FFNConfig, SelfAttentionConfig
from src.mintransformer.models.causal_llm import TransformerLM

def test_transformer_block(numpy_snapshot, ts_state_dict, in_embeddings, d_model, n_heads, d_ff, n_keys, theta):

    transformer_config = TransformerConfig(norm_type = "rms",
                                           ffn=FFNConfig(hidden_dim_multiplier=d_ff/d_model),attn=SelfAttentionConfig(n_heads = n_heads))
    seq_len = in_embeddings.shape[1]
    context_length = 2*seq_len # can be set arbitarily larger than seq_len
    rope_module = RotaryPositionalEmbedding(theta=theta, d_head=n_keys, context_length = context_length)
    
    transformer = TransformerBlock(d_model=d_model,
        context_length=context_length,
        block_config = transformer_config,
        rope_module=rope_module,
        device="cpu",
        dtype=torch.float32)

    _ = state_dict_mapper_tests_transformerblock(ts_state_dict[0], transformer)
    pos_ids = torch.arange(0, seq_len).view(1, -1)
    actual_output = transformer(in_embeddings, pos_ids)
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )

def test_transformer_lm(
    numpy_snapshot,
    vocab_size,
    d_model,
    n_layers,
    n_heads,
    d_ff,
    theta,
    ts_state_dict,
    in_indices):

    config = DecoderLMConfigTest(d_model = d_model,
                                 vocab_size = vocab_size,
                                 context_length = 512,
                                 pos_embedding_type = "rope",
                                 n_layers = n_layers, 
                                 theta = theta,
                                 transformer = TransformerConfig(norm_type = "rms",
                                                                attn=SelfAttentionConfig(n_heads=n_heads),
                                                                ffn = FFNConfig(hidden_dim_multiplier=
                                                                                d_ff/d_model)))

    transformer_lm = TransformerLM(config, device ="cpu", dtype = torch.float32)
        
    _ = state_dict_mapper_tests_transformerlm(ts_state_dict[0], transformer_lm)
    actual_output = transformer_lm(in_indices)
    numpy_snapshot.assert_match(actual_output, atol=1e-4, rtol=1e-2)

def test_transformer_lm_truncated_input(
    numpy_snapshot,
    vocab_size,
    d_model,
    n_layers,
    n_heads,
    d_ff,
    theta,
    ts_state_dict,
    in_indices):


    config = DecoderLMConfigTest(d_model = d_model,
                                 vocab_size = vocab_size,
                                 context_length = 512,
                                 pos_embedding_type = "rope",
                                 n_layers = n_layers, 
                                 theta = theta,
                                 transformer = TransformerConfig(norm_type = "rms",
                                                                attn=SelfAttentionConfig(n_heads=n_heads),
                                                                ffn = FFNConfig(hidden_dim_multiplier=
                                                                                d_ff/d_model)))

    transformer_lm = TransformerLM(config, device ="cpu", dtype = torch.float32)
        
    _ = state_dict_mapper_tests_transformerlm(ts_state_dict[0], transformer_lm)


    in_indices_truncated = in_indices[..., : in_indices.shape[-1] // 2]
    truncated_actual_output = transformer_lm(in_indices_truncated)
    
    numpy_snapshot.assert_match(
        truncated_actual_output,
        atol=1e-4,
    )