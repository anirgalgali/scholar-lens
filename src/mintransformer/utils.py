import torch
import torch.nn as nn
from collections import OrderedDict
from src.mintransformer.blocks import TransformerBlock
from src.mintransformer.models.causal_llm import TransformerLM
from src.mintransformer.config import ArchitectureConfig, TransformerConfig, SelfAttentionConfig, FFNConfig
import os
def keymap_for_transformerblock_testfixture(dest_keys: list, source_keys: list):

    key_map = {}
    source_keys_set = set(source_keys)
    replacements = [
        ("attn.out_proj.weight", "attn.output_proj.weight"),
        ("ff.net.glu.proj_up.weight", "ffn.w1.weight"),
        ("ff.net.glu.proj_gate.weight", "ffn.w3.weight"),
        ("ff.net.fc_out.weight", "ffn.w2.weight"),
        ("norm_attn.weight", "ln1.weight"),
        ("norm_ffn.weight", "ln2.weight"),
    ]

    for dest_key in dest_keys:

        if "qkv_proj" in dest_key:
            continue

        if "causal_mask" in dest_key:
            continue

        candidate_source_key = dest_key
        for dest_pattern, source_pattern in replacements:
            candidate_source_key = candidate_source_key.replace(dest_pattern, source_pattern)

        if candidate_source_key in source_keys_set:
            key_map[dest_key] = candidate_source_key
        else:
            print(f"Warning: Could not find a matching source key for {dest_key}")

    return key_map


def keymap_for_transformerlm_testfixture(dest_keys: list, source_keys: list):

    key_map = {}
    source_keys_set = set(source_keys)
    replacements = [
        ("token_embed.weight", "token_embeddings.weight"),
        ("attn.out_proj.weight", "attn.output_proj.weight"),
        ("ff.net.glu.proj_up.weight", "ffn.w1.weight"),
        ("ff.net.glu.proj_gate.weight", "ffn.w3.weight"),
        ("ff.net.fc_out.weight", "ffn.w2.weight"),
        ("norm_attn.weight", "ln1.weight"),
        ("norm_ffn.weight", "ln2.weight"),
        ("final_norm.weight","ln_final.weight"),
        ("lm_head.weight","lm_head.weight")]

    for dest_key in dest_keys:

        if "qkv_proj" in dest_key:
            continue

        if "causal_mask" in dest_key:
            continue

        candidate_source_key = dest_key
        for dest_pattern, source_pattern in replacements:
            candidate_source_key = candidate_source_key.replace(dest_pattern, source_pattern)

        if candidate_source_key in source_keys_set:
            key_map[dest_key] = candidate_source_key
        else:
            print(f"Warning: Could not find a matching source key for {dest_key}")

    return key_map


def state_dict_mapper_tests_transformerblock(test_sd: OrderedDict, my_model: TransformerBlock) -> dict:

    # Note this function modifies the state_dict of
    # my_model in place

    source_weight_dict = {k.replace("layers.0.", ""): v for k, v in test_sd.items() if "layers.0." in k}
    key_map = keymap_for_transformerblock_testfixture(my_model.state_dict().keys(), source_weight_dict)

    for param_name, param in my_model.named_parameters():
        if "qkv_proj" in param_name:
            prefixes = param_name.split(".")[1].split("_")[0]
            qkv_weights = []
            for prefix in prefixes:
                qkv_weights.append(source_weight_dict[f"attn.{prefix}_proj.weight"])
            qkv_weights = torch.cat(qkv_weights, dim=0)
            param.data.copy_(qkv_weights)

        else:
            if param_name in key_map:
                source_name = key_map[param_name]
                param.data.copy_(source_weight_dict[source_name])
    model_state_dict = my_model.state_dict()

    for valid_key in key_map:
        if not torch.allclose(model_state_dict[valid_key], source_weight_dict[key_map[valid_key]], atol=1e-6):
            raise AssertionError(f"Values for {valid_key} did not match while state dict loading")

    return key_map

def state_dict_mapper_tests_transformerlm(test_sd: OrderedDict, my_model: TransformerLM) -> dict:

    model_state_dict_red = {k.replace("decoder.", ""): v for k, v in my_model.state_dict().items()}
    model_state_dict_red_to_orig_map = {k.replace("decoder.", ""): k for k, _ in my_model.state_dict().items()}
    model_state_dict_orig_to_red_map = {k: k.replace("decoder.", "") for k, _ in my_model.state_dict().items()}
    key_map = keymap_for_transformerlm_testfixture(model_state_dict_red.keys(), test_sd)
       
    layer_str_match = "layers."

    for param_name, param in my_model.named_parameters():
    
        start_idx = param_name.find(layer_str_match)

        if start_idx != 1:
            layer_name = param_name[start_idx: start_idx + len(layer_str_match)+1] 
        else:
            layer_name = None

        if "qkv_proj" in param_name:
            # qkv matrices are always associated with a layer

            qkv_weights = []
            for prefix in "qkv":
                qkv_weights.append(test_sd[f"{layer_name}.attn.{prefix}_proj.weight"])
            qkv_weights = torch.cat(qkv_weights, dim=0)
            param.data.copy_(qkv_weights)
        
        else:
            if model_state_dict_orig_to_red_map[param_name] in key_map:
                source_name = key_map[model_state_dict_orig_to_red_map[param_name]]
                param.data.copy_(test_sd[source_name])
        

    model_state_dict = my_model.state_dict()
    for valid_key in key_map:
        if not torch.allclose(model_state_dict[model_state_dict_red_to_orig_map[valid_key]], test_sd[key_map[valid_key]], atol=1e-6):
            raise AssertionError(f"Values for {valid_key} did not match while state dict loading")

    return key_map

# MINE = odict_keys(['token_embed.weight', 'decoder.layers.0.attn.causal_mask', 'decoder.layers.0.attn.qkv_proj.weight', 'decoder.layers.0.attn.out_proj.weight', 'decoder.layers.0.ff.net.glu.proj_up.weight', 'decoder.layers.0.ff.net.glu.proj_gate.weight', 'decoder.layers.0.ff.net.fc_out.weight', 'decoder.layers.0.norm_attn.weight', 'decoder.layers.0.norm_ffn.weight', 'decoder.layers.1.attn.causal_mask', 'decoder.layers.1.attn.qkv_proj.weight', 'decoder.layers.1.attn.out_proj.weight', 'decoder.layers.1.ff.net.glu.proj_up.weight', 'decoder.layers.1.ff.net.glu.proj_gate.weight', 'decoder.layers.1.ff.net.fc_out.weight', 'decoder.layers.1.norm_attn.weight', 'decoder.layers.1.norm_ffn.weight', 'decoder.layers.2.attn.causal_mask', 'decoder.layers.2.attn.qkv_proj.weight', 'decoder.layers.2.attn.out_proj.weight', 'decoder.layers.2.ff.net.glu.proj_up.weight', 'decoder.layers.2.ff.net.glu.proj_gate.weight', 'decoder.layers.2.ff.net.fc_out.weight', 'decoder.layers.2.norm_attn.weight', 'decoder.layers.2.norm_ffn.weight', 'decoder.final_norm.weight', 'lm_head.weight'])


# TRUE: odict_keys(['token_embeddings.weight', 'layers.0.attn.q_proj.weight', 'layers.0.attn.k_proj.weight', 'layers.0.attn.v_proj.weight', 'layers.0.attn.output_proj.weight', 'layers.0.ffn.w1.weight', 'layers.0.ffn.w2.weight', '_layers.0.ffn.w3.weight', 'layers.0.ln1.weight', 'layers.0.ln2.weight', 'layers.1.attn.q_proj.weight', 'layers.1.attn.k_proj.weight', 'layers.1.attn.v_proj.weight', 'layers.1.attn.output_proj.weight', 'layers.1.ffn.w1.weight', 'layers.1.ffn.w2.weight', '.layers.1.ffn.w3.weight', '.layers.1.ln1.weight', 'layers.1.ln2.weight', 'layers.2.attn.q_proj.weight', 'layers.2.attn.k_proj.weight', 'layers.2.attn.v_proj.weight', 'layers.2.attn.output_proj.weight', 'layers.2.ffn.w1.weight', 'layers.2.ffn.w2.weight', 'layers.2.ffn.w3.weight', 'layers.2.ln1.weight', 'layers.2.ln2.weight', 'ln_final.weight', 'lm_head.weight'])

# if __name__ == '__main__':

#     torch.manual_seed(4)
#     batch_size = 4
#     n_queries = 12
#     n_heads = 4
#     d_head = 16
#     theta = 10000.0
#     d_ff = 128
#     d_model = n_heads * d_head
#     in_embeddings =  torch.randn(batch_size, n_queries, d_model)
#     n_layers = 3
#     vocab_size = 10000

#     config = ArchitectureConfig(d_model = d_model,
#                                     num_decoder_layers = n_layers,
#                                     vocab_size = vocab_size,
#                                     theta = theta,
#                                     transformer = TransformerConfig(norm_type = "rms",
#                                                                     attn=SelfAttentionConfig(n_heads=n_heads),
#                                                                     ffn = FFNConfig(hidden_dim_multiplier=
#                                                                                     d_ff/d_model)))
    
#     print(os.getcwd())
#     ts_state_dict = torch.load('src/mintransformer/tests/fixtures/ts_tests/model.pt', map_location="cpu")
#     ts_state_dict = {k.replace("_orig_mod.", ""): v for k, v in ts_state_dict.items()}
#     transformer_lm = TransformerLM(config, device ="cpu", dtype = torch.float32)
#     key_map = state_dict_mapper_tests_transformerlm(ts_state_dict, transformer_lm)
