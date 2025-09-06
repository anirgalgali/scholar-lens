import torch
import torch.nn as nn
from collections import OrderedDict
from src.mintransformer.blocks import TransformerBlock
from src.mintransformer.models.causal_llm import TransformerLM
from src.mintransformer.models.distilbert import DistilBERT
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

def keymap_for_distilbert_hf(dest_keys: list, source_keys: list):

    key_map = {}
    source_keys_set = set(source_keys)
    replacements = [
        ("word_embeddings.weight", "embeddings.word_embeddings.weight"),
        ("pos_embeddings.weight", "embeddings.position_embeddings.weight"),
        ("embedding_norm.weight","embeddings.LayerNorm.weight"),
        ("embedding_norm.bias","embeddings.LayerNorm.bias"),
        ("attn.out_proj.weight", "attention.out_lin.weight"),
        ("attn.out_proj.bias", "attention.out_lin.bias"),
        ("ff.net.fc_in.weight", "ffn.lin1.weight"),
        ("ff.net.fc_in.bias", "ffn.lin1.bias"),
        ("ff.net.fc_out.weight", "ffn.lin2.weight"),
        ("ff.net.fc_out.bias", "ffn.lin2.bias"),
        ("norm_attn.weight", "sa_layer_norm.weight"),
        ("norm_attn.bias", "sa_layer_norm.bias"),
        ("norm_ffn.weight", "output_layer_norm.weight"),
        ("norm_ffn.bias", "output_layer_norm.bias")]

    for dest_key in dest_keys:

        if "qkv_proj" in dest_key:
            continue
        
        if "qkv_bias" in dest_key:
            continue

        if "causal_mask" in dest_key:
            continue

        if "classifier" in dest_key:
            continue

        if "pre_classifier" in dest_key:
            continue
        
        if "layers" in dest_key:
            dest_key = dest_key.replace("layers","layer")
        
        candidate_source_key = dest_key
        for dest_pattern, source_pattern in replacements:
            candidate_source_key = candidate_source_key.replace(dest_pattern, source_pattern)

        if candidate_source_key in source_keys_set:
            if "layer" in dest_key:
                dest_key = dest_key.replace("layer","layers")
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


def state_dict_mapper_distilbert_hf(hf_sd: OrderedDict, my_model: DistilBERT) -> dict:

    keys_to_replace = ["distilbert.", "transformer."]

    dest_keys_all = []
    model_state_dict_red_to_orig_map = dict()
    model_state_dict_orig_to_red_map = dict()
    for k, v in my_model.state_dict().items():
        k_temp = k
        for r in keys_to_replace:
            k_temp = k_temp.replace(r,"")
        
        dest_keys_all.append(k_temp)
        model_state_dict_red_to_orig_map[k_temp] = k
        model_state_dict_orig_to_red_map[k] = k_temp


    source_keys_all = []
    source_red_to_orig = dict()
    for k, v in hf_sd.items():
        k_temp = k
        for r in keys_to_replace:
            k_temp = k_temp.replace(r,"")
        
        source_keys_all.append(k_temp)
        source_red_to_orig[k_temp] = k

    key_map = keymap_for_distilbert_hf(dest_keys_all, source_keys_all)
       
    layer_str_match = "layers."

    for param_name, param in my_model.named_parameters():
    
        start_idx = param_name.find(layer_str_match)

        if start_idx != -1:
            layer_name = param_name[start_idx: start_idx + len(layer_str_match)+1]
            layer_name_source = layer_name.replace("s","") 
        else:
            layer_name = None


        if "qkv_proj.weight" in param_name:
            qkv_weights = []

    #         # qkv matrices are always associated with a layer  
            for prefix in "qkv":
                qkv_weights.append(hf_sd[f"{''.join(keys_to_replace)}{layer_name_source}.attention.{prefix}_lin.weight"])
            qkv_weights = torch.cat(qkv_weights, dim=0)
            param.data.copy_(qkv_weights)

        elif "qkv_proj.bias" in param_name:
    #         # qkv matrices are always associated with a layer
            qkv_bias = []

            for prefix in "qkv":
                qkv_bias.append(hf_sd[f"{''.join(keys_to_replace)}{layer_name_source}.attention.{prefix}_lin.bias"])
            qkv_bias = torch.cat(qkv_bias, dim=0)
            param.data.copy_(qkv_bias)

        elif param_name in source_keys_all:
            param.data.copy_(hf_sd[param_name])
        
        else:

            if model_state_dict_orig_to_red_map[param_name] in key_map:
                source_name = key_map[model_state_dict_orig_to_red_map[param_name]]
                param.data.copy_(hf_sd[source_red_to_orig[source_name]])
        

    model_state_dict = my_model.state_dict()
    for valid_key in key_map:
        if not torch.allclose(model_state_dict[model_state_dict_red_to_orig_map[valid_key]], hf_sd[source_red_to_orig[key_map[valid_key]]], atol=1e-6):
            raise AssertionError(f"Values for {valid_key} did not match while state dict loading")

    return key_map

## HF
"""

odict_keys(['distilbert.embeddings.word_embeddings.weight', 'distilbert.embeddings.position_embeddings.weight', 'distilbert.embeddings.LayerNorm.weight', 'distilbert.embeddings.LayerNorm.bias', 'distilbert.transformer.layer.0.attention.q_lin.weight', 'distilbert.transformer.layer.0.attention.q_lin.bias', 'distilbert.transformer.layer.0.attention.k_lin.weight', 'distilbert.transformer.layer.0.attention.k_lin.bias', 'distilbert.transformer.layer.0.attention.v_lin.weight', 'distilbert.transformer.layer.0.attention.v_lin.bias', 'distilbert.transformer.layer.0.attention.out_lin.weight', 'distilbert.transformer.layer.0.attention.out_lin.bias', 'distilbert.transformer.layer.0.sa_layer_norm.weight', 'distilbert.transformer.layer.0.sa_layer_norm.bias', 'distilbert.transformer.layer.0.ffn.lin1.weight', 'distilbert.transformer.layer.0.ffn.lin1.bias', 'distilbert.transformer.layer.0.ffn.lin2.weight', 'distilbert.transformer.layer.0.ffn.lin2.bias', 'distilbert.transformer.layer.0.output_layer_norm.weight', 'distilbert.transformer.layer.0.output_layer_norm.bias', 'distilbert.transformer.layer.1.attention.q_lin.weight', 'distilbert.transformer.layer.1.attention.q_lin.bias', 'distilbert.transformer.layer.1.attention.k_lin.weight', 'distilbert.transformer.layer.1.attention.k_lin.bias', 'distilbert.transformer.layer.1.attention.v_lin.weight', 'distilbert.transformer.layer.1.attention.v_lin.bias', 'distilbert.transformer.layer.1.attention.out_lin.weight', 'distilbert.transformer.layer.1.attention.out_lin.bias', 'distilbert.transformer.layer.1.sa_layer_norm.weight', 'distilbert.transformer.layer.1.sa_layer_norm.bias', 'distilbert.transformer.layer.1.ffn.lin1.weight', 'distilbert.transformer.layer.1.ffn.lin1.bias', 'distilbert.transformer.layer.1.ffn.lin2.weight', 'distilbert.transformer.layer.1.ffn.lin2.bias', 'distilbert.transformer.layer.1.output_layer_norm.weight', 'distilbert.transformer.layer.1.output_layer_norm.bias', 'distilbert.transformer.layer.2.attention.q_lin.weight', 'distilbert.transformer.layer.2.attention.q_lin.bias', 'distilbert.transformer.layer.2.attention.k_lin.weight', 'distilbert.transformer.layer.2.attention.k_lin.bias', 'distilbert.transformer.layer.2.attention.v_lin.weight', 'distilbert.transformer.layer.2.attention.v_lin.bias', 'distilbert.transformer.layer.2.attention.out_lin.weight', 'distilbert.transformer.layer.2.attention.out_lin.bias', 'distilbert.transformer.layer.2.sa_layer_norm.weight', 'distilbert.transformer.layer.2.sa_layer_norm.bias', 'distilbert.transformer.layer.2.ffn.lin1.weight', 'distilbert.transformer.layer.2.ffn.lin1.bias', 'distilbert.transformer.layer.2.ffn.lin2.weight', 'distilbert.transformer.layer.2.ffn.lin2.bias', 'distilbert.transformer.layer.2.output_layer_norm.weight', 'distilbert.transformer.layer.2.output_layer_norm.bias', 'distilbert.transformer.layer.3.attention.q_lin.weight', 'distilbert.transformer.layer.3.attention.q_lin.bias', 'distilbert.transformer.layer.3.attention.k_lin.weight', 'distilbert.transformer.layer.3.attention.k_lin.bias', 'distilbert.transformer.layer.3.attention.v_lin.weight', 'distilbert.transformer.layer.3.attention.v_lin.bias', 'distilbert.transformer.layer.3.attention.out_lin.weight', 'distilbert.transformer.layer.3.attention.out_lin.bias', 'distilbert.transformer.layer.3.sa_layer_norm.weight', 'distilbert.transformer.layer.3.sa_layer_norm.bias', 'distilbert.transformer.layer.3.ffn.lin1.weight', 'distilbert.transformer.layer.3.ffn.lin1.bias', 'distilbert.transformer.layer.3.ffn.lin2.weight', 'distilbert.transformer.layer.3.ffn.lin2.bias', 'distilbert.transformer.layer.3.output_layer_norm.weight', 'distilbert.transformer.layer.3.output_layer_norm.bias', 'distilbert.transformer.layer.4.attention.q_lin.weight', 'distilbert.transformer.layer.4.attention.q_lin.bias', 'distilbert.transformer.layer.4.attention.k_lin.weight', 'distilbert.transformer.layer.4.attention.k_lin.bias', 'distilbert.transformer.layer.4.attention.v_lin.weight', 'distilbert.transformer.layer.4.attention.v_lin.bias', 'distilbert.transformer.layer.4.attention.out_lin.weight', 'distilbert.transformer.layer.4.attention.out_lin.bias', 'distilbert.transformer.layer.4.sa_layer_norm.weight', 'distilbert.transformer.layer.4.sa_layer_norm.bias', 'distilbert.transformer.layer.4.ffn.lin1.weight', 'distilbert.transformer.layer.4.ffn.lin1.bias', 'distilbert.transformer.layer.4.ffn.lin2.weight', 'distilbert.transformer.layer.4.ffn.lin2.bias', 'distilbert.transformer.layer.4.output_layer_norm.weight', 'distilbert.transformer.layer.4.output_layer_norm.bias', 'distilbert.transformer.layer.5.attention.q_lin.weight', 'distilbert.transformer.layer.5.attention.q_lin.bias', 'distilbert.transformer.layer.5.attention.k_lin.weight', 'distilbert.transformer.layer.5.attention.k_lin.bias', 'distilbert.transformer.layer.5.attention.v_lin.weight', 'distilbert.transformer.layer.5.attention.v_lin.bias', 'distilbert.transformer.layer.5.attention.out_lin.weight', 'distilbert.transformer.layer.5.attention.out_lin.bias', 'distilbert.transformer.layer.5.sa_layer_norm.weight', 'distilbert.transformer.layer.5.sa_layer_norm.bias', 'distilbert.transformer.layer.5.ffn.lin1.weight', 'distilbert.transformer.layer.5.ffn.lin1.bias', 'distilbert.transformer.layer.5.ffn.lin2.weight', 'distilbert.transformer.layer.5.ffn.lin2.bias', 'distilbert.transformer.layer.5.output_layer_norm.weight', 'distilbert.transformer.layer.5.output_layer_norm.bias', 'pre_classifier.weight', 'pre_classifier.bias', 'classifier.weight', 'classifier.bias'])

"""

## MINE
"""

odict_keys(['distilbert.word_embeddings.weight', 'distilbert.pos_embeddings.weight', 'distilbert.embedding_norm.weight', 'distilbert.embedding_norm.bias', 'distilbert.transformer.layers.0.attn.qkv_proj.weight', 'distilbert.transformer.layers.0.attn.qkv_proj.bias', 'distilbert.transformer.layers.0.attn.out_proj.weight', 'distilbert.transformer.layers.0.attn.out_proj.bias', 'distilbert.transformer.layers.0.ff.net.fc_in.weight', 'distilbert.transformer.layers.0.ff.net.fc_in.bias', 'distilbert.transformer.layers.0.ff.net.fc_out.weight', 'distilbert.transformer.layers.0.ff.net.fc_out.bias', 'distilbert.transformer.layers.0.norm_attn.weight', 'distilbert.transformer.layers.0.norm_attn.bias', 'distilbert.transformer.layers.0.norm_ffn.weight', 'distilbert.transformer.layers.0.norm_ffn.bias', 'distilbert.transformer.layers.1.attn.qkv_proj.weight', 'distilbert.transformer.layers.1.attn.qkv_proj.bias', 'distilbert.transformer.layers.1.attn.out_proj.weight', 'distilbert.transformer.layers.1.attn.out_proj.bias', 'distilbert.transformer.layers.1.ff.net.fc_in.weight', 'distilbert.transformer.layers.1.ff.net.fc_in.bias', 'distilbert.transformer.layers.1.ff.net.fc_out.weight', 'distilbert.transformer.layers.1.ff.net.fc_out.bias', 'distilbert.transformer.layers.1.norm_attn.weight', 'distilbert.transformer.layers.1.norm_attn.bias', 'distilbert.transformer.layers.1.norm_ffn.weight', 'distilbert.transformer.layers.1.norm_ffn.bias', 'distilbert.transformer.layers.2.attn.qkv_proj.weight', 'distilbert.transformer.layers.2.attn.qkv_proj.bias', 'distilbert.transformer.layers.2.attn.out_proj.weight', 'distilbert.transformer.layers.2.attn.out_proj.bias', 'distilbert.transformer.layers.2.ff.net.fc_in.weight', 'distilbert.transformer.layers.2.ff.net.fc_in.bias', 'distilbert.transformer.layers.2.ff.net.fc_out.weight', 'distilbert.transformer.layers.2.ff.net.fc_out.bias', 'distilbert.transformer.layers.2.norm_attn.weight', 'distilbert.transformer.layers.2.norm_attn.bias', 'distilbert.transformer.layers.2.norm_ffn.weight', 'distilbert.transformer.layers.2.norm_ffn.bias', 'distilbert.transformer.layers.3.attn.qkv_proj.weight', 'distilbert.transformer.layers.3.attn.qkv_proj.bias', 'distilbert.transformer.layers.3.attn.out_proj.weight', 'distilbert.transformer.layers.3.attn.out_proj.bias', 'distilbert.transformer.layers.3.ff.net.fc_in.weight', 'distilbert.transformer.layers.3.ff.net.fc_in.bias', 'distilbert.transformer.layers.3.ff.net.fc_out.weight', 'distilbert.transformer.layers.3.ff.net.fc_out.bias', 'distilbert.transformer.layers.3.norm_attn.weight', 'distilbert.transformer.layers.3.norm_attn.bias', 'distilbert.transformer.layers.3.norm_ffn.weight', 'distilbert.transformer.layers.3.norm_ffn.bias', 'distilbert.transformer.layers.4.attn.qkv_proj.weight', 'distilbert.transformer.layers.4.attn.qkv_proj.bias', 'distilbert.transformer.layers.4.attn.out_proj.weight', 'distilbert.transformer.layers.4.attn.out_proj.bias', 'distilbert.transformer.layers.4.ff.net.fc_in.weight', 'distilbert.transformer.layers.4.ff.net.fc_in.bias', 'distilbert.transformer.layers.4.ff.net.fc_out.weight', 'distilbert.transformer.layers.4.ff.net.fc_out.bias', 'distilbert.transformer.layers.4.norm_attn.weight', 'distilbert.transformer.layers.4.norm_attn.bias', 'distilbert.transformer.layers.4.norm_ffn.weight', 'distilbert.transformer.layers.4.norm_ffn.bias', 'distilbert.transformer.layers.5.attn.qkv_proj.weight', 'distilbert.transformer.layers.5.attn.qkv_proj.bias', 'distilbert.transformer.layers.5.attn.out_proj.weight', 'distilbert.transformer.layers.5.attn.out_proj.bias', 'distilbert.transformer.layers.5.ff.net.fc_in.weight', 'distilbert.transformer.layers.5.ff.net.fc_in.bias', 'distilbert.transformer.layers.5.ff.net.fc_out.weight', 'distilbert.transformer.layers.5.ff.net.fc_out.bias', 'distilbert.transformer.layers.5.norm_attn.weight', 'distilbert.transformer.layers.5.norm_attn.bias', 'distilbert.transformer.layers.5.norm_ffn.weight', 'distilbert.transformer.layers.5.norm_ffn.bias', 'pre_classifier.weight', 'pre_classifier.bias', 'classifier.weight', 'classifier.bias'])


"""