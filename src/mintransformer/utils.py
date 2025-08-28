import torch
import torch.nn as nn
from collections import OrderedDict
from src.mintransformer.blocks import TransformerBlock


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
            raise AssertionError("Values for {valid_key} did not match while state dict loading")

    return key_map
