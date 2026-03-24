import fire
import torch
import transformers
import os
import json

from torch import nn
from tools.assign_device_map import assign_device_map
from tools import global_state
from tools.model_config_compat import load_custom_config_compat, detect_model_family
# from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

torch.set_num_threads(8)
# torch.autograd.set_detect_anomaly(True)
# torch.set_anomaly_enabled(True)


def load_ckpt(ckpt_path):
    tensors = load_file(ckpt_path, device=0)
    # with safe_open(ckpt_path, framework="pt") as f:
    #     for k in f.keys():
    #         tensors[k] = f.get_tensor(k)
    return tensors


def convert(
    ckpt_path,
    save_path="/root/autodl-tmp/converted_models/temp",
    target_hidden_size=1024,
    raw_model_name="/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct",
    target_rms_norm_eps=1e-5,
    tie_word_emb_proj=False,
    tie_n=-1,
    use_aux_loss=True,
    use_std_like_attn=False,
    gpus=1,
    check_data_cls_loss=False,
    use_in_out_mlp=False,
    use_all_attn=False,
):  
    # Resolve family first.
    model_family = detect_model_family(raw_model_name)
    if model_family == "llama":
        from modeling.co_train_llama import CoTrainLM, CustomConfig, reinit_weight
        arch = "LlamaForCausalLM"
        model_type = "llama"
    elif model_family == "gemma2":
        from modeling.co_train_gemma2 import CoTrainLM, CustomConfig, reinit_weight
        arch = "Gemma2ForCausalLM"
        model_type = "gemma2"
    elif model_family == "qwen3":
        from modeling.co_train_qwen3 import CoTrainLM, CustomConfig, reinit_weight
        arch = "Qwen3ForCausalLM"
        model_type = "qwen3"
    elif model_family == "qwen2":
        from modeling.co_train_qwen import CoTrainLM, CustomConfig, reinit_weight
        arch = "Qwen2ForCausalLM"
        model_type = "qwen2"
    else:
        raise ValueError("Could not find corresponding teacher model")

    # Load checkpoint early.
    state_dict = load_ckpt(ckpt_path)
    # Infer/override target hidden size for Gemma2 only.
    if model_family == "gemma2":
        probe_keys = [
            "model.layers.0.mlp.zoom_down.weight",
            "model.layers.0.self_attn.zoom_down.weight",
            "model.zoom.weight",
        ]
        inferred_hidden_size = None
        for k in probe_keys:
            t = state_dict.get(k)
            if t is not None and t.ndim == 2:
                inferred_hidden_size = int(t.shape[0])
                print(f"[convert] Inferred target_hidden_size={inferred_hidden_size} from {k}: {tuple(t.shape)}")
                break

        if inferred_hidden_size is not None and int(target_hidden_size) != inferred_hidden_size:
            print(
                f"[convert] Override target_hidden_size: arg={target_hidden_size} -> ckpt={inferred_hidden_size}"
            )
            target_hidden_size = inferred_hidden_size

    # Load model config and model
    config = load_custom_config_compat(CustomConfig, raw_model_name)
    config.set_custom_kwargs(
        target_hidden_size=target_hidden_size, 
        target_rms_norm_eps=target_rms_norm_eps,
        use_aux_loss=use_aux_loss,
        use_std_like_attn=use_std_like_attn,
        check_data_cls_loss=check_data_cls_loss,
        tie_word_emb_proj=tie_word_emb_proj,
        use_in_out_mlp=use_in_out_mlp,
        use_all_attn=use_all_attn,
    )
    model: CoTrainLM = (
        CoTrainLM.from_pretrained(
            raw_model_name, config=config, torch_dtype=torch.bfloat16,
            device_map=assign_device_map(raw_model_name, gpus=gpus),
        )
    )
    if tie_n > 1:
        model.tie_custom_weights(tie_n)
    model.freeze_original_model()
    # https://github.com/huggingface/transformers/issues/35437
    model.apply(reinit_weight)

    # for n, p in model.named_parameters():
    #     if "lm_head" in n:
    #         print(p.requires_grad, p.shape)
    #         print("Found lm head")
    
    if config.tie_word_embeddings:
        model.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device="cuda:0", dtype=torch.bfloat16)
        model.lm_head.weight.data[:, :] = model.model.embed_tokens.weight.data.clone()
        
    for n, p in model.named_parameters():
        if "lm_head" in n:
            print(p.requires_grad, p.shape)
            print("Found lm head")

    # global_state.info_dict["shared_tokens_emb"] = False
    # if id(model.lm_head.weight.data) == id(model.model.embed_tokens.weight.data):
    #     assert id(model.model.layers[0].mlp.self_attn) == id(model.model.layers[1].mlp.self_attn)
    #     print("Also check the weights")
    # print("lm head", model.lm_head.weight.data[0, :6])
    # print("emb tok", model.model.embed_tokens.weight.data[0, :6])
    #     print("Are emb tokens and lm head sharing parameters? I'm not sure about this assertion... There are weird bugs between 3b and 8b, I'm speechless")
    #     global_state.info_dict["shared_tokens_emb"] = True

    # for n, p in state_dict.items():
    #     print(n, p.shape)

    # Compatibility pass is Gemma2-only to avoid changing Llama/Qwen behavior.
    if model_family == "gemma2":
        model_state = model.state_dict()
        transposed_keys = []
        dropped_unknown = []
        dropped_incompatible = []
        for k, v in list(state_dict.items()):
            tgt = model_state.get(k)
            if tgt is None:
                dropped_unknown.append(k)
                del state_dict[k]
                continue
            if tuple(v.shape) == tuple(tgt.shape):
                continue

            if v.ndim == 2 and tuple(v.shape[::-1]) == tuple(tgt.shape):
                state_dict[k] = v.t().contiguous()
                transposed_keys.append((k, tuple(v.shape), tuple(state_dict[k].shape)))
                continue

            dropped_incompatible.append((k, tuple(v.shape), tuple(tgt.shape)))
            del state_dict[k]

        if transposed_keys:
            print(f"[convert][gemma2] Auto-transposed {len(transposed_keys)} mismatched 2D tensors")
            for k, src_shape, dst_shape in transposed_keys[:5]:
                print(f"  - {k}: {src_shape} -> {dst_shape}")
            if len(transposed_keys) > 5:
                print(f"  ... and {len(transposed_keys) - 5} more")

        if dropped_unknown:
            print(f"[convert][gemma2] Dropped {len(dropped_unknown)} unknown tensors")
            for k in dropped_unknown[:5]:
                print(f"  - {k}")
            if len(dropped_unknown) > 5:
                print(f"  ... and {len(dropped_unknown) - 5} more")

        if dropped_incompatible:
            print(f"[convert][gemma2] Dropped {len(dropped_incompatible)} incompatible tensors")
            for k, src_shape, tgt_shape in dropped_incompatible[:5]:
                print(f"  - {k}: ckpt={src_shape}, model={tgt_shape}")
            if len(dropped_incompatible) > 5:
                print(f"  ... and {len(dropped_incompatible) - 5} more")

    # Keep global behavior unchanged for non-Gemma families.
    if model_family == "gemma2":
        # Guard: remove keys not present in current model to avoid unexpected_keys failures.
        model_state_keys = set(model.state_dict().keys())
        dropped_unknown_global = [k for k in list(state_dict.keys()) if k not in model_state_keys]
        if dropped_unknown_global:
            for k in dropped_unknown_global:
                del state_dict[k]
            print(f"[convert] Dropped {len(dropped_unknown_global)} unknown keys before load_state_dict")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        print(f"[convert] Unexpected keys after filtering: {len(unexpected_keys)}")
        for k in unexpected_keys[:10]:
            print(f"  - {k}")
        raise ValueError("Unexpected keys remain after compatibility filtering")

    def merge(module: nn.Module):
        if hasattr(module, "merge_weight"):
            module.merge_weight()
    
    # print("model.zoom before merge", model.model.zoom.weight.data[0, :6])
    # print("zoom_down before merge", model.zoom_down.weight.data[0, :6])
    print("lm head before merge", model.lm_head.weight.data[0, :6])
    print("emb tok before merge", model.model.embed_tokens.weight.data[0, :6])

    with torch.no_grad():
        model.apply(merge)

    print("lm head after merge", model.lm_head.weight.data[0, :6])
    print("emb tok after merge", model.model.embed_tokens.weight.data[0, :6])
    # for n, p in model.named_parameters():
    #     if "mlp" in n:
    #         print(n, p[0, :6])

    # reverse require grad
    model.freeze_original_model()
    for n, p in model.named_parameters():
        p.requires_grad = not p.requires_grad
    
    model.save_pretrained(save_directory=save_path, only_save_trainable=True)
    # Next, use a new ckpt to overwrite it
    # Now I'm already confused as to why this is necessary here
    #       It seems that at the time it was because save_pretrained does not save another word emb layer if sharing, so use the following instruction to re-save, so if tied, it should correspond to the same tensor
    if config.tie_word_embeddings:
        state_dict = model.get_trained_params()
        assert "lm_head.weight" in state_dict
        if tie_word_emb_proj:
            # print(state_dict.keys())
            assert (state_dict["lm_head.weight"][:6] - state_dict["model.embed_tokens.weight"][:6]).pow(2).sum() < 1e-5
            del state_dict["lm_head.weight"]
            # state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
        save_file(state_dict, filename=os.path.join(save_path, "model.safetensors"))

    config_json_path = os.path.join(save_path, "config.json")
    config_json = json.load(open(config_json_path, "r"))
    
    config_json["head_dim"] = config_json["hidden_size"] // config_json["num_attention_heads"]
    config_json["hidden_size"] = config_json["target_hidden_size"]
    config_json["architectures"][0] = arch
    config_json["model_type"] = model_type
    config_json["tie_word_embeddings"] = True if tie_word_emb_proj else False
    
    with open(config_json_path, "w", encoding="utf-8") as _out:
        json.dump(config_json, _out)

    import shutil
    try:
        shutil.copy(os.path.join(raw_model_name, "special_tokens_map.json"), save_path)
    except:
        pass
    shutil.copy(os.path.join(raw_model_name, "tokenizer_config.json"), save_path)
    shutil.copy(os.path.join(raw_model_name, "tokenizer.json"), save_path)
    try:
        shutil.copy(os.path.join(raw_model_name, "vocab.json"), save_path)
    except:
        pass
    # tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    fire.Fire(convert)
