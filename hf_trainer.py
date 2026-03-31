import random

import fire
import wandb
import torch
import os

orig_torch_load = torch.load


def _allow_unsafe_torch_load():
    flag = str(os.environ.get("LRC_ALLOW_UNSAFE_TORCH_LOAD", "0")).strip().lower()
    return flag in {"1", "true", "yes", "y"}


def torch_wrapper(*args, **kwargs):
    allow_unsafe = _allow_unsafe_torch_load()
    requested = kwargs.get("weights_only")

    if requested is None:
        kwargs["weights_only"] = not allow_unsafe
    elif requested is False and not allow_unsafe:
        print(
            "[security] Blocked torch.load(weights_only=False). "
            "Set LRC_ALLOW_UNSAFE_TORCH_LOAD=1 if you trust the checkpoint source."
        )
        kwargs["weights_only"] = True

    try:
        return orig_torch_load(*args, **kwargs)
    except TypeError as exc:
        if "weights_only" in str(exc):
            kwargs.pop("weights_only", None)
            return orig_torch_load(*args, **kwargs)
        raise

torch.load = torch_wrapper

NODE_CLASS_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS']

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, LlamaForCausalLM, Qwen2ForCausalLM, set_seed
from datasets import load_from_disk, IterableDatasetDict
from torch import nn
from data.get_any_data import get_any_dataset
from data.get_any_tokenize_func import get_any_tokenize_func, get_any_data_collator
from tools.global_state import hyper_params
from tools import global_state
from tools.assign_device_map import assign_device_map
from tools.model_config_compat import load_custom_config_compat, detect_model_family
from accelerate import Accelerator
from datetime import datetime

torch.set_num_threads(8)
print("init accelerate")
accelerator = Accelerator()
print("init accelerate done.")
# torch.autograd.set_detect_anomaly(True)
# torch.set_anomaly_enabled(True)


def get_current_time_short():
    now = datetime.now()
    time_str = now.strftime('%m%d%H')
    return time_str


def train_model(
    target_hidden_size=1024,
    raw_model_name="../models/Meta-Llama-3-8B-Instruct",
    model_cls = "distill",
    dataset_name="../datasets/squad_v2",
    output_dir="../ckpts",
    # Training parameter settings
    num_epochs=1,
    batch_size=4,
    learning_rate=1e-4,
    warmup_ratio=0.005,
    target_rms_norm_eps=1e-5,
    gradient_accumulation_steps=1,
    log_steps=100,
    save_steps=20000,
    max_steps=-1,
    data_max_len=1024,
    project_name="expt-small-llm",
    tie_n=-1,
    tie_word_emb_proj=False,
    max_grad_norm=1.0,
    # Hyperparameter settings
    aux_loss_scale_factor=1.0,
    use_aux_loss=True,
    use_logits_loss=True,
    use_std_like_attn=False,
    student_attn_from_scratch=False,
    del_layers="",
    ban_layers="",
    use_in_out_mlp=False,
    use_all_attn=False,
    use_additional_align=False,
    # Others
    gpus=1,
    resume_checkpoint=None,
    load_model_weight_path=None,  # for sft
    check_data_cls_loss=False,
    extra_tags="ordinary",
    kl_temperature=10.0,
    lr_scheduler="linear",
    aux_loss_type="mseloss",
    use_ntp_loss=True,
    str_ban_losses="no",
    # fsdp="",
    use_accelerate=False,
    adam_beta2=0.999,
):  
    def _resolve_teacher_cls(family):
        if family == "llama":
            return LlamaForCausalLM
        if family == "qwen2":
            return Qwen2ForCausalLM
        if family == "qwen3":
            return AutoModelForCausalLM
        if family == "gemma2":
            return AutoModelForCausalLM
        raise ValueError(f"Unsupported model family for teacher loading: {family}")

    hyper_params["gradient_accumulation_steps"] = gradient_accumulation_steps
    hyper_params["aux_loss_scale_factor"] = aux_loss_scale_factor
    # Load corresponding model cls
    model_family = detect_model_family(raw_model_name)
    if model_family == "llama":
        from modeling.co_train_llama import CoTrainLM, CustomConfig, reinit_weight
    elif model_family == "gemma2":
        from modeling.co_train_gemma2 import CoTrainLM, CustomConfig, reinit_weight
    elif model_family == "qwen3":
        from modeling.co_train_qwen3 import CoTrainLM, CustomConfig, reinit_weight
    elif model_family == "qwen2":
        from modeling.co_train_qwen import CoTrainLM, CustomConfig, reinit_weight
    else:
        raise ValueError("Could not find corresponding teacher model")
    # Process ban losses
    if isinstance(str_ban_losses, str):
        global_state.ban_losses += str_ban_losses.split(',')
    else:
        global_state.ban_losses += str_ban_losses
    # layers related processing, but actually fire in the new version can automatically handle? (or in the old version can already handle?)
    if isinstance(del_layers, str):
        del_layers = [int(x) for x in del_layers.split(',') if len(x) > 0] 
    if isinstance(ban_layers, str):
        ban_layers = [int(x) for x in ban_layers.split(',') if len(x) > 0]
    global_state.ban_layers += ban_layers

    print(f"(for debug) use-aux-loss", use_aux_loss)
    print(f"(for debug) tie word emb porj", tie_word_emb_proj)
    print(f"(for debug) use logits/kl loss", use_logits_loss)
    set_seed(429)
    # Load tokenizer
    
    if "tokenize" not in dataset_name:
        tokenizer = AutoTokenizer.from_pretrained(raw_model_name, use_fast=True) 
        # tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = None

    # Load and process dataset
    dataset = get_any_dataset(dataset_name, tokenizer)
    # dataset = dataset.shuffle(seed=429)
    if max_steps > 0:
        if type(dataset) is not IterableDatasetDict and \
              max_steps * batch_size * gradient_accumulation_steps < len(dataset["train"]):
            dataset["train"] = dataset["train"].select(
                list(range(max_steps * batch_size * gradient_accumulation_steps))
            )

    tokenize_func = get_any_tokenize_func(dataset_name, tokenizer, data_max_len)
    tokenized_datasets = dataset.map(tokenize_func, batched=False)
    if "tokenize" not in dataset_name:
        # Remove unnecessary columns
        tokenized_datasets = tokenized_datasets.remove_columns(
            dataset["train"].column_names
        )

    # Load model config and model
    config = load_custom_config_compat(CustomConfig, raw_model_name)
    config.set_custom_kwargs(
        target_hidden_size=target_hidden_size, 
        target_rms_norm_eps=target_rms_norm_eps,
        use_aux_loss=use_aux_loss,
        use_std_like_attn=use_std_like_attn,
        check_data_cls_loss=check_data_cls_loss,
        kl_temperature=kl_temperature,
        aux_loss_type=aux_loss_type,
        use_logits_loss=use_logits_loss,
        use_ntp_loss=use_ntp_loss,
        student_attn_from_scratch=student_attn_from_scratch,
        tie_word_emb_proj=tie_word_emb_proj,
        del_layers=del_layers,
        use_in_out_mlp=use_in_out_mlp,
        use_all_attn=use_all_attn,
        use_additional_align=use_additional_align,
    )

    # hf's from_pretrained seems to affect the normal initialization of non-pretrained parameters, so use another way of initialization
    local_rank = 0
    world_size = 1
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print("(for debug)", "Local Rank is", local_rank)
    if accelerator.is_main_process:
        print("Main Process Local Rank is", local_rank)

    if model_cls == "distill":
        model: CoTrainLM = (
            CoTrainLM.from_pretrained(
                raw_model_name, config=config, torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2" if not use_std_like_attn else "manual",
                device_map=None if use_accelerate else assign_device_map(raw_model_name, gpus=gpus, local_rank=local_rank),
            )
        )
        if tie_n > 1:
            model.tie_custom_weights(tie_n)
        model.freeze_original_model()

        # https://github.com/huggingface/transformers/issues/35437
        model.apply(reinit_weight)

        if load_model_weight_path is not None:
            # Load checkpoint, but do not load scheduler and optimizer
            from safetensors.torch import load_file
            sd = load_file(load_model_weight_path)
            missed, unexpected = model.load_state_dict(sd, strict=False)
            assert len(unexpected) == 0
            print("loaded model weights.")

    elif model_cls == "origin":
        config._attn_implementation = "flash_attention_2"
        _cls = _resolve_teacher_cls(model_family)
        model = _cls.from_pretrained(
            raw_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        model = model.to(dtype=torch.bfloat16, device="cuda:0")  # TODO: adapt to multi-GPU training
        for n, p in model.named_parameters():
            assert p.dtype == torch.bfloat16
    elif model_cls == "only_kd":
        from modeling.only_kd_llama import KDLlamaForCausalLM

        config._attn_implementation = "flash_attention_2"
        config.hidden_size = config.target_hidden_size
        config.num_attention_heads //= 2
        config.num_key_value_heads //= 2  # try to approximate as much as possible
        _cls = _resolve_teacher_cls(model_family)
        teacher = _cls.from_pretrained(raw_model_name, torch_dtype=torch.bfloat16,
                                                   attn_implementation="flash_attention_2")
        teacher = accelerator.prepare_model(teacher)
        model = KDLlamaForCausalLM(config)
        model = model.to(dtype=torch.bfloat16, device="cuda:0")
        model.set_teacher(teacher)
        model = accelerator.prepare_model(model)
        
        for n, p in model.named_parameters():
            assert p.dtype == torch.bfloat16
    elif model_cls == "tiny_bert":
        from modeling.tiny_bert_llama import TinyBertLlamaForCausalLM
        
        _cls = _resolve_teacher_cls(model_family)
        teacher = _cls.from_pretrained(
            raw_model_name, torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2"
        )
        
        # config._attn_implementation = "flash_attention_2"
        config.hidden_size = config.target_hidden_size
        teacher = accelerator.prepare_model(teacher)
        model = TinyBertLlamaForCausalLM(config)
        model = model.to(dtype=torch.bfloat16, device="cuda:0")
        model.set_teacher(teacher)
        model = accelerator.prepare_model(model)
        
        for n, p in model.named_parameters():
            assert p.dtype == torch.bfloat16, f"{n}'s type is {p.dtype}"
    else:
        raise ValueError

    assert isinstance(extra_tags, str) or isinstance(extra_tags, tuple)
    data_real_name = os.path.split(dataset_name)[-1]
    if data_real_name.endswith("jsonl"):
        data_real_name = os.path.split(data_real_name)[-1]
    tags = [
        data_real_name, 
        # f"std_attn={int(use_std_like_attn)}",
        # f"aux={int(use_aux_loss)}",
        "v4",
    ] + (extra_tags.split(",") if isinstance(extra_tags, str) else list(extra_tags))
    print("[for debug] tags:", tags)
    output_dir = os.path.join(*[output_dir, "-".join(tags), get_current_time_short()])
    os.makedirs(output_dir, exist_ok=True)

    lr_scheduler_kwargs = {}
    if lr_scheduler == "warmup_stable_decay":
        lr_scheduler_kwargs = {
            "num_warmup_steps": int(max_steps * warmup_ratio) + 1,
            "num_stable_steps": int(max_steps * (0.9 - warmup_ratio)) + 1,
        }
        lr_scheduler_kwargs["num_decay_steps"] = max_steps - lr_scheduler_kwargs["num_warmup_steps"] - lr_scheduler_kwargs["num_stable_steps"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=log_steps,
        save_steps=save_steps,
        max_steps=max_steps,
        save_total_limit=10,
        bf16=True,  # use bfloat16 precision
        gradient_checkpointing=True,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        max_grad_norm=max_grad_norm,
        logging_dir="./logs",
        report_to="wandb" if local_rank == 0 else "none",
        lr_scheduler_type=lr_scheduler,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        adam_beta2=adam_beta2,
    )

    _upload_cfg = training_args.to_dict()
    _upload_cfg.update(config.to_dict())
    if local_rank == 0:
        wandb.init(
            project=project_name, 
            name=f"{extra_tags}-ths{target_hidden_size}lr{learning_rate}"
                f"kl_t{kl_temperature}",
            config=_upload_cfg,
            tags=tags,
        )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=get_any_data_collator(dataset_name, tokenizer, data_max_len)
    )

    # Start training

    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # saving final model and tokenizer
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()
    if tokenizer:
        tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train_model)