import json
import os


def _parse_hf_repo_id(model_ref):
    if not isinstance(model_ref, str):
        return None
    s = model_ref.strip()
    if not s:
        return None
    if "huggingface.co/" in s:
        s = s.split("huggingface.co/", 1)[1].strip("/")
    if s.startswith("https://") or s.startswith("http://"):
        return None
    parts = [p for p in s.split("/") if p]
    if len(parts) < 2:
        return None
    return "/".join(parts[:2])


def _load_config_json(model_ref):
    local_cfg = os.path.join(model_ref, "config.json")
    if os.path.isdir(model_ref) and os.path.exists(local_cfg):
        with open(local_cfg, "r", encoding="utf-8") as f:
            return json.load(f)

    if os.path.isfile(model_ref) and os.path.basename(model_ref) == "config.json":
        with open(model_ref, "r", encoding="utf-8") as f:
            return json.load(f)

    repo_id = _parse_hf_repo_id(model_ref)
    if repo_id is None:
        raise FileNotFoundError(f"Could not resolve config.json for model_ref={model_ref}")

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required to resolve remote model config."
        ) from exc

    cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_custom_config_compat(CustomConfig, model_ref):
    try:
        return CustomConfig.from_pretrained(model_ref)
    except Exception as exc:
        cfg_json = _load_config_json(model_ref)
        rope_scaling = cfg_json.get("rope_scaling")
        rope_type = ""
        if isinstance(rope_scaling, dict):
            rope_type = str(rope_scaling.get("rope_type", "")).lower()

        if rope_type == "llama3":
            print(
                "[compat] Detected llama3 rope_scaling in config. "
                "Applying legacy fallback rope_scaling=None for transformers 4.41.x compatibility."
            )
            cfg_json["rope_scaling"] = None
            return CustomConfig.from_dict(cfg_json)

        raise exc


def detect_model_family(model_ref):
    # Prefer config metadata over string matching in model name.
    try:
        cfg_json = _load_config_json(model_ref)
    except Exception:
        cfg_json = {}

    model_type = str(cfg_json.get("model_type", "")).lower()
    archs = cfg_json.get("architectures", [])
    arch_text = " ".join([str(x).lower() for x in archs])

    if model_type == "llama" or "llamaforcausallm" in arch_text:
        return "llama"
    if model_type == "gemma2" or "gemma2forcausallm" in arch_text:
        return "gemma2"
    if model_type == "qwen3" or "qwen3forcausallm" in arch_text:
        return "qwen3"
    if model_type in ("qwen2", "qwen") or "qwen2forcausallm" in arch_text:
        return "qwen2"

    ref = str(model_ref).lower()
    if "llama" in ref:
        return "llama"
    if "gemma-2" in ref or "gemma2" in ref or "gemma" in ref:
        return "gemma2"
    if "qwen3" in ref:
        return "qwen3"
    if "qwen" in ref:
        return "qwen2"
    return "unknown"
