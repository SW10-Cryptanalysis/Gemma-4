import torch
import os
import logging
from transformers import AutoModelForCausalLM, AutoConfig
from easy_logging import EasyFormatter
from src.config import cfg

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger("model")
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _get_text_model(model: AutoModelForCausalLM) -> torch.nn.Module:
    """Return the text decoder sub-module, handling multimodal wrappers.

    Gemma 4 (and similar multimodal models) nest the language model under
    model.model.language_model, whereas text-only models (Llama, Mistral,
    Gemma 1/2/3) expose embed_tokens directly on model.model.
    """
    inner = model.model  # type: ignore
    if hasattr(inner, "language_model"):
        return inner.language_model
    return inner


def get_model() -> AutoModelForCausalLM:
    """Load pre-trained model and adapt it for sequence-bottleneck conditions."""
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        logger.info(f"Loading pre-trained model: {cfg.model_name_or_path}")

    model_config = AutoConfig.from_pretrained(
        cfg.model_name_or_path,
        attn_implementation=(
            "flash_attention_2"
            if cfg.model_family in cfg.FLASH_ATTN_COMPATIBLE
            else "eager"
        ),
        use_cache=False,
    )

    model_config.pad_token_id = cfg.pad_token_id
    model_config.bos_token_id = cfg.bos_token_id
    model_config.eos_token_id = cfg.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    model.resize_token_embeddings(cfg.vocab_size)

    # Reinitialize embeddings — the pre-trained values for token IDs 0–2707
    # encode the model's native vocabulary, which is meaningless for homophones.
    init_std = model.config.initializer_range  # typically 0.02 for Gemma-4
    text_model = _get_text_model(model)

    text_model.embed_tokens.weight.data.normal_(mean=0.0, std=init_std)

    # Gemma 4 uses Per-Layer Embeddings (PLE): an additional embed_tokens_per_layer
    # weight that also encodes token identity and must be reinitialized.
    if hasattr(text_model, "embed_tokens_per_layer"):
        text_model.embed_tokens_per_layer.weight.data.normal_(mean=0.0, std=init_std)
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            logger.info("Reinitialized Per-Layer Embeddings (embed_tokens_per_layer).")

    if model.lm_head.weight.data_ptr() != text_model.embed_tokens.weight.data_ptr():
        model.lm_head.weight.data.normal_(mean=0.0, std=init_std)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        logger.info(f"Pre-trained {cfg.model_name_or_path} Model loaded successfully!")
        logger.info(f"Parameters:       {model.num_parameters():,}")
        logger.info(f"VRAM for Weights: {(model.get_memory_footprint() / 1e9):.4f} GB")

    return model  # type: ignore
