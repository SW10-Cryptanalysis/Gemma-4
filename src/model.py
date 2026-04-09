import torch
from transformers import AutoModelForCausalLM, AutoConfig
import logging
from easy_logging import EasyFormatter
from src.config import cfg

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger("model")
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def get_model() -> AutoModelForCausalLM:
    """Load pre-trained model and adapt it for sequence-bottleneck conditions."""
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
    )

    model.resize_token_embeddings(cfg.vocab_size)

    # Reinitialize embeddings — the pre-trained values for token IDs 0–2707
    # encode the model's native vocabulary, which is meaningless for homophones.
    init_std = model.config.initializer_range  # typically 0.02 for Gemma-4
    model.model.embed_tokens.weight.data.normal_(mean=0.0, std=init_std)
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight.data.normal_(mean=0.0, std=init_std)

    logger.info(f"Pre-trained {cfg.model_name_or_path} Model loaded successfully!")
    logger.info(f"Parameters:       {model.num_parameters():,}")
    logger.info(f"VRAM for Weights: {(model.get_memory_footprint() / 1e9):.4f} GB")

    return model  # type: ignore
