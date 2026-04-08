import os
import torch
from pathlib import Path
from src.model import get_model
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from datasets import load_from_disk
import logging
from easy_logging import EasyFormatter
from src.config import cfg

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger("train")
logger.addHandler(handler)


class PretokenizedCipherDataset(Dataset):
    """Dataset wrapper for loading pre-tokenized cipher samples from disk."""

    def __init__(
        self, directory_path: str | Path, max_samples: int | None = None,
    ) -> None:
        """Load a serialized Hugging Face dataset from `directory_path`."""
        self.hf_dataset = load_from_disk(str(directory_path))

        if max_samples is not None and max_samples < len(self.hf_dataset):
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                logger.info(
                    f"Subsetting dataset from {len(self.hf_dataset):,} to {max_samples:,} samples.",
                )
            self.hf_dataset = self.hf_dataset.select(range(max_samples))

        if len(self.hf_dataset) == 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.warning(f"Dataset at {directory_path} is empty!")

    def __len__(self) -> int:
        """Return the number of examples available in the dataset."""
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Fetch one sample and convert token arrays into `torch.long` tensors."""
        item = self.hf_dataset[idx]

        if (
            len(item["input_ids"]) > cfg.max_context
            or len(item["labels"]) > cfg.max_context
        ):
            logger.info(
                f"Sample {idx} truncated: input_ids {len(item['input_ids'])} -> {cfg.max_context}, labels {len(item['labels'])} -> {cfg.max_context}",
            )

        # Mandatory Training Objective (Equal Loss Weighting)
        input_ids = item["input_ids"][: cfg.max_context]
        labels = item["labels"][: cfg.max_context]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def train() -> None:
    """Start FSDP fine-tuning with Equal Loss Weighting and Optimized Checkpointing."""
    current_output_dir = cfg.final_output_dir
    current_output_dir.mkdir(parents=True, exist_ok=True)

    model = get_model()

    suffix = "Using" if cfg.use_spaces else "Not using"
    logger.info(suffix + " space tokens in training.")

    subset_size = int(os.environ.get("TRAIN_SUBSET_SIZE", 0))
    max_train_samples = subset_size if subset_size > 0 else None

    train_ds = PretokenizedCipherDataset(
        cfg.tokenized_train_dir, max_samples=max_train_samples,
    )
    val_ds = PretokenizedCipherDataset(cfg.tokenized_val_dir)

    args = TrainingArguments(
        output_dir=str(current_output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        gradient_checkpointing=cfg.gradient_checkpointing,
        eval_strategy="steps",
        eval_steps=cfg.save_steps,
        per_device_eval_batch_size=cfg.batch_size,
        eval_accumulation_steps=4,
        logging_steps=cfg.log_steps,
        save_steps=cfg.save_steps,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        tf32=cfg.tf32,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_torch_fused",
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "transformer_layer_cls_to_wrap": "Gemma4TextDecoderLayer",
            "backward_prefetch": "backward_pre",
            "use_orig_params": True,
            "sync_module_states": True,
            "activation_checkpointing": True,
            "limit_all_gathers": True,
        },
    )

    trainer = Trainer(
        model=model,  # type: ignore
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    checkpoint = None
    if os.path.isdir(current_output_dir) and any(current_output_dir.iterdir()):
        checkpoint = True
        logger.info(
            f"Checkpoint detected in {current_output_dir} - Resuming training...",
        )
    else:
        logger.info(f"Initiating Fine-tuning on {torch.cuda.get_device_name(0)}...")

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(f"{current_output_dir}/model")


if __name__ == "__main__":
    train()
