import torch
import argparse
import logging
from datasets import load_from_disk
from transformers import AutoModelForCausalLM
from easy_logging import EasyFormatter
from src.config import cfg

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger("evaluate")
logger.addHandler(handler)


def decode_prediction(ids: list[int], force_exact_len: bool = False) -> str:
    chars = []
    for idx in ids:
        if idx == cfg.space_token_id:
            chars.append("_")
        elif idx >= cfg.char_offset:
            chars.append("?")
        elif idx == cfg.eos_token_id:
            if not force_exact_len:
                break
            chars.append("?")
        else:
            if force_exact_len:
                chars.append("?")
    return "".join(chars)


def evaluate() -> None:
    """Evaluate the SER of the fine-tuned model utilizing the generative cache."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to saved model folder",
    )
    cmd_args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading fine-tuned model from {cmd_args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        cmd_args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    # Re-enable cache for fast autoregressive generation
    model.config.use_cache = True
    model.eval()

    test_arrow_path = cfg.tokenized_test_dir
    logger.info(f"Loading Test data from Arrow shards at {test_arrow_path}...")
    test_ds = load_from_disk(str(test_arrow_path))

    num_samples = min(50, len(test_ds))
    total_ser = 0.0
    evaluated_count = 0

    logger.info(f"Starting generation on {num_samples} samples...")

    for i in range(num_samples):
        item = test_ds[i]
        all_ids = item["input_ids"]

        try:
            sep_idx = all_ids.index(cfg.sep_token_id)
            input_ids = all_ids[: sep_idx + 1]
            true_ids = all_ids[sep_idx + 1 :]
            true_plain = decode_prediction(true_ids)
        except ValueError:
            logger.warning(f"Sample {i} missing SEP token. Skipping.")
            continue

        target_len = len(true_plain)
        input_tensor = torch.tensor([input_ids]).to(device)

        with torch.no_grad():
            output_ids = model.generate(  # type: ignore
                input_tensor,
                min_new_tokens=target_len,
                max_new_tokens=target_len,
                do_sample=False,
                use_cache=True,
                pad_token_id=cfg.pad_token_id,
                bos_token_id=cfg.bos_token_id,
            )

        generated_part = output_ids[0][len(input_ids) :]
        pred_plain = decode_prediction(generated_part.tolist(), force_exact_len=True)

        min_len = min(len(true_plain), len(pred_plain))
        if min_len > 0:
            # Hamming distance calculates the number of mismatching characters at identical positions
            dist = sum(
                c1 != c2
                for c1, c2 in zip(
                    true_plain[:min_len],
                    pred_plain[:min_len],
                    strict=True,
                )
            )

            # Add penalty for any length mismatch just in case (though lengths should now be perfectly equal)
            dist += abs(len(true_plain) - len(pred_plain))

            ser = dist / max(1, len(true_plain))
            total_ser += ser
            evaluated_count += 1

            if i % 10 == 0:
                logger.info(f"Sample {i} | SER: {ser:.4f}")
                logger.info(f"  True: {true_plain[:60]}")
                logger.info(f"  Pred: {pred_plain[:60]}")

    # Divide by evaluated_count to get the true average
    avg_ser = total_ser / max(1, evaluated_count)
    logger.info("=" * 30)
    logger.info(f"FINAL AVERAGE SYMBOL ERROR RATE (SER): {avg_ser:.4f}")


if __name__ == "__main__":
    evaluate()
