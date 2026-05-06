import json
import os
import sys
from dataclasses import dataclass
import logging
import argparse
from easy_logging import EasyFormatter
from pathlib import Path

MAX_PLAIN_SPACES = 13077
MAX_PLAIN_NORMAL = 10063

DATA_DIR = Path(__file__).parent.parent.parent / "Ciphers"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
HOMOPHONE_FILE = "metadata.json"

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger("config")
logger.addHandler(handler)

VALID_FAMILIES = ["gemma", "llama", "mistral", "jamba", "mamba"]

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--without-spaces",
    action="store_true",
    default=False,
    help="If enabled the model trains without space tokens in the training dataset",
)
parser.add_argument(
    "--model-family",
    type=str,
    default=None,
    choices=VALID_FAMILIES,
    help=f"Model architecture family. Required. Choose from: {', '.join(VALID_FAMILIES)}",
)
parser.add_argument(
    "--model-path",
    type=str,
    default=None,
    help="HuggingFace model identifier or local path to the pre-trained model. Required.",
)
cli_args, _ = parser.parse_known_args()

# Fail early and clearly if required flags are missing
_errors = []
if cli_args.model_family is None:
    _errors.append(
        "  --model-family is required. Choose from: " + ", ".join(VALID_FAMILIES),
    )
if cli_args.model_path is None:
    _errors.append(
        "  --model-path is required. Provide a HuggingFace model ID or local path.",
    )
if _errors:
    logger.error("Missing required arguments:\n" + "\n".join(_errors))
    logger.error(
        "Example: sbatch train.slurm --model-family gemma --model-path google/gemma-4-E4B",
    )
    sys.exit(1)


@dataclass
class Config:
    """Config dataclass for fine-tuning pre-trained HuggingFace causal LMs."""

    FSDP_LAYER_MAP = {
        "gemma": "Gemma4TextDecoderLayer",
        "llama": "LlamaDecoderLayer",
        "mistral": "MistralDecoderLayer",
        "jamba": "JambaDecoderLayer",
        "mamba": None,  # Mamba is not transformer-based; FSDP wrapping differs
    }

    FLASH_ATTN_COMPATIBLE = {"gemma", "llama", "mistral", "jamba"}

    # HF Model Identifier and family — set from CLI flags
    model_name_or_path: str = cli_args.model_path
    model_family: str = cli_args.model_family

    buffer: int = 10
    unique_letters: int = 26
    unique_homophones: int = 0

    @property
    def vocab_size(self) -> int:
        """Dynamically calculate vocab size padded to the nearest multiple of 64."""
        raw = self.unique_homophones + self.unique_letters + self.buffer
        return (raw + 63) // 64 * 64

    @property
    def max_context(self) -> int:
        """Calculate dynamic variables after the dataclass is initialized."""
        if self.use_spaces:
            return (MAX_PLAIN_SPACES * 2) + self.buffer
        return (MAX_PLAIN_NORMAL * 2) + self.buffer

    @property
    def final_output_dir(self) -> Path:
        """Return the output directory path for saving fine-tuned models, differentiated by space token usage."""
        suffix = "spaces" if self.use_spaces else "normal"
        return self.output_dir / self.model_family / suffix

    # Token IDs
    pad_token_id: int = 0

    @property
    def sep_token_id(self) -> int:
        """Separator Token."""
        return self.unique_homophones + 1

    @property
    def space_token_id(self) -> int:
        """Space Token."""
        return self.sep_token_id + 1

    @property
    def bos_token_id(self) -> int:
        """Beginning of Sequence Token."""
        return self.space_token_id + 1

    @property
    def eos_token_id(self) -> int:
        """End of Sequence Token."""
        return self.bos_token_id + 1

    @property
    def char_offset(self) -> int:
        """Offset for character token IDs."""
        return self.eos_token_id + 1

    # FINE-TUNING TRAINING PARAMS
    batch_size: int = 16
    grad_accum: int = 1
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-5
    epochs: int = 2
    log_steps: int = 5
    save_steps: int = 500
    max_train_samples: int = 500000
    use_spaces: bool = not cli_args.without_spaces
    weight_decay: float = 0.01
    warmup_steps: int = 1500
    fp16: bool = False
    bf16: bool = True
    tf32: bool = True

    output_dir: Path = OUTPUT_DIR
    data_dir: Path = DATA_DIR

    @property
    def tokenized_train_dir(self) -> Path:
        """Path for tokenized training data."""
        suffix = "spaced" if self.use_spaces else "normal"
        return self.data_dir / f"tokenized_{suffix}_truncated_4000" / "Training"

    @property
    def tokenized_val_dir(self) -> Path:
        """Path for tokenized validation data."""
        suffix = "spaced" if self.use_spaces else "normal"
        return self.data_dir / f"tokenized_{suffix}_truncated_4000" / "Validation"

    def load_homophones(self) -> None:
        """Load homophone mappings from the metadata file."""
        homophone_path = os.path.join(DATA_DIR, HOMOPHONE_FILE)
        if not os.path.exists(homophone_path):
            raise FileNotFoundError(
                f"Metadata file not found at: {homophone_path}. "
                "Cannot determine unique_homophones — aborting.",
                1,
            )
        try:
            with open(homophone_path) as f:
                meta = json.load(f)
                self.unique_homophones = int(meta["max_symbol_id"])
        except OSError as e:
            raise OSError(f"Could not read file: {homophone_path}") from e
        except (ValueError, KeyError) as e:
            raise ValueError(
                f"Invalid or missing 'max_symbol_id' in {homophone_path}",
            ) from e
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            logger.info(
                f"Config initialized: unique_homophones={self.unique_homophones}, sep_token_id={self.sep_token_id}, space_token_id={self.space_token_id}, bos_token_id={self.bos_token_id}, eos_token_id={self.eos_token_id}, char_offset={self.char_offset}, vocab_size={self.vocab_size}",
            )
            logger.info(
                f"Max len set to {self.max_context} based on use_spaces={self.use_spaces}",
            )
            logger.info(f"Training dir set to: {self.tokenized_train_dir}")
            logger.info(f"Validation dir set to: {self.tokenized_val_dir}")


cfg = Config()
