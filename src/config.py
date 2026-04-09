import json
import os
from dataclasses import dataclass
import logging
import argparse
from easy_logging import EasyFormatter
from pathlib import Path

TEXT_LEN = 9961
TOTAL_SEQ = TEXT_LEN * 2
BUFFER = 5
UNIQUE_HOMOPHONE_COUNT = 2503
UNIQUE_LETTER_COUNT = 26

DATA_DIR = Path(__file__).parent.parent.parent / "Ciphers"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
HOMOPHONE_FILE = "metadata.json"

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger("config")
logger.addHandler(handler)


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--without-spaces",
    action="store_true",
    default=False,
    help="If enabled the model trains without space tokens in the training dataset",
)
cli_args, _ = parser.parse_known_args()


@dataclass
class Config:
    """Config dataclass optimized for fine-tuning pre-trained Gemma-4."""

    # HF Model Identifier
    model_name_or_path: str = "google/gemma-4-E4B"

    # ARCHITECTURE
    unique_homophones: int = UNIQUE_HOMOPHONE_COUNT
    unique_letters: int = UNIQUE_LETTER_COUNT
    vocab_size: int = UNIQUE_HOMOPHONE_COUNT + UNIQUE_LETTER_COUNT + BUFFER
    max_context: int = TOTAL_SEQ + 1

    @property
    def final_output_dir(self) -> Path:
        """Return the output directory path for saving fine-tuned models, differentiated by space token usage."""
        suffix = "spaces" if self.use_spaces else "normal"
        return self.output_dir / suffix

    # Token IDs
    pad_token_id: int = 0

    @property
    def sep_token_id(self) -> int:
        """Seperator Token."""
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
    batch_size: int = 1
    grad_accum: int = 32
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-5
    epochs: int = 2
    log_steps: int = 5
    save_steps: int = 500
    use_spaces: bool = not cli_args.without_spaces
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    fp16: bool = False
    bf16: bool = True
    tf32: bool = True

    output_dir: Path = OUTPUT_DIR
    data_dir: Path = DATA_DIR

    @property
    def tokenized_train_dir(self) -> Path:
        """Dictionary path for tokenized training data.

        Differentiates between spaced and normal tokenization based on config.
        """
        suffix = "spaced" if self.use_spaces else "normal"
        return self.data_dir / f"tokenized_{suffix}" / "Training"

    @property
    def tokenized_val_dir(self) -> Path:
        """Dictionary path for tokenized validation data.

        Differentiates between spaced and normal tokenization based on config.
        """
        suffix = "spaced" if self.use_spaces else "normal"
        return self.data_dir / f"tokenized_{suffix}" / "Validation"

    @property
    def tokenized_test_dir(self) -> Path:
        """Dictionary path for tokenized test data.

        Differentiates between spaced and normal tokenization based on config.
        """
        suffix = "spaced" if self.use_spaces else "normal"
        return self.data_dir / f"tokenized_{suffix}" / "Test"

    def load_homophones(self) -> None:
        """Load homophone mappings from the metadata file."""
        homophone_path = os.path.join(DATA_DIR, HOMOPHONE_FILE)
        if os.path.exists(homophone_path):
            try:
                with open(homophone_path) as f:
                    meta = json.load(f)
                    self.unique_homophones = int(meta["max_symbol_id"])
            except OSError as e:
                logger.warning("Could not read file: %s", HOMOPHONE_FILE)
                logger.warning("Using default value: %d", self.unique_homophones)
                logger.warning("Error details: %s", str(e))
            except (ValueError, KeyError) as e:
                logger.warning("Invalid or missing data in: %s", HOMOPHONE_FILE)
                logger.warning("Using default value: %d", self.unique_homophones)
                logger.warning("Error details: %s", str(e))

        raw = self.unique_homophones + self.unique_letters + BUFFER
        self.vocab_size = (raw + 63) // 64 * 64


cfg = Config()
cfg.load_homophones()
