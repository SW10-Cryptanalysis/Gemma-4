import sys
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom CLI flags so pytest doesn't reject them."""
    parser.addoption(
        "--model-family",
        action="store",
        default=None,
        help="Model architecture family (e.g. gemma, llama, mistral)",
    )
    parser.addoption(
        "--model-path",
        action="store",
        default=None,
        help="HuggingFace model ID or local path to the pre-trained model",
    )
    parser.addoption(
        "--dataset-path",
        action="store",
        default=None,
        help="Path to the dataset (e.g., 'tokenized_spaced', 'tokenized_normal_truncated_4000')",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Inject custom flags into sys.argv so argparse in config.py can find them.

    pytest processes addopts internally and never adds them to sys.argv, so
    config.py's parse_known_args() would see nothing without this step.
    This hook runs before test collection, ensuring config.py imports cleanly.
    """
    for flag, dest in [
        ("--model-family", "model_family"),
        ("--model-path", "model_path"),
        ("--dataset-path", "dataset_path"),
    ]:
        try:
            value = config.getoption(dest)
        except ValueError:
            value = None
        if value and flag not in sys.argv:
            sys.argv.extend([flag, value])
