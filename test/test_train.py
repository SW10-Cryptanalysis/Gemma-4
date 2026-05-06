import pytest
import torch
import numpy as np

from src.train import PretokenizedCipherDataset, train
from src.config import Config, cfg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hf_dataset(mocker, n: int = 10, seq_len: int = 100):
    """Return a minimal fake HuggingFace Dataset with *n* identical samples."""
    sample = {
        "input_ids": list(range(seq_len)),
        "labels": [-100] * (seq_len // 2) + list(range(seq_len // 2)),
    }
    ds = mocker.Mock()
    ds.__len__ = mocker.Mock(return_value=n)
    ds.__getitem__ = mocker.Mock(side_effect=lambda i: sample)
    ds.select = mocker.Mock(return_value=ds)  # select returns itself by default
    return ds


# ---------------------------------------------------------------------------
# PretokenizedCipherDataset
# ---------------------------------------------------------------------------


class TestPretokenizedCipherDataset:
    def test_len_matches_underlying_dataset(self, mocker):
        ds_mock = _make_hf_dataset(mocker, n=10)
        mocker.patch("src.train.load_from_disk", return_value=ds_mock)

        ds = PretokenizedCipherDataset("fake/path")

        assert len(ds) == 10

    def test_getitem_returns_input_ids_and_labels(self, mocker):
        ds_mock = _make_hf_dataset(mocker, n=5)
        mocker.patch("src.train.load_from_disk", return_value=ds_mock)

        ds = PretokenizedCipherDataset("fake/path")
        item = ds[0]

        assert "input_ids" in item
        assert "labels" in item

    def test_getitem_returns_long_tensors(self, mocker):
        ds_mock = _make_hf_dataset(mocker, n=5)
        mocker.patch("src.train.load_from_disk", return_value=ds_mock)

        ds = PretokenizedCipherDataset("fake/path")
        item = ds[0]

        assert item["input_ids"].dtype == torch.long
        assert item["labels"].dtype == torch.long

    def test_getitem_returns_1d_tensors(self, mocker):
        ds_mock = _make_hf_dataset(mocker, n=5)
        mocker.patch("src.train.load_from_disk", return_value=ds_mock)

        ds = PretokenizedCipherDataset("fake/path")
        item = ds[0]

        assert item["input_ids"].ndim == 1
        assert item["labels"].ndim == 1

    def test_long_input_truncated_to_max_context(self, mocker):
        """Sequences longer than cfg.max_context must be truncated silently."""
        sample = {
            "input_ids": list(range(cfg.max_context + 500)),
            "labels": list(range(cfg.max_context + 500)),
        }
        ds_mock = mocker.Mock()
        ds_mock.__len__ = mocker.Mock(return_value=1)
        ds_mock.__getitem__ = mocker.Mock(return_value=sample)
        mocker.patch("src.train.load_from_disk", return_value=ds_mock)

        ds = PretokenizedCipherDataset("fake/path")
        item = ds[0]

        assert item["input_ids"].shape[0] == cfg.max_context
        assert item["labels"].shape[0] == cfg.max_context

    def test_short_input_not_padded(self, mocker):
        """Sequences shorter than max_context must not be zero-padded by the dataset."""
        short_len = 50
        sample = {
            "input_ids": list(range(short_len)),
            "labels": list(range(short_len)),
        }
        ds_mock = mocker.Mock()
        ds_mock.__len__ = mocker.Mock(return_value=1)
        ds_mock.__getitem__ = mocker.Mock(return_value=sample)
        mocker.patch("src.train.load_from_disk", return_value=ds_mock)

        ds = PretokenizedCipherDataset("fake/path")
        item = ds[0]

        assert item["input_ids"].shape[0] == short_len

    def test_max_samples_subsets_dataset(self, mocker):
        """When max_samples is given, dataset.select must be called with that range."""
        ds_mock = _make_hf_dataset(mocker, n=100)
        mocker.patch("src.train.load_from_disk", return_value=ds_mock)

        PretokenizedCipherDataset("fake/path", max_samples=20)

        ds_mock.select.assert_called_once_with(range(20))

    def test_none_max_samples_skips_select(self, mocker):
        """With max_samples=None the full dataset is used; select must not be called."""
        ds_mock = _make_hf_dataset(mocker, n=100)
        mocker.patch("src.train.load_from_disk", return_value=ds_mock)

        PretokenizedCipherDataset("fake/path", max_samples=None)

        ds_mock.select.assert_not_called()

    def test_max_samples_larger_than_dataset_skips_select(self, mocker):
        """Requesting more samples than available must not call select."""
        ds_mock = _make_hf_dataset(mocker, n=10)
        mocker.patch("src.train.load_from_disk", return_value=ds_mock)

        PretokenizedCipherDataset("fake/path", max_samples=999)

        ds_mock.select.assert_not_called()


# ---------------------------------------------------------------------------
# TrainingArguments — shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def captured_training_args(mocker, tmp_path):
    """Run train() with all heavy dependencies mocked.

    Returns the mock of TrainingArguments so tests can inspect how it was called.
    The output directory is redirected to tmp_path so no real disk writes occur.
    """
    # Redirect cfg.final_output_dir to tmp_path (it is a @property).
    mocker.patch.object(
        Config,
        "final_output_dir",
        new_callable=mocker.PropertyMock,
        return_value=tmp_path,
    )

    # Suppress model loading.
    mock_model = mocker.Mock()
    mocker.patch("src.train.get_model", return_value=mock_model)

    # Suppress dataset loading.
    fake_ds = _make_hf_dataset(mocker, n=4, seq_len=10)
    mocker.patch("src.train.PretokenizedCipherDataset", return_value=fake_ds)

    # Capture TrainingArguments call without constructing a real one.
    mock_args_cls = mocker.patch(
        "src.train.TrainingArguments", return_value=mocker.Mock()
    )

    # Prevent real Trainer / CUDA usage.
    mock_trainer = mocker.Mock()
    mocker.patch("src.train.Trainer", return_value=mock_trainer)
    mocker.patch("torch.cuda.get_device_name", return_value="Mock GPU")
    mocker.patch.object(cfg, "load_homophones")

    # Force the "fresh start" branch (no checkpoint) so CUDA name is logged.
    mocker.patch("os.path.isdir", return_value=False)

    train()
    return mock_args_cls


# ---------------------------------------------------------------------------
# Warmup regression: warmup_steps=0.05 was cast to int(0)
# ---------------------------------------------------------------------------


class TestWarmupRatioRegression:
    """Guard against the warmup_steps vs warmup_ratio bug in TrainingArguments."""

    def test_warmup_steps_passed_to_training_arguments(self, captured_training_args):
        _, kwargs = captured_training_args.call_args
        assert "warmup_steps" in kwargs, (
            "TrainingArguments must receive warmup_steps=, not warmup_ratio=. "
            "warmup_steps expects an int; 1500."
        )

    def test_warmup_steps_value_matches_config(self, captured_training_args):
        _, kwargs = captured_training_args.call_args
        assert kwargs["warmup_steps"] == cfg.warmup_steps


# ---------------------------------------------------------------------------
# FSDP configuration
# ---------------------------------------------------------------------------


class TestFSDPConfig:
    def test_fsdp_layer_class_is_gemma4(self, captured_training_args):
        """FSDP must wrap Gemma4TextDecoderLayer, not LlamaDecoderLayer.

        Wrapping the wrong layer class causes FSDP to fall back to wrapping
        the entire model as a single unit, breaking parameter sharding and
        likely causing an OOM on L4 GPUs.
        """
        _, kwargs = captured_training_args.call_args
        layer_cls = kwargs.get("fsdp_config", {}).get(
            "transformer_layer_cls_to_wrap", ""
        )

        assert layer_cls == "Gemma4TextDecoderLayer", (
            "fsdp_config['transformer_layer_cls_to_wrap'] must be "
            f"'Gemma4TextDecoderLayer'. Got: {layer_cls!r}"
        )

    def test_fsdp_config_does_not_reference_llama(self, captured_training_args):
        """Any leftover reference to LlamaDecoderLayer must be absent."""
        _, kwargs = captured_training_args.call_args
        layer_cls = kwargs.get("fsdp_config", {}).get(
            "transformer_layer_cls_to_wrap", ""
        )

        assert (
            "Llama" not in layer_cls
        ), f"FSDP config still references a Llama layer class: {layer_cls!r}"

    def test_fsdp_activation_checkpointing_enabled(self, captured_training_args):
        _, kwargs = captured_training_args.call_args
        fsdp_cfg = kwargs.get("fsdp_config", {})
        assert fsdp_cfg.get("activation_checkpointing") is True

    def test_fsdp_full_shard_present(self, captured_training_args):
        _, kwargs = captured_training_args.call_args
        fsdp_str = kwargs.get("fsdp", "")
        assert "full_shard" in fsdp_str


# ---------------------------------------------------------------------------
# Dynamic Padding Collator
# ---------------------------------------------------------------------------


class TestDynamicPaddingCollator:
    def test_collator_pads_to_max_in_batch(self, mocker):
        from src.train import dynamic_padding_collator
        from src.config import Config

        mocker.patch.object(
            Config, "pad_token_id", new_callable=mocker.PropertyMock, return_value=0
        )

        features = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([10, 11, 12]),
            },
            {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([10, 11])},
        ]

        collated = dynamic_padding_collator(features)

        assert collated["input_ids"].shape == (2, 3)
        assert collated["labels"][1, 2].item() == -100
        assert collated["attention_mask"][1, 2].item() == 0


# ---------------------------------------------------------------------------
# Compute Metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_compute_metrics_only_evaluates_after_sep(self, mocker):
        from src.train import compute_metrics

        mocker.patch.object(
            Config, "sep_token_id", new_callable=mocker.PropertyMock, return_value=5
        )
        mocker.patch.object(
            Config, "eos_token_id", new_callable=mocker.PropertyMock, return_value=9
        )

        labels = np.array([[1, 5, 10, 20, 9]])
        preds = np.array([[1, 10, 99, 0, 0]])  # Offset by 1 for the shift

        metrics = compute_metrics((preds, labels))
        assert metrics["ser"] == 0.5

    def test_compute_metrics_with_padding_ignored(self, mocker):
        from src.train import compute_metrics

        mocker.patch.object(
            Config, "sep_token_id", new_callable=mocker.PropertyMock, return_value=5
        )
        mocker.patch.object(
            Config, "eos_token_id", new_callable=mocker.PropertyMock, return_value=9
        )

        labels = np.array([[1, 5, 10, -100]])
        preds = np.array([[0, 10, 0, 0]])

        metrics = compute_metrics((preds, labels))
        assert metrics["ser"] == 0.0

    def test_compute_metrics_no_valid_symbols(self, mocker):
        from src.train import compute_metrics

        mocker.patch.object(
            Config, "sep_token_id", new_callable=mocker.PropertyMock, return_value=5
        )

        labels = np.array([[1, 2, 3]])
        preds = np.array([[1, 2, 3]])

        metrics = compute_metrics((preds, labels))
        assert metrics["ser"] == 0.0
