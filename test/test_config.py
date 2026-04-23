import json
import pytest
from src.config import Config, cfg


# ---------------------------------------------------------------------------
# Test types
# ---------------------------------------------------------------------------


class TestDefaultValues:
    def test_model_name_is_gemma4_e4b(self):
        assert isinstance(cfg.model_name_or_path, str)

    def test_default_learning_rate(self):
        assert isinstance(cfg.learning_rate, float)

    def test_default_batch_size_is_one(self):
        assert isinstance(cfg.batch_size, int)

    def test_default_grad_accum(self):
        assert isinstance(cfg.grad_accum, int)

    def test_default_epochs(self):
        assert isinstance(cfg.epochs, int)

    def test_bf16_enabled(self):
        assert isinstance(cfg.bf16, bool)

    def test_fp16_disabled(self):
        assert isinstance(cfg.fp16, bool)

    def test_tf32_enabled(self):
        assert isinstance(cfg.tf32, bool)

    def test_gradient_checkpointing_enabled(self):
        assert isinstance(cfg.gradient_checkpointing, bool)

    def test_weight_decay(self):
        assert isinstance(cfg.weight_decay, float)


# ---------------------------------------------------------------------------
# Token IDs: ordering and uniqueness
# ---------------------------------------------------------------------------


class TestTokenIds:
    def test_pad_token_id_is_zero(self):
        assert cfg.pad_token_id == 0

    def test_sep_token_id_follows_homophones(self):
        assert cfg.sep_token_id == cfg.unique_homophones + 1

    def test_space_token_id_follows_sep(self):
        assert cfg.space_token_id == cfg.sep_token_id + 1

    def test_bos_token_id_follows_space(self):
        assert cfg.bos_token_id == cfg.space_token_id + 1

    def test_eos_token_id_follows_bos(self):
        assert cfg.eos_token_id == cfg.bos_token_id + 1

    def test_char_offset_follows_eos(self):
        assert cfg.char_offset == cfg.eos_token_id + 1

    def test_special_token_ids_are_strictly_increasing(self):
        ids = [
            cfg.sep_token_id,
            cfg.space_token_id,
            cfg.bos_token_id,
            cfg.eos_token_id,
            cfg.char_offset,
        ]
        assert ids == sorted(
            ids
        ), "Special token IDs must be in strictly ascending order."
        assert len(ids) == len(set(ids)), "Special token IDs must all be unique."

    def test_pad_token_not_equal_to_any_special_token(self):
        special = {
            cfg.sep_token_id,
            cfg.space_token_id,
            cfg.bos_token_id,
            cfg.eos_token_id,
        }
        assert cfg.pad_token_id not in special

    def test_eos_is_below_char_offset(self):
        """EOS must be < char_offset so decode_prediction can distinguish them."""
        assert cfg.eos_token_id < cfg.char_offset

    def test_letter_z_fits_within_vocab(self):
        """Token ID for 'z' (char_offset + 25) must be inside the vocabulary."""
        max_char_token = cfg.char_offset + 25
        assert max_char_token < cfg.vocab_size


# ---------------------------------------------------------------------------
# Vocabulary size: padding to 64 and coverage
# ---------------------------------------------------------------------------


class TestVocabSize:
    def test_vocab_size_is_multiple_of_64(self):
        """vocab_size is rounded up to the nearest multiple of 64 for tensor-core alignment."""
        assert cfg.vocab_size % 64 == 0

    def test_vocab_size_covers_all_homophones(self):
        assert cfg.vocab_size > cfg.unique_homophones

    def test_load_homophones_updates_unique_homophones(self, tmp_path, mocker):
        c = Config()
        meta = {"max_symbol_id": 3000}
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text(json.dumps(meta))

        mocker.patch("os.path.join", return_value=str(meta_path))
        mocker.patch("os.path.exists", return_value=True)
        c.load_homophones()

        assert c.unique_homophones == 3000

    def test_load_homophones_keeps_vocab_size_multiple_of_64(self, tmp_path, mocker):
        c = Config()
        meta = {"max_symbol_id": 3001}  # not naturally aligned
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text(json.dumps(meta))

        mocker.patch("os.path.join", return_value=str(meta_path))
        mocker.patch("os.path.exists", return_value=True)
        c.load_homophones()

        assert c.vocab_size % 64 == 0

    def test_load_homophones_includes_letters_and_buffer(self, tmp_path, mocker):
        c = Config()
        meta = {"max_symbol_id": 3000}
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text(json.dumps(meta))

        mocker.patch("os.path.join", return_value=str(meta_path))
        mocker.patch("os.path.exists", return_value=True)
        c.load_homophones()

        assert c.vocab_size >= 3000 + cfg.unique_letters + cfg.buffer

    def test_load_homophones_missing_file_raises_error(self, mocker):
        mocker.patch("os.path.exists", return_value=False)

        # Expect the FileNotFoundError
        with pytest.raises(FileNotFoundError):
            cfg.load_homophones()

    def test_load_homophones_invalid_json_fails(self, tmp_path, mocker):
        bad_path = tmp_path / "metadata.json"
        bad_path.write_text("{ this is not valid json }")

        mocker.patch("os.path.join", return_value=str(bad_path))
        mocker.patch("os.path.exists", return_value=True)

        with pytest.raises(ValueError):
            cfg.load_homophones()

    def test_load_homophones_missing_key_fails(self, tmp_path, mocker):
        bad_path = tmp_path / "metadata.json"
        bad_path.write_text(json.dumps({"wrong_key": 999}))

        mocker.patch("os.path.join", return_value=str(bad_path))
        mocker.patch("os.path.exists", return_value=True)

        with pytest.raises(ValueError):
            cfg.load_homophones()
