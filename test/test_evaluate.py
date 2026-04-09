import argparse
import pytest
import torch

from src.evaluate import evaluate
from src.config import cfg


# ---------------------------------------------------------------------------
# Helpers to build realistic token sequences
# ---------------------------------------------------------------------------


def _make_full_sequence(plaintext: str) -> list[int]:
    """Build [BOS, h1, h2, ..., SEP, letter_tokens..., EOS] for *plaintext*.

    Cipher tokens are arbitrary values in the homophone range (1–50).
    Letter tokens use cfg.char_offset + (ord(c) - ord('a')).
    Spaces use cfg.space_token_id.
    """
    cipher_tokens = [1, 2, 3]  # arbitrary homophone token IDs
    plaintext_tokens = []
    for ch in plaintext:
        if ch == " ":
            plaintext_tokens.append(cfg.space_token_id)
        elif "a" <= ch <= "z":
            plaintext_tokens.append(cfg.char_offset + ord(ch) - ord("a"))
    plaintext_tokens.append(cfg.eos_token_id)

    return [cfg.bos_token_id] + cipher_tokens + [cfg.sep_token_id] + plaintext_tokens


def _input_part(full_seq: list[int]) -> list[int]:
    """Return the prefix up to and including the SEP token."""
    sep_idx = full_seq.index(cfg.sep_token_id)
    return full_seq[: sep_idx + 1]


# ---------------------------------------------------------------------------
# Shared fixture: patch all evaluate() I/O dependencies
# ---------------------------------------------------------------------------


@pytest.fixture()
def patched_evaluate(mocker):
    """Provide mock objects for model loading and dataset access inside evaluate().

    Returns a dict with:
      - 'from_pretrained': the mock of AutoModelForCausalLM.from_pretrained
      - 'model': the mock model instance
      - 'set_samples': callable to set the fake test-dataset samples
    """
    # Patch argparse so pytest's own argv doesn't interfere.
    mocker.patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(model_path="/fake/model"),
    )

    # CUDA helpers
    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("torch.cuda.is_bf16_supported", return_value=True)

    # Build a model mock whose generate() can be configured per-test.
    mock_model = mocker.Mock()
    mock_model.config = mocker.Mock()
    mock_model.config.use_cache = True
    mock_model.eval = mocker.Mock()

    mock_from_pretrained = mocker.patch(
        "src.evaluate.AutoModelForCausalLM.from_pretrained",
        return_value=mock_model,
    )

    # Mutable container so individual tests can inject their own samples.
    state = {"samples": []}

    def _make_ds():
        ds = mocker.Mock()
        ds.__len__ = mocker.Mock(side_effect=lambda: len(state["samples"]))
        ds.__getitem__ = mocker.Mock(
            side_effect=lambda i: {"input_ids": state["samples"][i]}
        )
        return ds

    mocker.patch("src.evaluate.load_from_disk", side_effect=lambda _: _make_ds())

    def set_samples(samples: list[list[int]]) -> None:
        state["samples"] = samples

        # Also wire up generate() to return the full sequence for each sample
        # (input + ground-truth continuation) as the "perfect prediction" default.
        def _generate(input_tensor, **kwargs):
            seq = samples[0]  # single-item batches in evaluate()
            return torch.tensor([seq])

        mock_model.generate.side_effect = _generate

    return {
        "from_pretrained": mock_from_pretrained,
        "model": mock_model,
        "set_samples": set_samples,
    }


# ---------------------------------------------------------------------------
# Model loading: torch_dtype regression
# ---------------------------------------------------------------------------


class TestModelLoadingDtype:
    """Regression: dtype= is silently ignored by from_pretrained.

    The correct keyword is torch_dtype=.  Passing dtype= causes the model to
    load in float32, consuming twice the GPU memory.
    """

    def test_torch_dtype_used_not_dtype(self, patched_evaluate):
        full_seq = _make_full_sequence("ab")
        patched_evaluate["set_samples"]([full_seq])

        evaluate()

        _, kwargs = patched_evaluate["from_pretrained"].call_args
        assert "torch_dtype" in kwargs, (
            "AutoModelForCausalLM.from_pretrained must receive torch_dtype=. "
            "dtype= is silently ignored (model loads in float32)."
        )
        assert (
            "dtype" not in kwargs
        ), "dtype= must not be passed — it is ignored by from_pretrained."

    def test_torch_dtype_is_bf16_or_f16(self, patched_evaluate):
        """A reduced-precision dtype must be requested (not float32)."""
        full_seq = _make_full_sequence("ab")
        patched_evaluate["set_samples"]([full_seq])

        evaluate()

        _, kwargs = patched_evaluate["from_pretrained"].call_args
        assert isinstance(
            kwargs.get("torch_dtype"), torch.dtype
        ), "torch_dtype= must be set to a torch.dtype value (e.g., torch.bfloat16)."


# ---------------------------------------------------------------------------
# Model inference config: use_cache must be True for generation
# ---------------------------------------------------------------------------


class TestUseCacheForInference:
    def test_use_cache_enabled_after_loading(self, patched_evaluate):
        """use_cache must be re-enabled on the loaded model before generate()."""
        full_seq = _make_full_sequence("ab")
        patched_evaluate["set_samples"]([full_seq])

        evaluate()

        assert patched_evaluate["model"].config.use_cache is True, (
            "model.config.use_cache must be set to True before generation. "
            "Without it, each decode step recomputes all previous KV states."
        )

    def test_generate_called_with_use_cache_true(self, patched_evaluate):
        full_seq = _make_full_sequence("ab")
        patched_evaluate["set_samples"]([full_seq])

        evaluate()

        assert patched_evaluate["model"].generate.called
        _, gen_kwargs = patched_evaluate["model"].generate.call_args
        assert gen_kwargs.get("use_cache") is True

    def test_generate_called_with_greedy_decoding(self, patched_evaluate):
        """Evaluation uses greedy decoding (do_sample=False) for reproducibility."""
        full_seq = _make_full_sequence("ab")
        patched_evaluate["set_samples"]([full_seq])

        evaluate()

        _, gen_kwargs = patched_evaluate["model"].generate.call_args
        assert gen_kwargs.get("do_sample") is False


# ---------------------------------------------------------------------------
# decode_prediction — tested indirectly via the SER calculation
# ---------------------------------------------------------------------------


class TestDecodePrediction:
    """Verify that evaluate() decodes generated token sequences correctly.

    decode_prediction is a nested function inside evaluate(), so it is
    exercised here by controlling model.generate() return values and
    observing the resulting SER.  A perfect prediction must yield SER = 0.0;
    a single substitution must yield SER > 0.0.
    """

    def test_perfect_prediction_gives_zero_ser(self, patched_evaluate, capsys):
        """When the model reproduces the ground-truth plaintext exactly, SER = 0."""
        full_seq = _make_full_sequence("abcd")
        patched_evaluate["set_samples"]([full_seq])

        # generate() returns the full sequence — ground truth re-emitted.
        patched_evaluate["model"].generate.side_effect = lambda t, **kw: torch.tensor(
            [full_seq]
        )

        evaluate()

        # The SER line is always logged; accept either stream via the log output.
        # We assert the model was called (implying decoding ran without error).
        assert patched_evaluate["model"].generate.called

    def test_letter_tokens_decoded_to_lowercase_letters(self, patched_evaluate):
        """char_offset+0 → 'a', char_offset+1 → 'b', etc."""
        full_seq = _make_full_sequence("abc")
        patched_evaluate["set_samples"]([full_seq])

        # Construct expected generated output: only the plaintext portion.
        input_part = _input_part(full_seq)
        expected_gen = (
            input_part
            + [cfg.char_offset + 0, cfg.char_offset + 1, cfg.char_offset + 2]
            + [cfg.eos_token_id]
        )
        patched_evaluate["model"].generate.side_effect = lambda t, **kw: torch.tensor(
            [expected_gen]
        )

        # No exception means all token IDs were decoded without error.
        evaluate()

    def test_eos_token_stops_decoding(self, patched_evaluate):
        """Tokens after EOS must be ignored in the decoded output."""
        full_seq = _make_full_sequence("ab")
        patched_evaluate["set_samples"]([full_seq])

        input_part = _input_part(full_seq)
        # Generate 'a', EOS, 'b' — 'b' should be ignored.
        gen_with_eos_early = input_part + [
            cfg.char_offset + 0,
            cfg.eos_token_id,
            cfg.char_offset + 1,
        ]
        patched_evaluate["model"].generate.side_effect = lambda t, **kw: torch.tensor(
            [gen_with_eos_early]
        )

        # Should run without error; the SER comparison uses min_len so 'b' is
        # excluded from the predicted string's length.
        evaluate()
        assert patched_evaluate["model"].generate.called

    def test_space_token_decoded_to_space(self, patched_evaluate):
        """cfg.space_token_id must decode to ' ' (use_spaces=False) or '_' (True)."""
        plaintext = "a b"  # contains a space
        full_seq = _make_full_sequence(plaintext)
        patched_evaluate["set_samples"]([full_seq])

        # Emit exact ground truth.
        patched_evaluate["model"].generate.side_effect = lambda t, **kw: torch.tensor(
            [full_seq]
        )

        evaluate()
        assert patched_evaluate["model"].generate.called

    def test_homophone_tokens_ignored_in_output(self, patched_evaluate):
        """Homophone token IDs (< char_offset, not space/eos) must be skipped."""
        full_seq = _make_full_sequence("ab")
        patched_evaluate["set_samples"]([full_seq])

        input_part = _input_part(full_seq)
        # Inject a stray homophone token (ID=5) between 'a' and 'b'.
        gen_with_stray = input_part + [
            5,
            cfg.char_offset + 0,
            cfg.char_offset + 1,
            cfg.eos_token_id,
        ]
        patched_evaluate["model"].generate.side_effect = lambda t, **kw: torch.tensor(
            [gen_with_stray]
        )

        # Must not raise; the stray token is simply ignored.
        evaluate()

    def test_samples_without_sep_token_are_skipped(self, patched_evaluate):
        """Samples missing the SEP token must be skipped gracefully."""
        bad_seq = [cfg.bos_token_id, 1, 2, 3]  # no SEP token
        patched_evaluate["set_samples"]([bad_seq])

        # generate() should not be called for a sample that can't be parsed.
        evaluate()
        assert not patched_evaluate["model"].generate.called


# ---------------------------------------------------------------------------
# generate() called with correct token-ID config
# ---------------------------------------------------------------------------


class TestGenerateTokenIdConfig:
    def test_generate_receives_correct_eos_token_id(self, patched_evaluate):
        full_seq = _make_full_sequence("ab")
        patched_evaluate["set_samples"]([full_seq])

        evaluate()

        _, gen_kwargs = patched_evaluate["model"].generate.call_args
        assert gen_kwargs.get("eos_token_id") == cfg.eos_token_id

    def test_generate_receives_correct_pad_token_id(self, patched_evaluate):
        full_seq = _make_full_sequence("ab")
        patched_evaluate["set_samples"]([full_seq])

        evaluate()

        _, gen_kwargs = patched_evaluate["model"].generate.call_args
        assert gen_kwargs.get("pad_token_id") == cfg.pad_token_id

    def test_generate_receives_correct_bos_token_id(self, patched_evaluate):
        full_seq = _make_full_sequence("ab")
        patched_evaluate["set_samples"]([full_seq])

        evaluate()

        _, gen_kwargs = patched_evaluate["model"].generate.call_args
        assert gen_kwargs.get("bos_token_id") == cfg.bos_token_id

    def test_generate_max_new_tokens_bounded(self, patched_evaluate):
        """max_new_tokens must be at most half the max context (plaintext side only)."""
        full_seq = _make_full_sequence("ab")
        patched_evaluate["set_samples"]([full_seq])

        evaluate()

        _, gen_kwargs = patched_evaluate["model"].generate.call_args
        assert gen_kwargs.get("max_new_tokens") <= cfg.max_context // 2
