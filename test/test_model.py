import pytest
import torch

from src.config import cfg
from src.model import get_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_dependencies(mocker):
    """
    Sets up the base mock objects for AutoConfig and AutoModelForCausalLM
    so we don't accidentally download or load real artifacts.
    """
    # Mock Config
    mock_config = mocker.Mock()
    mock_config.initializer_range = 0.02

    # Mock Model
    mock_model = mocker.Mock()
    mock_model.config = mock_config
    mock_model.num_parameters.return_value = 3_000_000_000
    mock_model.get_memory_footprint.return_value = 6_000_000_000

    # Mock weight tensors and their methods
    mock_model.model.embed_tokens.weight.data.normal_ = mocker.Mock()
    mock_model.lm_head.weight.data.normal_ = mocker.Mock()

    # Patch the Hugging Face classes
    mock_auto_config = mocker.patch(
        "src.model.AutoConfig.from_pretrained", return_value=mock_config
    )
    mock_auto_model = mocker.patch(
        "src.model.AutoModelForCausalLM.from_pretrained", return_value=mock_model
    )

    return mock_config, mock_model, mock_auto_config, mock_auto_model


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestGetModel:
    def test_loads_config_with_correct_args(self, mock_dependencies):
        mock_config, _, mock_auto_config, _ = mock_dependencies

        get_model()

        mock_auto_config.assert_called_once_with(
            cfg.model_name_or_path,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )

    def test_sets_custom_token_ids_on_config(self, mock_dependencies):
        mock_config, _, _, _ = mock_dependencies

        get_model()

        assert mock_config.pad_token_id == cfg.pad_token_id
        assert mock_config.bos_token_id == cfg.bos_token_id
        assert mock_config.eos_token_id == cfg.eos_token_id

    def test_loads_model_with_correct_args(self, mock_dependencies):
        mock_config, mock_model, _, mock_auto_model = mock_dependencies

        get_model()

        mock_auto_model.assert_called_once_with(
            cfg.model_name_or_path,
            config=mock_config,
            torch_dtype=torch.bfloat16,
        )

    def test_resizes_token_embeddings(self, mock_dependencies):
        _, mock_model, _, _ = mock_dependencies

        get_model()

        mock_model.resize_token_embeddings.assert_called_once_with(cfg.vocab_size)

    def test_reinitializes_untied_weights(self, mock_dependencies):
        """
        If embed_tokens and lm_head have different memory pointers,
        both sets of weights must be re-initialized.
        """
        _, mock_model, _, _ = mock_dependencies

        # Simulate untied weights by returning different fake pointer IDs
        mock_model.model.embed_tokens.weight.data_ptr.return_value = 11111
        mock_model.lm_head.weight.data_ptr.return_value = 22222

        get_model()

        mock_model.model.embed_tokens.weight.data.normal_.assert_called_once_with(
            mean=0.0, std=0.02
        )
        mock_model.lm_head.weight.data.normal_.assert_called_once_with(
            mean=0.0, std=0.02
        )

    def test_skips_lm_head_reinit_when_tied(self, mock_dependencies):
        """
        If embed_tokens and lm_head share the exact same memory pointer,
        the code must avoid double-initializing the weights.
        """
        _, mock_model, _, _ = mock_dependencies

        # Simulate tied weights by returning the same fake pointer ID
        mock_model.model.embed_tokens.weight.data_ptr.return_value = 99999
        mock_model.lm_head.weight.data_ptr.return_value = 99999

        get_model()

        # The embed tokens should be initialized, but lm_head should be skipped
        mock_model.model.embed_tokens.weight.data.normal_.assert_called_once_with(
            mean=0.0, std=0.02
        )
        mock_model.lm_head.weight.data.normal_.assert_not_called()
