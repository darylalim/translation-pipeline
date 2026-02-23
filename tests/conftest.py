import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="session")
def app_module():
    """Import streamlit_app with all heavy dependencies mocked."""
    mock_st = MagicMock()
    mock_st.cache_resource = lambda f: f
    mock_st.session_state = {}

    col1, col_swap, col2 = MagicMock(), MagicMock(), MagicMock()
    col1.selectbox.return_value = "English"
    col2.selectbox.return_value = "Spanish"
    left_col, right_col = MagicMock(), MagicMock()
    mock_st.columns.side_effect = [
        (col1, col_swap, col2),
        (left_col, right_col),
    ]
    mock_st.button.return_value = False

    mock_torch = MagicMock()
    mock_dotenv = MagicMock()
    mock_transformers = MagicMock()

    patches = {
        "streamlit": mock_st,
        "torch": mock_torch,
        "dotenv": mock_dotenv,
        "transformers": mock_transformers,
    }

    originals = {}
    for mod_name, mock_obj in patches.items():
        originals[mod_name] = sys.modules.get(mod_name)
        sys.modules[mod_name] = mock_obj

    if "streamlit_app" in sys.modules:
        del sys.modules["streamlit_app"]
    module = importlib.import_module("streamlit_app")

    for mod_name, orig in originals.items():
        if orig is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = orig

    return module


@pytest.fixture()
def mock_processor():
    """MagicMock processor with configured tokenizer."""
    processor = MagicMock()

    mock_input_ids = MagicMock()
    mock_input_ids.shape = (1, 10)

    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = lambda self, key: (
        mock_input_ids if key == "input_ids" else MagicMock()
    )
    mock_inputs.to.return_value = mock_inputs

    processor.tokenizer.return_value = mock_inputs
    processor.tokenizer.decode.return_value = "  translated text  "
    processor.tokenizer.pad_token_id = 0
    processor.tokenizer.convert_tokens_to_ids.return_value = 107

    return processor


@pytest.fixture()
def mock_model():
    """MagicMock model with configured generate output."""
    model = MagicMock()
    model.device = "cpu"

    mock_generated = MagicMock()
    mock_generated.shape = (5,)

    mock_output_sequence = MagicMock()
    mock_output_sequence.__getitem__ = lambda self, key: mock_generated
    mock_output = MagicMock()
    mock_output.__getitem__ = lambda self, key: mock_output_sequence

    model.generate.return_value = mock_output

    return model


@pytest.fixture()
def patched_translate(app_module, mock_model, mock_processor):
    """Patch load_model for translation tests."""
    with patch.object(
        app_module,
        "load_model",
        return_value=(mock_model, mock_processor, 107, 5_000_000),
    ):
        yield {
            "translate": app_module.translate,
            "model": mock_model,
            "processor": mock_processor,
        }
