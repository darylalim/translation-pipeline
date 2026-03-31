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

    # Text tab columns
    text_left_col, text_right_col = MagicMock(), MagicMock()
    text_clear_col, text_count_col = MagicMock(), MagicMock()
    text_spacer_col, text_copy_col, text_download_col = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    # Image tab columns
    img_left_col, img_right_col = MagicMock(), MagicMock()
    img_spacer_col, img_copy_col, img_download_col = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    # Column calls are position-dependent — update this list if st.columns
    # calls are added, removed, or reordered in streamlit_app.py:
    # 1. Text tab language selectors [5, 1, 5]
    # 2. Text tab content [2]
    # 3. Text tab clear/count [2]
    # 4. Text tab copy/download [8, 1, 1]
    # 5. Image tab language selectors [5, 1, 5]
    # 6. Image tab content [2]
    # 7. Image tab copy/download [8, 1, 1]
    _columns_calls = iter(
        [
            (col1, col_swap, col2),
            (text_left_col, text_right_col),
            (text_clear_col, text_count_col),
            (text_spacer_col, text_copy_col, text_download_col),
            (col1, col_swap, col2),
            (img_left_col, img_right_col),
            (img_spacer_col, img_copy_col, img_download_col),
        ]
    )

    def _mock_columns(*args, **kwargs):
        try:
            return next(_columns_calls)
        except StopIteration:
            n = args[0] if args else 2
            if isinstance(n, list):
                n = len(n)
            return tuple(MagicMock() for _ in range(n))

    mock_st.columns = MagicMock(side_effect=_mock_columns)
    mock_st.button.return_value = False

    # Mock st.tabs to return context managers
    text_tab = MagicMock()
    image_tab = MagicMock()
    mock_st.tabs.return_value = [text_tab, image_tab]

    # file_uploader returns None so Image.open is not called at import time
    mock_st.file_uploader.return_value = None

    mock_torch = MagicMock()
    mock_dotenv = MagicMock()
    mock_transformers = MagicMock()
    mock_components_v1 = MagicMock()
    mock_components = MagicMock()
    mock_components.v1 = mock_components_v1
    mock_components.__path__ = []
    mock_st.components = mock_components
    mock_st.__path__ = []  # make Python treat mock_st as a package

    patches = {
        "streamlit": mock_st,
        "streamlit.components": mock_components,
        "streamlit.components.v1": mock_components_v1,
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


@pytest.fixture()
def mock_processor_image():
    """MagicMock processor configured for image translation."""
    processor = MagicMock()

    mock_input_ids = MagicMock()
    mock_input_ids.shape = (1, 266)  # 256 image tokens + 10 prompt tokens

    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = lambda self, key: (
        mock_input_ids if key == "input_ids" else MagicMock()
    )
    mock_inputs.to.return_value = mock_inputs

    processor.apply_chat_template.return_value = mock_inputs
    processor.tokenizer.decode.return_value = "  translated text  "
    processor.tokenizer.pad_token_id = 0
    processor.tokenizer.convert_tokens_to_ids.return_value = 107

    return processor


@pytest.fixture()
def patched_translate_image(app_module, mock_model, mock_processor_image):
    """Patch load_model for image translation tests."""
    with patch.object(
        app_module,
        "load_model",
        return_value=(mock_model, mock_processor_image, 107, 5_000_000),
    ):
        yield {
            "translate_image": app_module.translate_image,
            "model": mock_model,
            "processor": mock_processor_image,
        }
