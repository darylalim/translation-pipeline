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

    # Content columns
    left_col, right_col = MagicMock(), MagicMock()
    # Button columns — left nested [3, 2, 1] (inside left_col)
    btn_translate_col = MagicMock()
    btn_left_spacer = MagicMock()
    btn_clear_col = MagicMock()
    # Button columns — right nested [1, 4, 1] (inside right_col)
    copy_col = MagicMock()
    btn_right_spacer = MagicMock()
    download_col = MagicMock()

    # Column calls are position-dependent — update this list if st.columns
    # calls are added, removed, or reordered in streamlit_app.py:
    # 1. Language selectors [10, 1, 10]
    # 2. Content columns [2]
    # 3. Buttons left nested [3, 2, 1]
    # 4. Buttons right nested [1, 4, 1]
    _columns_calls = iter(
        [
            (col1, col_swap, col2),
            (left_col, right_col),
            (btn_translate_col, btn_left_spacer, btn_clear_col),
            (copy_col, btn_right_spacer, download_col),
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

    mock_mlx_lm = MagicMock()
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
        "mlx_lm": mock_mlx_lm,
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
def patched_translate(app_module):
    """Patch load_model and generate for translation tests."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    with (
        patch.object(
            app_module,
            "load_model",
            return_value=(mock_model, mock_tokenizer),
        ),
        patch.object(
            app_module,
            "generate",
            return_value="translated text",
        ),
    ):
        yield {
            "translate": app_module.translate,
            "model": mock_model,
            "tokenizer": mock_tokenizer,
        }
