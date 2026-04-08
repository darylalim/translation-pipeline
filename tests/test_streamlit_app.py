import json
from unittest.mock import MagicMock, call, patch


class TestConstants:
    def test_model_id(self, app_module):
        assert app_module.MODEL_ID == "mlx-community/translategemma-4b-it-8bit"

    def test_max_new_tokens(self, app_module):
        assert app_module.MAX_NEW_TOKENS == 512


class TestLanguageConfiguration:
    def test_languages_has_9_entries(self, app_module):
        assert len(app_module.LANGUAGES) == 9

    def test_languages_values_are_str(self, app_module):
        for name, code in app_module.LANGUAGES.items():
            assert isinstance(code, str), f"{name}: expected str value"

    def test_source_langs_has_9_entries(self, app_module):
        assert len(app_module.SOURCE_LANGS) == 9

    def test_bcp47_codes(self, app_module):
        expected = {
            "English": "en",
            "Chinese": "zh",
            "Dutch": "nl",
            "French": "fr",
            "German": "de",
            "Indonesian": "id",
            "Italian": "it",
            "Spanish": "es",
            "Vietnamese": "vi",
        }
        for name, expected_code in expected.items():
            assert app_module.LANGUAGES[name] == expected_code, f"{name} code mismatch"

    def test_source_langs_is_sorted(self, app_module):
        assert app_module.SOURCE_LANGS == sorted(app_module.SOURCE_LANGS)

    def test_source_langs_contains_english(self, app_module):
        assert "English" in app_module.SOURCE_LANGS

    def test_english_can_target_all_non_english(self, app_module):
        non_english = sorted(n for n in app_module.LANGUAGES if n != "English")
        assert len(non_english) == 8
        assert non_english == sorted(non_english)

    def test_non_english_can_target_all_other_languages(self, app_module):
        for source in app_module.LANGUAGES:
            if source == "English":
                continue
            valid_targets = sorted(n for n in app_module.LANGUAGES if n != source)
            assert "English" in valid_targets, (
                f"{source} should be able to target English"
            )
            assert len(valid_targets) == 8, f"{source} should have 8 targets"
            assert source not in valid_targets, f"{source} should not target itself"


class TestBuildPrompt:
    def test_contains_language_names(self, app_module):
        prompt = app_module.build_prompt("Hello", "English", "en", "Spanish", "es")
        assert "English" in prompt
        assert "Spanish" in prompt

    def test_contains_language_codes(self, app_module):
        prompt = app_module.build_prompt("Hello", "English", "en", "Spanish", "es")
        assert "(en)" in prompt
        assert "(es)" in prompt

    def test_contains_source_text(self, app_module):
        prompt = app_module.build_prompt(
            "Translate me", "English", "en", "French", "fr"
        )
        assert "Translate me" in prompt

    def test_uses_gemma_chat_format(self, app_module):
        prompt = app_module.build_prompt("Hello", "English", "en", "Spanish", "es")
        assert prompt.startswith("<start_of_turn>user\n")
        assert "<end_of_turn>\n<start_of_turn>model\n" in prompt

    def test_returns_string(self, app_module):
        prompt = app_module.build_prompt("Hello", "English", "en", "Spanish", "es")
        assert isinstance(prompt, str)


class TestSwapLanguages:
    def test_swaps_source_and_target(self, app_module):
        mock_state = {"source_lang": "English", "target_lang": "Spanish"}
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._swap_languages()
        assert mock_state["source_lang"] == "Spanish"
        assert mock_state["target_lang"] == "English"

    def test_swap_is_reversible(self, app_module):
        mock_state = {"source_lang": "English", "target_lang": "French"}
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._swap_languages()
            app_module._swap_languages()
        assert mock_state["source_lang"] == "English"
        assert mock_state["target_lang"] == "French"

    def test_swap_copies_translation_to_source_text(self, app_module):
        mock_state = {
            "source_lang": "English",
            "target_lang": "Spanish",
            "translation_result": "hola",
        }
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._swap_languages()
        assert mock_state["source_text"] == "hola"
        assert "translation_result" not in mock_state

    def test_double_swap_with_translation_is_not_reversible(self, app_module):
        mock_state = {
            "source_lang": "English",
            "target_lang": "Spanish",
            "translation_result": "hola",
        }
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._swap_languages()
            app_module._swap_languages()
        assert mock_state["source_lang"] == "English"
        assert mock_state["target_lang"] == "Spanish"
        assert mock_state["source_text"] == "hola"
        assert "translation_result" not in mock_state

    def test_swap_without_translation_does_not_set_source_text(self, app_module):
        mock_state = {"source_lang": "English", "target_lang": "Spanish"}
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._swap_languages()
        assert "source_text" not in mock_state


class TestTranslate:
    def test_returns_string(self, patched_translate):
        result = patched_translate["translate"](
            "Hello", "English", "en", "Spanish", "es"
        )
        assert isinstance(result, str)

    def test_returns_generated_text(self, patched_translate):
        result = patched_translate["translate"](
            "Hello", "English", "en", "Spanish", "es"
        )
        assert result == "translated text"

    def test_generate_called_with_correct_args(self, app_module, patched_translate):
        patched_translate["translate"]("Hello", "English", "en", "Spanish", "es")
        expected_prompt = app_module.build_prompt(
            "Hello", "English", "en", "Spanish", "es"
        )
        app_module.generate.assert_called_once_with(
            patched_translate["model"],
            patched_translate["tokenizer"],
            prompt=expected_prompt,
            max_tokens=512,
        )

    def test_generate_called_exactly_once(self, app_module, patched_translate):
        patched_translate["translate"]("Hello", "English", "en", "Spanish", "es")
        assert app_module.generate.call_count == 1

    def test_strips_end_of_turn_token(self, app_module):
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
                return_value="hola mundo<end_of_turn>",
            ),
        ):
            result = app_module.translate(
                "hello world", "English", "en", "Spanish", "es"
            )
        assert result == "hola mundo"

    def test_strips_repeated_end_of_turn_tokens(self, app_module):
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
                return_value="hola mundo<end_of_turn><end_of_turn><end_of_turn>",
            ),
        ):
            result = app_module.translate(
                "hello world", "English", "en", "Spanish", "es"
            )
        assert result == "hola mundo"

    def test_clean_output_unchanged(self, app_module):
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
                return_value="hola mundo",
            ),
        ):
            result = app_module.translate(
                "hello world", "English", "en", "Spanish", "es"
            )
        assert result == "hola mundo"

    def test_strips_whitespace_around_translation(self, app_module):
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
                return_value="  hola mundo  <end_of_turn>",
            ),
        ):
            result = app_module.translate(
                "hello world", "English", "en", "Spanish", "es"
            )
        assert result == "hola mundo"

    def test_strips_content_after_end_of_turn(self, app_module):
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
                return_value="hola mundo<end_of_turn>extra garbage",
            ),
        ):
            result = app_module.translate(
                "hello world", "English", "en", "Spanish", "es"
            )
        assert result == "hola mundo"


class TestClipboardSanitization:
    """Verify json.dumps + < escaping produces safe JS string literals.

    The clipboard copy injects translation output into an inline <script> tag.
    These tests document the security assumptions for that code path.
    """

    def test_escapes_double_quotes(self):
        safe = json.dumps('text with "quotes"').replace("<", "\\u003c")
        assert '\\"' in safe

    def test_escapes_backslashes(self):
        safe = json.dumps("text with \\backslash").replace("<", "\\u003c")
        assert "\\\\" in safe

    def test_escapes_newlines(self):
        safe = json.dumps("line1\nline2").replace("<", "\\u003c")
        assert "\\n" in safe

    def test_escapes_script_close_tag(self):
        safe = json.dumps("</script>").replace("<", "\\u003c")
        assert "</script>" not in safe
        assert "\\u003c" in safe

    def test_result_is_valid_js_string_literal(self):
        safe = json.dumps("hello world").replace("<", "\\u003c")
        assert safe.startswith('"')
        assert safe.endswith('"')


class TestButtonLayout:
    def test_button_left_group_columns(self, app_module):
        calls = app_module.st.columns.call_args_list
        assert calls[2] == call([3, 2, 1])

    def test_button_right_group_columns(self, app_module):
        calls = app_module.st.columns.call_args_list
        assert calls[3] == call([1, 4, 1])


class TestLoadModel:
    def test_returns_model_and_tokenizer_from_load(self, app_module):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        with patch.object(
            app_module, "load", return_value=(mock_model, mock_tokenizer)
        ):
            model, tokenizer = app_module.load_model()
        assert model is mock_model
        assert tokenizer is mock_tokenizer

    def test_load_called_with_correct_model_id(self, app_module):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        with patch.object(
            app_module, "load", return_value=(mock_model, mock_tokenizer)
        ) as mock_load:
            app_module.load_model()
        mock_load.assert_called_once_with("mlx-community/translategemma-4b-it-8bit")

    def test_registers_end_of_turn_as_eos_token(self, app_module):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        with patch.object(
            app_module, "load", return_value=(mock_model, mock_tokenizer)
        ):
            app_module.load_model()
        mock_tokenizer.add_eos_token.assert_called_once_with("<end_of_turn>")
