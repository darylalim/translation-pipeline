from dataclasses import asdict, is_dataclass
from unittest.mock import MagicMock, patch


class TestConstants:
    def test_model_id(self, app_module):
        assert app_module.MODEL_ID == "google/translategemma-4b-it"

    def test_max_new_tokens(self, app_module):
        assert app_module.MAX_NEW_TOKENS == 512

    def test_accepted_image_types(self, app_module):
        assert app_module.ACCEPTED_IMAGE_TYPES == ["png", "jpg", "jpeg", "webp"]


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


class TestTranslationResult:
    def test_field_access(self, app_module):
        result = app_module.TranslationResult(
            response="hello",
            prompt_eval_count=10,
            prompt_eval_duration=100,
            eval_count=5,
            eval_duration=200,
        )
        assert result.response == "hello"
        assert result.prompt_eval_count == 10
        assert result.prompt_eval_duration == 100
        assert result.eval_count == 5
        assert result.eval_duration == 200

    def test_asdict(self, app_module):
        result = app_module.TranslationResult(
            response="hello",
            prompt_eval_count=10,
            prompt_eval_duration=100,
            eval_count=5,
            eval_duration=200,
        )
        d = asdict(result)
        assert d == {
            "response": "hello",
            "prompt_eval_count": 10,
            "prompt_eval_duration": 100,
            "eval_count": 5,
            "eval_duration": 200,
        }

    def test_equality(self, app_module):
        kwargs = {
            "response": "hello",
            "prompt_eval_count": 10,
            "prompt_eval_duration": 100,
            "eval_count": 5,
            "eval_duration": 200,
        }
        assert app_module.TranslationResult(**kwargs) == app_module.TranslationResult(
            **kwargs
        )

    def test_is_dataclass(self, app_module):
        assert is_dataclass(app_module.TranslationResult)


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


class TestTranslate:
    def test_returns_translation_result(self, app_module, patched_translate):
        result = patched_translate["translate"](
            "Hello",
            "English",
            "en",
            "Spanish",
            "es",
        )
        assert isinstance(result, app_module.TranslationResult)

    def test_response_is_stripped(self, patched_translate):
        result = patched_translate["translate"](
            "Hello",
            "English",
            "en",
            "Spanish",
            "es",
        )
        assert result.response == "translated text"

    def test_tokenizer_called_with_correct_args(self, patched_translate):
        patched_translate["translate"](
            "Hello",
            "English",
            "en",
            "Spanish",
            "es",
        )
        call_kwargs = patched_translate["processor"].tokenizer.call_args[1]
        assert call_kwargs["return_tensors"] == "pt"
        assert call_kwargs["add_special_tokens"] is True

    def test_inputs_moved_to_model_device(self, patched_translate):
        patched_translate["translate"](
            "Hello",
            "English",
            "en",
            "Spanish",
            "es",
        )
        patched_translate["processor"].tokenizer.return_value.to.assert_called_with(
            "cpu"
        )

    def test_generate_called_with_correct_args(self, patched_translate):
        patched_translate["translate"](
            "Hello",
            "English",
            "en",
            "Spanish",
            "es",
        )
        call_kwargs = patched_translate["model"].generate.call_args[1]
        assert call_kwargs["do_sample"] is False
        assert call_kwargs["max_new_tokens"] == 512
        assert call_kwargs["top_p"] is None
        assert call_kwargs["top_k"] is None
        assert call_kwargs["eos_token_id"] == 107
        assert call_kwargs["pad_token_id"] == 0

    def test_decode_uses_skip_special_tokens(self, patched_translate):
        patched_translate["translate"](
            "Hello",
            "English",
            "en",
            "Spanish",
            "es",
        )
        call_kwargs = patched_translate["processor"].tokenizer.decode.call_args[1]
        assert call_kwargs["skip_special_tokens"] is True

    def test_prompt_eval_count_matches_input_length(self, patched_translate):
        result = patched_translate["translate"](
            "Hello",
            "English",
            "en",
            "Spanish",
            "es",
        )
        assert result.prompt_eval_count == 10

    def test_eval_count_matches_generated_length(self, patched_translate):
        result = patched_translate["translate"](
            "Hello",
            "English",
            "en",
            "Spanish",
            "es",
        )
        assert result.eval_count == 5

    def test_timing_durations_are_non_negative_ints(self, patched_translate):
        result = patched_translate["translate"](
            "Hello",
            "English",
            "en",
            "Spanish",
            "es",
        )
        assert isinstance(result.prompt_eval_duration, int)
        assert isinstance(result.eval_duration, int)
        assert result.prompt_eval_duration >= 0
        assert result.eval_duration >= 0

    def test_generate_called_exactly_once(self, patched_translate):
        patched_translate["translate"](
            "Hello",
            "English",
            "en",
            "Spanish",
            "es",
        )
        assert patched_translate["model"].generate.call_count == 1


class TestTranslateImage:
    def test_returns_translation_result(self, app_module, patched_translate_image):
        mock_image = MagicMock()
        result = patched_translate_image["translate_image"](mock_image, "en", "es")
        assert isinstance(result, app_module.TranslationResult)

    def test_response_is_stripped(self, patched_translate_image):
        mock_image = MagicMock()
        result = patched_translate_image["translate_image"](mock_image, "en", "es")
        assert result.response == "translated text"

    def test_apply_chat_template_called_with_correct_args(self, patched_translate_image):
        mock_image = MagicMock()
        patched_translate_image["translate_image"](mock_image, "en", "es")
        call_kwargs = patched_translate_image["processor"].apply_chat_template.call_args[1]
        assert "images" not in call_kwargs
        assert call_kwargs["tokenize"] is True
        assert call_kwargs["add_generation_prompt"] is True
        assert call_kwargs["return_dict"] is True
        assert call_kwargs["return_tensors"] == "pt"

    def test_apply_chat_template_message_format(self, patched_translate_image):
        mock_image = MagicMock()
        patched_translate_image["translate_image"](mock_image, "en", "de")
        call_args = patched_translate_image["processor"].apply_chat_template.call_args[0]
        messages = call_args[0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"][0]
        assert content["type"] == "image"
        assert content["source_lang_code"] == "en"
        assert content["target_lang_code"] == "de"

    def test_inputs_moved_to_model_device(self, patched_translate_image):
        mock_image = MagicMock()
        patched_translate_image["translate_image"](mock_image, "en", "es")
        call_args = patched_translate_image["processor"].apply_chat_template.return_value.to.call_args
        assert call_args[0][0] == "cpu"

    def test_generate_called_with_correct_args(self, patched_translate_image):
        mock_image = MagicMock()
        patched_translate_image["translate_image"](mock_image, "en", "es")
        call_kwargs = patched_translate_image["model"].generate.call_args[1]
        assert call_kwargs["do_sample"] is False
        assert call_kwargs["max_new_tokens"] == 512
        assert call_kwargs["top_p"] is None
        assert call_kwargs["top_k"] is None
        assert call_kwargs["eos_token_id"] == 107
        assert call_kwargs["pad_token_id"] == 0

    def test_decode_uses_skip_special_tokens(self, patched_translate_image):
        mock_image = MagicMock()
        patched_translate_image["translate_image"](mock_image, "en", "es")
        call_kwargs = patched_translate_image["processor"].tokenizer.decode.call_args[1]
        assert call_kwargs["skip_special_tokens"] is True

    def test_prompt_eval_count_matches_input_length(self, patched_translate_image):
        mock_image = MagicMock()
        result = patched_translate_image["translate_image"](mock_image, "en", "es")
        assert result.prompt_eval_count == 266

    def test_eval_count_matches_generated_length(self, patched_translate_image):
        mock_image = MagicMock()
        result = patched_translate_image["translate_image"](mock_image, "en", "es")
        assert result.eval_count == 5

    def test_timing_durations_are_non_negative_ints(self, patched_translate_image):
        mock_image = MagicMock()
        result = patched_translate_image["translate_image"](mock_image, "en", "es")
        assert isinstance(result.prompt_eval_duration, int)
        assert isinstance(result.eval_duration, int)
        assert result.prompt_eval_duration >= 0
        assert result.eval_duration >= 0

    def test_generate_called_exactly_once(self, patched_translate_image):
        mock_image = MagicMock()
        patched_translate_image["translate_image"](mock_image, "en", "es")
        assert patched_translate_image["model"].generate.call_count == 1


class TestLoadModel:
    def test_returns_4_tuple(self, app_module):
        mock_proc = self._make_processor()
        mock_model = self._make_model()
        with (
            patch.object(app_module, "AutoProcessor") as MockAutoProc,
            patch.object(app_module, "AutoModelForImageTextToText") as MockAutoModel,
        ):
            MockAutoProc.from_pretrained.return_value = mock_proc
            MockAutoModel.from_pretrained.return_value = mock_model
            result = app_module.load_model()
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_processor_loaded_with_correct_args(self, app_module):
        mock_proc = self._make_processor()
        mock_model = self._make_model()
        with (
            patch.object(app_module, "AutoProcessor") as MockAutoProc,
            patch.object(app_module, "AutoModelForImageTextToText") as MockAutoModel,
        ):
            MockAutoProc.from_pretrained.return_value = mock_proc
            MockAutoModel.from_pretrained.return_value = mock_model
            app_module.load_model()
        call_args, call_kwargs = MockAutoProc.from_pretrained.call_args
        assert call_args[0] == "google/translategemma-4b-it"
        assert call_kwargs["use_fast"] is True
        assert "token" not in call_kwargs

    def test_model_loaded_with_correct_args(self, app_module):
        mock_proc = self._make_processor()
        mock_model = self._make_model()
        with (
            patch.object(app_module, "AutoProcessor") as MockAutoProc,
            patch.object(app_module, "AutoModelForImageTextToText") as MockAutoModel,
        ):
            MockAutoProc.from_pretrained.return_value = mock_proc
            MockAutoModel.from_pretrained.return_value = mock_model
            app_module.load_model()
        call_args, call_kwargs = MockAutoModel.from_pretrained.call_args
        assert call_args[0] == "google/translategemma-4b-it"
        assert call_kwargs["device_map"] == "auto"
        assert call_kwargs["dtype"] == app_module.torch.bfloat16
        assert "token" not in call_kwargs

    def test_eos_token_extracted(self, app_module):
        mock_proc = self._make_processor(eos_id=999)
        mock_model = self._make_model()
        with (
            patch.object(app_module, "AutoProcessor") as MockAutoProc,
            patch.object(app_module, "AutoModelForImageTextToText") as MockAutoModel,
        ):
            MockAutoProc.from_pretrained.return_value = mock_proc
            MockAutoModel.from_pretrained.return_value = mock_model
            _, _, eos_token_id, _ = app_module.load_model()
        mock_proc.tokenizer.convert_tokens_to_ids.assert_called_with("<end_of_turn>")
        assert eos_token_id == 999

    def test_load_duration_is_non_negative_int(self, app_module):
        mock_proc = self._make_processor()
        mock_model = self._make_model()
        with (
            patch.object(app_module, "AutoProcessor") as MockAutoProc,
            patch.object(app_module, "AutoModelForImageTextToText") as MockAutoModel,
        ):
            MockAutoProc.from_pretrained.return_value = mock_proc
            MockAutoModel.from_pretrained.return_value = mock_model
            _, _, _, load_duration = app_module.load_model()
        assert isinstance(load_duration, int)
        assert load_duration >= 0

    def test_controlled_timing(self, app_module):
        mock_proc = self._make_processor()
        mock_model = self._make_model()
        with (
            patch.object(app_module, "AutoProcessor") as MockAutoProc,
            patch.object(app_module, "AutoModelForImageTextToText") as MockAutoModel,
            patch.object(app_module.time, "perf_counter_ns", side_effect=[1000, 6000]),
        ):
            MockAutoProc.from_pretrained.return_value = mock_proc
            MockAutoModel.from_pretrained.return_value = mock_model
            _, _, _, load_duration = app_module.load_model()
        assert load_duration == 5000

    @staticmethod
    def _make_processor(eos_id=107):
        proc = MagicMock()
        proc.tokenizer.convert_tokens_to_ids.return_value = eos_id
        return proc

    @staticmethod
    def _make_model():
        return MagicMock()
