"""Tests for language configuration and target language selection."""
import pytest

from streamlit_app import (
    BIDIRECTIONAL_WITH_ENGLISH,
    ENGLISH_TO_ONLY,
    LANGUAGES,
    Lang,
    NON_ENGLISH_PAIRS,
    SOURCE_LANGS,
    get_target_languages,
)


class TestLanguageConfiguration:
    """Tests for language constant definitions."""

    def test_all_bidirectional_languages_in_languages_dict(self) -> None:
        """All bidirectional languages must exist in LANGUAGES dict."""
        for lang in BIDIRECTIONAL_WITH_ENGLISH:
            assert lang in LANGUAGES, f"'{lang}' missing from LANGUAGES"

    def test_all_english_to_only_languages_in_languages_dict(self) -> None:
        """All English-to-only languages must exist in LANGUAGES dict."""
        for lang in ENGLISH_TO_ONLY:
            assert lang in LANGUAGES, f"'{lang}' missing from LANGUAGES"

    def test_all_non_english_pair_languages_in_languages_dict(self) -> None:
        """All languages in non-English pairs must exist in LANGUAGES dict."""
        for source, targets in NON_ENGLISH_PAIRS.items():
            assert source in LANGUAGES, f"Source '{source}' missing from LANGUAGES"
            for target in targets:
                assert target in LANGUAGES, f"Target '{target}' missing from LANGUAGES"

    def test_no_overlap_between_bidirectional_and_english_to_only(self) -> None:
        """Bidirectional and English-to-only sets must not overlap."""
        overlap = BIDIRECTIONAL_WITH_ENGLISH & ENGLISH_TO_ONLY
        assert not overlap, f"Overlapping languages: {overlap}"

    def test_english_in_languages_dict(self) -> None:
        """English must be in LANGUAGES dict."""
        assert Lang.ENGLISH in LANGUAGES

    def test_source_langs_contains_english(self) -> None:
        """SOURCE_LANGS must contain English."""
        assert Lang.ENGLISH in SOURCE_LANGS

    def test_source_langs_contains_all_bidirectional(self) -> None:
        """SOURCE_LANGS must contain all bidirectional languages."""
        for lang in BIDIRECTIONAL_WITH_ENGLISH:
            assert lang in SOURCE_LANGS, f"'{lang}' missing from SOURCE_LANGS"

    def test_source_langs_excludes_english_to_only(self) -> None:
        """SOURCE_LANGS must not contain English-to-only languages (except those in non-English pairs)."""
        non_english_sources = set(NON_ENGLISH_PAIRS.keys())
        for lang in ENGLISH_TO_ONLY:
            if lang not in non_english_sources:
                assert lang not in SOURCE_LANGS, f"'{lang}' should not be in SOURCE_LANGS"

    def test_expected_language_count(self) -> None:
        """Verify expected number of languages."""
        assert len(BIDIRECTIONAL_WITH_ENGLISH) == 21
        assert len(ENGLISH_TO_ONLY) == 16


class TestGetTargetLanguages:
    """Tests for get_target_languages function."""

    def test_english_source_returns_all_non_english(self, english_source: str) -> None:
        """English source should return all bidirectional + English-to-only languages."""
        targets = get_target_languages(english_source)
        expected = BIDIRECTIONAL_WITH_ENGLISH | ENGLISH_TO_ONLY
        assert set(targets) == expected

    def test_english_source_excludes_english(self, english_source: str) -> None:
        """English source should not include English as target."""
        targets = get_target_languages(english_source)
        assert Lang.ENGLISH not in targets

    def test_bidirectional_source_includes_english(self, bidirectional_source: str) -> None:
        """Bidirectional source should include English as target."""
        targets = get_target_languages(bidirectional_source)
        assert Lang.ENGLISH in targets

    def test_bidirectional_with_non_english_pair_includes_both(self) -> None:
        """Bidirectional language with non-English pair should include both English and pair."""
        targets = get_target_languages(Lang.JAPANESE)
        assert Lang.ENGLISH in targets
        assert Lang.CHINESE in targets

    def test_cantonese_targets(self) -> None:
        """Cantonese should target English, Chinese, and Taiwanese Mandarin."""
        targets = get_target_languages(Lang.CANTONESE)
        assert Lang.ENGLISH in targets
        assert Lang.CHINESE in targets
        assert Lang.TAIWANESE_MANDARIN in targets
        assert len(targets) == 3

    def test_taiwanese_mandarin_targets(self, non_english_only_source: str) -> None:
        """Taiwanese Mandarin (non-English-only source) should only target Cantonese."""
        targets = get_target_languages(non_english_only_source)
        assert targets == [Lang.CANTONESE]

    def test_arabic_targets(self) -> None:
        """Arabic should target English and Swahili."""
        targets = get_target_languages(Lang.ARABIC)
        assert set(targets) == {Lang.ENGLISH, Lang.SWAHILI}

    def test_swahili_targets(self) -> None:
        """Swahili should target English, Arabic, and Chinese."""
        targets = get_target_languages(Lang.SWAHILI)
        assert set(targets) == {Lang.ENGLISH, Lang.ARABIC, Lang.CHINESE}

    def test_chinese_targets(self) -> None:
        """Chinese should target English, Japanese, and Swahili."""
        targets = get_target_languages(Lang.CHINESE)
        assert set(targets) == {Lang.ENGLISH, Lang.JAPANESE, Lang.SWAHILI}

    def test_simple_bidirectional_only_targets_english(self) -> None:
        """Simple bidirectional language (no non-English pairs) should only target English."""
        targets = get_target_languages(Lang.FRENCH)
        assert targets == [Lang.ENGLISH]

    def test_returns_sorted_list(self) -> None:
        """Targets should be returned as a sorted list."""
        targets = get_target_languages(Lang.ENGLISH)
        assert targets == sorted(targets)

    def test_unknown_source_returns_empty(self) -> None:
        """Unknown source language should return empty list."""
        targets = get_target_languages("Unknown Language")
        assert targets == []
