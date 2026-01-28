"""Pytest configuration and fixtures for translation pipeline tests."""
import pytest

from streamlit_app import Lang


@pytest.fixture
def english_source() -> Lang:
    """Fixture for English source language."""
    return Lang.ENGLISH


@pytest.fixture
def bidirectional_source() -> Lang:
    """Fixture for a bidirectional language (Japanese)."""
    return Lang.JAPANESE


@pytest.fixture
def non_english_only_source() -> Lang:
    """Fixture for a non-English-only source (Taiwanese Mandarin)."""
    return Lang.TAIWANESE_MANDARIN
