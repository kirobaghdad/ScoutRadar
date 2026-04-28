import pandas as pd
import pytest

from src.validation.api import load_api_key
from src.validation.utils import _parse_datetime, _normalize_text

def test_normalize_text():
    """Test text normalization rules."""
    assert _normalize_text(" Lionel Messi  ") == "lionel messi"
    assert _normalize_text("Müller") == "muller"
    assert _normalize_text(pd.NA) == ""
    assert _normalize_text(None) == ""

def test_parse_datetime():
    """Test standard date parsing."""
    test_series = pd.Series(["2024-01-01", "Invalid Date", None])
    parsed = _parse_datetime(test_series)
    
    assert parsed.dt.year[0] == 2024
    assert pd.isna(parsed[1])
    assert pd.isna(parsed[2])


def test_load_api_key_prefers_environment_variable(tmp_path, monkeypatch):
    key_file = tmp_path / "key_api.txt"
    key_file.write_text("file_key_1234567890", encoding="utf-8")
    monkeypatch.setenv("API_FOOTBALL_KEY", "env_key_1234567890")

    assert load_api_key(key_file) == "env_key_1234567890"


def test_load_api_key_falls_back_to_file(tmp_path, monkeypatch):
    key_file = tmp_path / "key_api.txt"
    key_file.write_text("API_FOOTBALL_KEY=file_key_1234567890", encoding="utf-8")
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)

    assert load_api_key(key_file) == "file_key_1234567890"


def test_load_api_key_requires_env_or_file(tmp_path, monkeypatch):
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)

    with pytest.raises(FileNotFoundError, match="Set API_FOOTBALL_KEY"):
        load_api_key(tmp_path / "missing_key.txt")
