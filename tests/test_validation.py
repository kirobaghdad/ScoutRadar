import pandas as pd
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
