import pandas as pd
import pytest
from src.data import normalize_target


def test_normalize_target_maps_positive_values_to_one():
    result = normalize_target(pd.DataFrame({"age": [50, 60, 70], "num": [0, 1, 4]}))
    assert result["target"].tolist() == [0, 1, 1]


def test_normalize_target_rejects_unknown_target_column():
    with pytest.raises(ValueError):
        normalize_target(pd.DataFrame({"age": [50], "labelx": [1]}))
