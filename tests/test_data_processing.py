# Тести для обробки даних
import pytest
from src.core.data_processing import preprocess_data

def test_preprocess_data():
    data = {"col1": [1, 2, None]}
    result = preprocess_data(data)
    assert result["col1"].isna().sum() == 0
