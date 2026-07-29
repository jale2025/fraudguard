from src.data_helper import is_parquet
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


def test_is_parquet() -> None:
    """Verify that a valid Parquet file is correctly identified by is_parquet.

    This test creates a pandas DataFrame, exports it to a Parquet file, and
    asserts that `is_parquet` returns True for the generated file.
    """
    # Arrange
    data_dict = {"A": [1, 2], "B": [3, 4]}
    df = pd.DataFrame(data_dict)

    # Act
    dest_path = Path("./temp/test_data.parquet") 
    dest_path.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(dest_path, index=False)

    # Assert
    df_loaded = pd.read_parquet(dest_path)
    assert df_loaded.shape == (2, 2)

    assert is_parquet(dest_path) is True
