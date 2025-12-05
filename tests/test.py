import sys
import os
import pandas as pd
import pytest

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

from src.data_loader import load_data


def test_load_data_file_not_found():
    """Loader should raise ValueError if file does not exist"""
    with pytest.raises(ValueError):
        load_data("non_existent_file.txt")


def test_load_data_returns_dataframe(tmp_path):
    """Loader should correctly read a pipe-delimited file"""

    # Create temporary pipe-delimited file
    d = tmp_path / "dummy.txt"
    d.write_text("TotalPremium|TotalClaims\n100|50")

    # Load it
    df = load_data(str(d))

    # Assert DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert 'TotalPremium' in df.columns
    assert 'TotalClaims' in df.columns

    # Assert values parsed correctly
    assert df['TotalPremium'].iloc[0] == 100
    assert df['TotalClaims'].iloc[0] == 50