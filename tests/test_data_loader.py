import sys
import os
import pandas as pd
import pytest

# --- THE FIX IS HERE ---
# 1. Get the directory of THIS file (tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Get the parent directory (alpha-care-insurance/)
parent_dir = os.path.dirname(current_dir)
# 3. Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

# Now Python can find 'src' inside the parent directory
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