import os

import pandas as pd
from langchain.tools import tool


@tool
def filter_columns(path: str, drop_columns: list) -> str:
    """
    Remove specific columns from a dataset and save the result as a new file.

    Args:
        path (str): Path to the input dataset file. Supported formats are .csv and .xlsx.
        drop_columns (list): List of column names to delete from the dataset.

    Returns:
        str: Path to the newly created dataset file with "_columns_filtered" suffix.

    Notes:
        - The resulting file will keep the same file format as the original (CSV/XLSX).
        - If one or more column names are not found, they will be ignored silently (dataset remains unchanged for them).
        - The function does not overwrite the original dataset, instead it creates a new file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = path.split(".")[-1].lower()
    if ext == "csv":
        df = pd.read_csv(path)
    elif ext in ["xls", "xlsx"]:
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format. Only CSV and Excel are allowed.")

    # Drop requested columns (ignore missing ones)
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])

    # Build new file path
    new_path = path.replace(f".{ext}", f"_columns_filtered.{ext}")

    if ext == "csv":
        df.to_csv(new_path, index=False)
    else:
        df.to_excel(new_path, index=False)

    return new_path
