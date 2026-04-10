import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Loads dataset from given path.

    Args:
        path (str): Path to CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataset is empty")

    return df
