import pandas as pd

def get_false_positives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cases where model predicted 1 but actual is 0
    """
    return df[(df["actual"] == 0) & (df["predicted"] == 1)]


def get_false_negatives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cases where model predicted 0 but actual is 1
    """
    return df[(df["actual"] == 1) & (df["predicted"] == 0)]


def summarize_failures(fp: pd.DataFrame, fn: pd.DataFrame) -> dict:
    """
    Summary of failure counts
    """
    return {
        "false_positives": len(fp),
        "false_negatives": len(fn)
    }
