import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy = correct predictions / total
    """
    return np.mean(y_true == y_pred)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Precision = TP / (TP + FP)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if (tp + fp) == 0:
        return 0.0

    return tp / (tp + fp)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Recall = TP / (TP + FN)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if (tp + fn) == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    if (p + r) == 0:
        return 0.0

    return 2 * (p * r) / (p + r)
