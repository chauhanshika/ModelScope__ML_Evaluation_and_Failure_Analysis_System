from src.config import DATA_PATH
from src.data_loader import load_data
from src.metrics import accuracy, precision, recall, f1_score
from src.failure_analysis import get_false_positives, get_false_negatives, summarize_failures


def main():
    df = load_data(DATA_PATH)

    y_true = df["actual"].values
    y_pred = df["predicted"].values

    print("=== Model Evaluation Metrics ===")
    print(f"Accuracy : {accuracy(y_true, y_pred):.4f}")
    print(f"Precision: {precision(y_true, y_pred):.4f}")
    print(f"Recall   : {recall(y_true, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred):.4f}")

    # 🔥 Failure Analysis
    fp = get_false_positives(df)
    fn = get_false_negatives(df)

    summary = summarize_failures(fp, fn)

    print("\n=== Failure Analysis ===")
    print(f"False Positives: {summary['false_positives']}")
    print(f"False Negatives: {summary['false_negatives']}")

    print("\nSample False Positives:")
    print(fp.head())

    print("\nSample False Negatives:")
    print(fn.head())


if __name__ == "__main__":
    main()
