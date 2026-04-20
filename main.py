from src.config import DATA_PATH
from src.data_loader import load_data
from src.metrics import accuracy, precision, recall, f1_score
from src.failure_analysis import get_false_positives, get_false_negatives, summarize_failures
from src.visualization import plot_confusion_matrix, plot_error_distribution
from src.db import create_table, insert_data, query_prediction_distribution
from src.insights import generate_insights


def main():
    df = load_data(DATA_PATH)

    # SQL Layer
    create_table()
    insert_data(df)

    print("\n=== SQL Query Result ===")
    print(query_prediction_distribution())

    y_true = df["actual"].values
    y_pred = df["predicted"].values

    # Metrics
    acc = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n=== Model Evaluation Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    metrics_dict = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

    # Failure Analysis
    fp = get_false_positives(df)
    fn = get_false_negatives(df)
    failure_summary = summarize_failures(fp, fn)

    print("\n=== Failure Analysis ===")
    print(f"False Positives: {failure_summary['false_positives']}")
    print(f"False Negatives: {failure_summary['false_negatives']}")

    # 🔥 Insights
    insights = generate_insights(metrics_dict, failure_summary)

    print("\n=== Insights ===")
    print(insights)

    # Visualization
    plot_confusion_matrix(y_true, y_pred)
    plot_error_distribution(
        failure_summary["false_positives"],
        failure_summary["false_negatives"]
    )


if __name__ == "__main__":
    main()
