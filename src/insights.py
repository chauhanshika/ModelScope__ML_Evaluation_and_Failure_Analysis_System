def generate_insights(metrics: dict, failures: dict) -> str:
    """
    Generate human-readable insights from metrics and failure analysis
    """

    acc = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]

    fp = failures["false_positives"]
    fn = failures["false_negatives"]

    insights = []

    # Accuracy insight
    if acc > 0.8:
        insights.append("Model shows strong overall accuracy.")
    else:
        insights.append("Model accuracy is relatively low, indicating room for improvement.")

    # Precision vs Recall
    if precision > recall:
        insights.append("Model is conservative in predicting positives (high precision, low recall).")
    elif recall > precision:
        insights.append("Model captures positives well but has more false alarms (high recall).")
    else:
        insights.append("Model has balanced precision and recall.")

    # Failure analysis
    if fn > fp:
        insights.append("High false negatives — model is missing actual positive cases.")
    elif fp > fn:
        insights.append("High false positives — model is over-predicting positives.")
    else:
        insights.append("False positives and negatives are balanced.")

    # F1 Score
    if f1 < 0.6:
        insights.append("Low F1 score — poor balance between precision and recall.")
    elif f1 < 0.8:
        insights.append("Moderate F1 score — can be improved.")
    else:
        insights.append("Strong F1 score — good model balance.")

    return "\n".join(insights)
