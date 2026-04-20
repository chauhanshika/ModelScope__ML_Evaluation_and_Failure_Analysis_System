import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    plt.show()


def plot_error_distribution(fp_count, fn_count):
    labels = ["False Positives", "False Negatives"]
    values = [fp_count, fn_count]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Error Distribution")
    plt.ylabel("Count")
    plt.show()
