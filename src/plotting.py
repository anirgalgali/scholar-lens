import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import wandb


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    font_scale: float = 0.8,
    do_log: bool = False,
):

    cm = confusion_matrix(y_true, y_pred)

    figsize_inch = max(10, len(class_names) * 0.65)
    sns.set_context("talk", font_scale=font_scale)

    fig, ax = plt.subplots(figsize=(figsize_inch, figsize_inch * 0.8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    fig.tight_layout()

    if do_log:
        image = wandb.Image(fig)
        plt.close(fig)
        return image

    plt.close(fig)
