from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, accuracy_score, classification_report
)

# set paths
path = Path(__file__).parent
img_dir = path / "imgs"
img_dir.mkdir(exist_ok=True)

# age buckets expected by the model
AGE_CLASSES = [
    "0-2", "3-9", "10-19", "20-29", "30-39", 
    "40-49", "50-59", "60-69", "more than 70"
]

def age_to_range(age):
    if age <= 2:
        return "0-2"
    elif age <= 9:
        return "3-9"
    elif age <= 19:
        return "10-19"
    elif age <= 29:
        return "20-29"
    elif age <= 39:
        return "30-39"
    elif age <= 49:
        return "40-49"
    elif age <= 59:
        return "50-59"
    elif age <= 69:
        return "60-69"
    else:
        return "more than 70"

def plot_conf_matrix(y_pred, y_true, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=AGE_CLASSES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=AGE_CLASSES)
    disp.plot(xticks_rotation=45, cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(img_dir / f"conf_matrix_{model_name}.png")
    plt.clf()

def plot_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Bar chart
    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="mako")
    plt.ylim(0, 1)
    plt.title(f"Performance Metrics - {model_name}")
    plt.tight_layout()
    plt.savefig(img_dir / f"metrics_{model_name}.png")
    plt.clf()

def plot_misclassifications(y_true, y_pred, model_name):
    df_mis = pd.DataFrame({"true": y_true, "pred": y_pred})
    mis_df = df_mis[df_mis["true"] != df_mis["pred"]]
    most_common = mis_df.groupby(["true", "pred"]).size().reset_index(name="count")
    most_common = most_common.sort_values(by="count", ascending=False).head(10)

    # Plot top 10 misclassifications
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="count",
        y=most_common.apply(lambda x: f"{x['true']} → {x['pred']}", axis=1),
        data=most_common,
        palette="rocket"
    )
    plt.xlabel("Count")
    plt.ylabel("Misclassification")
    plt.title(f"Top Misclassifications - {model_name}")
    plt.tight_layout()
    plt.savefig(img_dir / f"top_misclassifications_{model_name}.png")
    plt.clf()

def plot_conf_heatmap(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=AGE_CLASSES, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, xticklabels=AGE_CLASSES, yticklabels=AGE_CLASSES, cmap="YlGnBu", fmt=".2f")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Heatmap - {model_name}")
    plt.tight_layout()
    plt.savefig(img_dir / f"heatmap_{model_name}.png")
    plt.clf()

def main():
    df = pd.read_csv(path / "temp_output_labeled.csv")

    for model in ["age_classify_v001", "fairface_classifier", "vit_age_classifier"]:
        model_df = df[df["model_name"] == model]
        if model_df.empty:
            print(f"No data found for {model}")
            continue

        y_true = model_df["true_label"].apply(age_to_range)
        y_pred = model_df["label"]

        # Generate all plots
        plot_conf_matrix(y_pred, y_true, model)
        plot_metrics(y_true, y_pred, model)
        plot_misclassifications(y_true, y_pred, model)
        plot_conf_heatmap(y_true, y_pred, model)

    print(f"All visualizations saved to {img_dir}")

if __name__ == "__main__":
    main()