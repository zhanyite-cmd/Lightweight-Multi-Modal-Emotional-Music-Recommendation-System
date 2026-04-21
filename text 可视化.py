import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
)

# ========================= PATH CONFIG =========================
BASE_OUTPUT_DIR = r"D:\树莓派\face_detection-检测+分类+大模型\face_detection-检测+分类+大模型\face_detect\runs\Goemotions\distilbert_goemotions_outputs"
CHECKPOINT_DIR = os.path.join(BASE_OUTPUT_DIR, "checkpoint-1358")
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "re_visualized")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# files in base output dir
METRICS_CSV = os.path.join(BASE_OUTPUT_DIR, "trainer_metrics.csv")
LOG_HISTORY_JSON = os.path.join(BASE_OUTPUT_DIR, "trainer_state_log_history.json")
PREDICTIONS_CSV = os.path.join(BASE_OUTPUT_DIR, "test_predictions.csv")

# file in checkpoint dir
TRAINER_STATE_JSON = os.path.join(CHECKPOINT_DIR, "trainer_state.json")

# output figures / reports
CURVE_PATH = os.path.join(OUTPUT_DIR, "training_curves.png")
CM_COUNT_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix_counts.png")
CM_NORM_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix_normalized.png")
ACCURACY_TXT_PATH = os.path.join(OUTPUT_DIR, "accuracy.txt")
CLASSIFICATION_REPORT_PATH = os.path.join(OUTPUT_DIR, "classification_report.txt")


# ========================= UTIL =========================
def smooth(values, factor=0.85):
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return values

    smoothed = []
    for v in values:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + v * (1 - factor))
        else:
            smoothed.append(v)
    return np.array(smoothed)


# ========================= LOAD METRICS =========================
def load_metrics():
    # 1. 优先读 trainer_metrics.csv
    if os.path.exists(METRICS_CSV):
        print(f"Loaded metrics CSV: {METRICS_CSV}")
        return pd.read_csv(METRICS_CSV)

    # 2. 再读 trainer_state_log_history.json
    if os.path.exists(LOG_HISTORY_JSON):
        print(f"Loaded log_history JSON: {LOG_HISTORY_JSON}")
        with open(LOG_HISTORY_JSON, "r", encoding="utf-8") as f:
            logs = json.load(f)

    # 3. 最后读 checkpoint 下的 trainer_state.json
    elif os.path.exists(TRAINER_STATE_JSON):
        print(f"Loaded trainer_state JSON: {TRAINER_STATE_JSON}")
        with open(TRAINER_STATE_JSON, "r", encoding="utf-8") as f:
            state = json.load(f)
            logs = state.get("log_history", [])
    else:
        raise FileNotFoundError(
            "No metrics file found:\n"
            f"- {METRICS_CSV}\n"
            f"- {LOG_HISTORY_JSON}\n"
            f"- {TRAINER_STATE_JSON}"
        )

    rows = []
    for log in logs:
        rows.append({
            "step": log.get("step", np.nan),
            "epoch": log.get("epoch", np.nan),
            "train_loss": log.get("loss", np.nan),
            "val_loss": log.get("eval_loss", np.nan),
            "val_accuracy": log.get("eval_accuracy", np.nan),
            "val_f1": log.get("eval_f1", np.nan),
        })

    df = pd.DataFrame(rows)
    df.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
    print(f"Generated metrics CSV: {METRICS_CSV}")
    return df


# ========================= LOAD PREDICTIONS =========================
def load_predictions():
    if not os.path.exists(PREDICTIONS_CSV):
        raise FileNotFoundError(f"Prediction file not found: {PREDICTIONS_CSV}")

    df = pd.read_csv(PREDICTIONS_CSV)
    print(f"Loaded predictions CSV: {PREDICTIONS_CSV}")
    print("Prediction columns:", list(df.columns))
    return df


# ========================= DETECT LABEL COLUMNS =========================
def detect_label_columns(df):
    """
    自动适配常见列名:
    - true_label / pred_label
    - label / prediction
    - y_true / y_pred
    - true / pred
    """
    possible_true_cols = ["true_label", "label", "y_true", "true", "gold"]
    possible_pred_cols = ["pred_label", "prediction", "y_pred", "pred", "predict"]

    true_col = None
    pred_col = None

    for col in possible_true_cols:
        if col in df.columns:
            true_col = col
            break

    for col in possible_pred_cols:
        if col in df.columns:
            pred_col = col
            break

    if true_col is None or pred_col is None:
        raise ValueError(
            "Cannot detect label columns.\n"
            f"Current columns: {list(df.columns)}\n"
            f"Expected true columns: {possible_true_cols}\n"
            f"Expected pred columns: {possible_pred_cols}"
        )

    return true_col, pred_col


# ========================= PLOT CURVES =========================
def plot_curves(df):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11

    train_df = df.dropna(subset=["train_loss"])
    val_df = df.dropna(subset=["val_loss"])
    acc_df = df.dropna(subset=["val_accuracy"])
    f1_df = df.dropna(subset=["val_f1"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training Metrics", fontsize=15)

    # Train loss
    ax = axes[0, 0]
    if not train_df.empty:
        ax.plot(train_df["step"], train_df["train_loss"], label="raw")
        ax.plot(train_df["step"], smooth(train_df["train_loss"]), linestyle="--", label="smooth")
    ax.set_title("Train Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid()
    ax.legend()

    # Validation Accuracy
    ax = axes[0, 1]
    if not acc_df.empty:
        ax.plot(acc_df["epoch"], acc_df["val_accuracy"], label="raw")
        ax.plot(acc_df["epoch"], smooth(acc_df["val_accuracy"]), linestyle="--", label="smooth")
    ax.set_title("Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid()
    ax.legend()

    # Validation Loss
    ax = axes[1, 0]
    if not val_df.empty:
        ax.plot(val_df["epoch"], val_df["val_loss"], label="raw")
        ax.plot(val_df["epoch"], smooth(val_df["val_loss"]), linestyle="--", label="smooth")
    ax.set_title("Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid()
    ax.legend()

    # Validation F1
    ax = axes[1, 1]
    if not f1_df.empty:
        ax.plot(f1_df["epoch"], f1_df["val_f1"], label="raw")
        ax.plot(f1_df["epoch"], smooth(f1_df["val_f1"]), linestyle="--", label="smooth")
    ax.set_title("Validation F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.grid()
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(CURVE_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved curves: {CURVE_PATH}")


# ========================= COMPUTE METRICS =========================
def compute_metrics(df):
    true_col, pred_col = detect_label_columns(df)

    y_true = df[true_col]
    y_pred = df[pred_col]

    # 准确率
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.4f} ({acc * 100:.2f}%)")

    with open(ACCURACY_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.6f} ({acc * 100:.2f}%)\n")
    print(f"Saved accuracy: {ACCURACY_TXT_PATH}")

    # 分类报告
    report = classification_report(y_true, y_pred, digits=4)
    print("\nClassification Report:")
    print(report)

    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved classification report: {CLASSIFICATION_REPORT_PATH}")

    return acc, report


# ========================= CONFUSION MATRIX =========================
def plot_cm(df):
    true_col, pred_col = detect_label_columns(df)

    y_true = df[true_col]
    y_pred = df[pred_col]

    # 如果你任务固定是三分类，可以直接固定顺序
    preferred_labels = ["negative", "neutral", "positive"]
    unique_labels = list(set(y_true.dropna().tolist()) | set(y_pred.dropna().tolist()))

    if all(label in unique_labels for label in preferred_labels):
        labels = preferred_labels
    else:
        labels = sorted(unique_labels)

    print("Detected labels:", labels)

    # Count matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(CM_COUNT_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    # Normalized matrix
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    cm_norm_df = pd.DataFrame(cm_norm, index=labels, columns=labels)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm_df, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(CM_NORM_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved confusion matrix counts: {CM_COUNT_PATH}")
    print(f"Saved normalized confusion matrix: {CM_NORM_PATH}")


# ========================= MAIN =========================
def main():
    print("========== Start Visualization ==========")
    print(f"BASE_OUTPUT_DIR: {BASE_OUTPUT_DIR}")
    print(f"CHECKPOINT_DIR: {CHECKPOINT_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    # 训练曲线
    metrics_df = load_metrics()
    print(f"Metrics shape: {metrics_df.shape}")
    plot_curves(metrics_df)

    # 测试集评估
    if os.path.exists(PREDICTIONS_CSV):
        pred_df = load_predictions()
        print(f"Predictions shape: {pred_df.shape}")

        compute_metrics(pred_df)
        plot_cm(pred_df)
    else:
        print(f"Warning: prediction file not found: {PREDICTIONS_CSV}")
        print("Skip accuracy / classification report / confusion matrix.")

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()