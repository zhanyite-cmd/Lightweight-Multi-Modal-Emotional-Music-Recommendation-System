import os
import time
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from multiprocessing import freeze_support
from datasets import Dataset

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def main():
    # ========================== Config ==========================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    # 这里填“模型文件夹路径”，不是某个 json 文件路径
    LOCAL_MODEL_PATH = r"D:\树莓派\face_detection-检测+分类+大模型\face_detection-检测+分类+大模型\face_detect\runs\Goemotions\distilbert\distilbert-base-uncased"
    DATASET_DIR = r"D:\树莓派\face_detection-检测+分类+大模型\face_detection-检测+分类+大模型\face_detect\runs\Goemotions\data"

    MAX_LENGTH = 48
    BATCH_SIZE = 64
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    RANDOM_SEED = 42
    INFERENCE_RUNS = 50

    OUTPUT_DIR = r"./distilbert_goemotions_outputs"
    BEST_MODEL_DIR = r"./distilbert_goemotions_best_model"
    LOG_DIR = r"./distilbert_goemotions_logs"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, "trainer_metrics.csv")
    CURVE_FIG_PATH = os.path.join(OUTPUT_DIR, "training_curves.png")
    CM_COUNT_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix_counts.png")
    CM_NORM_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix_normalized.png")
    REPORT_TXT_PATH = os.path.join(OUTPUT_DIR, "classification_report.txt")
    TEST_PRED_CSV_PATH = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    TRAINER_STATE_JSON_PATH = os.path.join(OUTPUT_DIR, "trainer_state_log_history.json")
    SPEED_CSV_PATH = os.path.join(OUTPUT_DIR, "inference_speed.csv")

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Local model path: {LOCAL_MODEL_PATH}")
    print(f"Dataset path: {DATASET_DIR}")
    print("=" * 60)

    # ========================== Emotion Mapping ==========================
    EMOTION_MAPPING = {
        "neutral": "neutral",

        "joy": "positive",
        "gratitude": "positive",
        "love": "positive",
        "pride": "positive",
        "excitement": "positive",
        "amusement": "positive",
        "relief": "positive",
        "approval": "positive",
        "admiration": "positive",
        "desire": "positive",
        "optimism": "positive",
        "curiosity": "positive",
        "surprise": "positive",

        "anger": "negative",
        "frustration": "negative",
        "annoyance": "negative",
        "sadness": "negative",
        "fear": "negative",
        "disgust": "negative",
        "disapproval": "negative",
        "remorse": "negative",
        "embarrassment": "negative",
        "guilt": "negative",
        "disappointment": "negative",
        "grief": "negative",
        "nervousness": "negative",
        "horror": "negative",
        "shame": "negative",
        "confusion": "negative"
    }

    # ========================== Parse emotions.txt ==========================
    def parse_emotions_txt(data_dir):
        id2emotion = {}
        txt_path = os.path.join(data_dir, "emotions.txt")

        if not os.path.exists(txt_path):
            print(f"⚠️ 未找到 {txt_path}，将使用最小映射")
            return {0: "neutral"}

        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t") if "\t" in line else line.split()
                    if len(parts) >= 2:
                        emotion_id = int(parts[0])
                        emotion_name = parts[1].lower()
                        id2emotion[emotion_id] = emotion_name
        except Exception as e:
            print(f"⚠️ 解析 emotions.txt 失败：{e}")
            id2emotion = {0: "neutral"}

        print(f"✅ emotions.txt 解析完成，共 {len(id2emotion)} 个标签")
        return id2emotion

    id2emotion = parse_emotions_txt(DATASET_DIR)

    # ========================== Load Dataset ==========================
    def load_goemotions(data_dir):
        col_names = ["text", "emotion_ids", "extra_id"]

        train_path = os.path.join(data_dir, "train.tsv")
        val_path = os.path.join(data_dir, "dev.tsv")
        test_path = os.path.join(data_dir, "test.tsv")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"未找到训练集：{train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"未找到验证集：{val_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"未找到测试集：{test_path}")

        df_train = pd.read_csv(train_path, sep="\t", names=col_names, header=None)
        df_val = pd.read_csv(val_path, sep="\t", names=col_names, header=None)
        df_test = pd.read_csv(test_path, sep="\t", names=col_names, header=None)

        def get_coarse_label(emotion_id_str):
            try:
                s = str(emotion_id_str).strip().replace("[", "").replace("]", "").replace(" ", "")
                if not s:
                    return "neutral"

                # 保持和你之前逻辑一致：只取第一个标签
                main_id = int(s.split(",")[0])
                fine_emotion = id2emotion.get(main_id, "neutral")
                return EMOTION_MAPPING.get(fine_emotion, "neutral")
            except Exception:
                return "neutral"

        for df in [df_train, df_val, df_test]:
            df.dropna(subset=["text"], inplace=True)
            df.reset_index(drop=True, inplace=True)
            df["text"] = df["text"].astype(str)
            df["coarse_label"] = df["emotion_ids"].apply(get_coarse_label)

        print("\n📊 数据加载完成：")
        print(f"- Training: {len(df_train)}")
        print(f"- Validation: {len(df_val)}")
        print(f"- Test: {len(df_test)}")
        print(f"📌 Train distribution: {df_train['coarse_label'].value_counts().to_dict()}")
        print(f"📌 Val distribution: {df_val['coarse_label'].value_counts().to_dict()}")
        print(f"📌 Test distribution: {df_test['coarse_label'].value_counts().to_dict()}")

        return df_train, df_val, df_test

    df_train, df_val, df_test = load_goemotions(DATASET_DIR)

    # ========================== Label Encoding ==========================
    label_encoder = LabelEncoder()
    target_labels = ["positive", "negative", "neutral"]
    label_encoder.fit(target_labels)

    def encode_labels(df):
        neutral_id = label_encoder.transform(["neutral"])[0]
        df["labels"] = df["coarse_label"].apply(
            lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else neutral_id
        )
        return df

    df_train = encode_labels(df_train)
    df_val = encode_labels(df_val)
    df_test = encode_labels(df_test)

    print("\n📌 标签编码映射：")
    for idx, cls_name in enumerate(label_encoder.classes_):
        print(f"{idx} -> {cls_name}")

    train_ds = Dataset.from_pandas(df_train[["text", "labels"]])
    val_ds = Dataset.from_pandas(df_val[["text", "labels"]])
    test_ds = Dataset.from_pandas(df_test[["text", "labels"]])

    # ========================== Load Tokenizer ==========================
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            LOCAL_MODEL_PATH,
            local_files_only=True
        )
        print(f"\n✅ 本地 DistilBERT tokenizer 加载成功：{LOCAL_MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"❌ 加载本地 tokenizer 失败：{e}")

    def tokenize_func(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH
        )

    tokenized_train = train_ds.map(tokenize_func, batched=True, batch_size=1000, desc="Tokenizing train")
    tokenized_val = val_ds.map(tokenize_func, batched=True, batch_size=1000, desc="Tokenizing val")
    tokenized_test = test_ds.map(tokenize_func, batched=True, batch_size=1000, desc="Tokenizing test")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ========================== Load Model ==========================
    try:
        model = DistilBertForSequenceClassification.from_pretrained(
            LOCAL_MODEL_PATH,
            num_labels=3,
            local_files_only=True,
            ignore_mismatched_sizes=True
        ).to(DEVICE)
        print(f"✅ 本地 DistilBERT model 加载成功：{LOCAL_MODEL_PATH}")
        print(f"📌 num_labels = {model.config.num_labels}")
    except Exception as e:
        raise RuntimeError(f"❌ 加载本地 DistilBERT model 失败：{e}")

    # ========================== Metrics ==========================
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        return {
            "accuracy": accuracy,
            "f1": f1
        }

    # ========================== Training Arguments ==========================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=50,

        logging_dir=LOG_DIR,
        logging_steps=100,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,

        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,

        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=RANDOM_SEED
    )

    # ========================== Trainer ==========================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    # ========================== Train ==========================
    print("\n🚀 开始训练 DistilBERT...")
    start_train = time.time()
    trainer.train()
    train_time = time.time() - start_train
    print(f"\n✅ 训练完成，总耗时：{train_time:.2f} 秒")

    # ========================== Export trainer state ==========================
    with open(TRAINER_STATE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    print(f"✅ trainer.state.log_history 已保存：{TRAINER_STATE_JSON_PATH}")

    # ========================== Test Evaluation ==========================
    print("\n📝 开始测试集评估...")
    test_preds = trainer.predict(tokenized_test)
    pred_labels = np.argmax(test_preds.predictions, axis=-1)
    true_labels = test_preds.label_ids

    test_accuracy = accuracy_score(true_labels, pred_labels)
    test_f1 = f1_score(true_labels, pred_labels, average="weighted")

    print("\n" + "=" * 60)
    print(f"📈 Test Accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")
    print(f"📈 Test Weighted F1: {test_f1:.4f}")
    print("=" * 60)

    report_text = classification_report(
        true_labels,
        pred_labels,
        target_names=label_encoder.classes_,
        digits=4
    )
    print("\n📋 分类报告：")
    print(report_text)

    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    # ========================== Save Predictions ==========================
    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    pred_df = df_test.copy()
    pred_df["true_id"] = true_labels
    pred_df["pred_id"] = pred_labels
    pred_df["true_label"] = pred_df["true_id"].map(id2label)
    pred_df["pred_label"] = pred_df["pred_id"].map(id2label)
    pred_df["correct"] = pred_df["true_id"] == pred_df["pred_id"]
    pred_df.to_csv(TEST_PRED_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ 测试集预测结果已保存：{TEST_PRED_CSV_PATH}")

    # ========================== Inference Speed ==========================
    def test_inference_speed(texts, model, tokenizer, n_runs=INFERENCE_RUNS):
        model.eval()
        times = []

        if len(texts) == 0:
            return [0.0]

        warmup_text = texts[0]
        inputs = tokenizer(
            warmup_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        ).to(DEVICE)

        with torch.no_grad():
            _ = model(**inputs)

        print("\n⚡ 开始测试推理速度...")
        for text in tqdm(texts[:n_runs], desc="Inference Speed Test", ncols=80):
            start = time.time()
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH
            ).to(DEVICE)
            with torch.no_grad():
                _ = model(**inputs)
            times.append((time.time() - start) * 1000)

        return times

    inference_times = test_inference_speed(df_test["text"].tolist(), model, tokenizer)
    avg_infer_ms = float(np.mean(inference_times))

    pd.DataFrame([{
        "Avg Inference Time (ms)": avg_infer_ms,
        "Total Training Time (s)": float(train_time),
        "Test Accuracy": float(test_accuracy),
        "Test Weighted F1": float(test_f1)
    }]).to_csv(SPEED_CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"⚡ 平均单条推理时间：{avg_infer_ms:.2f} ms")

    # ========================== Export Metrics CSV ==========================
    rows = []
    for log in trainer.state.log_history:
        rows.append({
            "step": log.get("step", np.nan),
            "epoch": log.get("epoch", np.nan),
            "train_loss": log.get("loss", np.nan),
            "val_loss": log.get("eval_loss", np.nan),
            "val_accuracy": log.get("eval_accuracy", np.nan),
            "val_f1": log.get("eval_f1", np.nan),
            "learning_rate": log.get("learning_rate", np.nan)
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ 训练指标 CSV 已保存：{METRICS_CSV_PATH}")

    # ========================== Plot Curves ==========================
    def smooth_curve(values, factor=0.85):
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

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.unicode_minus"] = False

    train_df = metrics_df.dropna(subset=["train_loss"]).copy()
    val_df = metrics_df.dropna(subset=["val_loss"]).copy()
    acc_df = metrics_df.dropna(subset=["val_accuracy"]).copy()
    f1_df = metrics_df.dropna(subset=["val_f1"]).copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training and Validation Metrics", fontsize=15, fontweight="bold")

    # Train Loss
    ax = axes[0, 0]
    if len(train_df) > 0:
        x = train_df["step"].values
        y = train_df["train_loss"].values.astype(float)
        ax.plot(x, y, marker="o", linewidth=2.0, label="Raw")
        ax.plot(x, smooth_curve(y), linestyle="--", linewidth=2.0, label="Smoothed")
    ax.set_title("Train Loss")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Validation Accuracy
    ax = axes[0, 1]
    if len(acc_df) > 0:
        x = acc_df["epoch"].values
        y = acc_df["val_accuracy"].values.astype(float)
        ax.plot(x, y, marker="o", linewidth=2.0, label="Raw")
        ax.plot(x, smooth_curve(y, factor=0.6), linestyle="--", linewidth=2.0, label="Smoothed")
    ax.set_title("Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Validation Loss
    ax = axes[1, 0]
    if len(val_df) > 0:
        x = val_df["epoch"].values
        y = val_df["val_loss"].values.astype(float)
        ax.plot(x, y, marker="o", linewidth=2.0, label="Raw")
        ax.plot(x, smooth_curve(y, factor=0.6), linestyle="--", linewidth=2.0, label="Smoothed")
    ax.set_title("Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Validation F1
    ax = axes[1, 1]
    if len(f1_df) > 0:
        x = f1_df["epoch"].values
        y = f1_df["val_f1"].values.astype(float)
        ax.plot(x, y, marker="o", linewidth=2.0, label="Raw")
        ax.plot(x, smooth_curve(y, factor=0.6), linestyle="--", linewidth=2.0, label="Smoothed")
    ax.set_title("Validation F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(CURVE_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ 训练曲线图已保存：{CURVE_FIG_PATH}")

    # ========================== Confusion Matrix ==========================
    cm_count = confusion_matrix(true_labels, pred_labels)
    cm_count_df = pd.DataFrame(cm_count, index=label_encoder.classes_, columns=label_encoder.classes_)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_count_df, annot=True, fmt="d", cmap="Blues", linewidths=0.5, cbar=True, square=True)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(CM_COUNT_PATH, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ 计数混淆矩阵已保存：{CM_COUNT_PATH}")

    cm_norm = confusion_matrix(true_labels, pred_labels, normalize="true")
    cm_norm_df = pd.DataFrame(cm_norm, index=label_encoder.classes_, columns=label_encoder.classes_)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm_df, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5, cbar=True, square=True, vmin=0, vmax=1)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(CM_NORM_PATH, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ 归一化混淆矩阵已保存：{CM_NORM_PATH}")

    # ========================== Save Model ==========================
    trainer.save_model(BEST_MODEL_DIR)
    tokenizer.save_pretrained(BEST_MODEL_DIR)

    print("\n📁 已保存文件：")
    print(f"- Best model: {BEST_MODEL_DIR}")
    print(f"- Metrics CSV: {METRICS_CSV_PATH}")
    print(f"- Training curves: {CURVE_FIG_PATH}")
    print(f"- Confusion matrix count: {CM_COUNT_PATH}")
    print(f"- Confusion matrix normalized: {CM_NORM_PATH}")
    print(f"- Classification report: {REPORT_TXT_PATH}")
    print(f"- Test predictions: {TEST_PRED_CSV_PATH}")
    print(f"- Trainer state json: {TRAINER_STATE_JSON_PATH}")
    print(f"- Inference speed csv: {SPEED_CSV_PATH}")
    print("\n🎉 DistilBERT + GoEmotions 训练完成！")


if __name__ == "__main__":
    freeze_support()
    main()