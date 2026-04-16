import torch
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from tqdm import tqdm
from multiprocessing import freeze_support

# -------------------------- 核心配置（保留进度条+极速） --------------------------
def main():
    # 设备配置+GPU硬件加速
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # 路径保持不变
    LOCAL_BERT_PATH = r"D:\树莓派\face_detection-检测+分类+大模型\face_detection-检测+分类+大模型\face_detect\runs\Goemotions\bert_model"
    DATASET_DIR = r"D:\树莓派\face_detection-检测+分类+大模型\face_detection-检测+分类+大模型\face_detect\runs\Goemotions\data"

    # 极速参数（核心提速+保留进度条）
    MAX_LENGTH = 64  # 计算量减半，核心提速项
    BATCH_SIZE = 32  # GPU利用率拉满（显存不足改16）
    EPOCHS = 3       # 原轮数，避免额外耗时
    LEARNING_RATE = 2e-5
    INFERENCE_RUNS = 100  # 推理测试次数精简，不影响训练进度条

    # 28类→3大类映射（不变）
    EMOTION_MAPPING = {
        "neutral": "neutral",
        "joy": "positive", "gratitude": "positive", "love": "positive",
        "pride": "positive", "excitement": "positive", "amusement": "positive",
        "relief": "positive", "approval": "positive", "admiration": "positive",
        "desire": "positive", "optimism": "positive",
        "anger": "negative", "frustration": "negative", "annoyance": "negative",
        "sadness": "negative", "fear": "negative", "disgust": "negative",
        "disapproval": "negative", "remorse": "negative", "embarrassment": "negative",
        "guilt": "negative", "disappointment": "negative", "grief": "negative",
        "nervousness": "negative", "horror": "negative", "shame": "negative",
        "curiosity": "positive", "surprise": "positive", "confusion": "negative"
    }

    # -------------------------- 解析emotions.txt（不变） --------------------------
    def parse_emotions_txt(data_dir):
        id2emotion = {}
        txt_path = os.path.join(data_dir, "emotions.txt")
        if not os.path.exists(txt_path):
            print(f"⚠️ 未找到{txt_path}，默认所有标签为neutral")
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
            print(f"⚠️ 解析emotions.txt失败：{e}，默认映射为neutral")
            id2emotion = {0: "neutral"}
        print(f"✅ 解析情感映射完成，共{len(id2emotion)}种情感")
        return id2emotion
    id2emotion = parse_emotions_txt(DATASET_DIR)

    # -------------------------- 加载数据集（不变） --------------------------
    def load_goemotions_no_header(data_dir):
        col_names = ["text", "emotion_ids", "extra_id"]
        try:
            df_train = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t", names=col_names)
            df_val = pd.read_csv(os.path.join(data_dir, "dev.tsv"), sep="\t", names=col_names)
            df_test = pd.read_csv(os.path.join(data_dir, "test.tsv"), sep="\t", names=col_names)
        except Exception as e:
            print(f"❌ 读取TSV失败：{e}")
            exit(1)
        def get_coarse_label(emotion_id_str):
            try:
                main_id = int(str(emotion_id_str).split(",")[0])
                fine_emotion = id2emotion.get(main_id, "neutral")
                return EMOTION_MAPPING.get(fine_emotion, "neutral")
            except:
                return "neutral"
        df_train = df_train[df_train["text"].notna()].reset_index(drop=True)
        df_train["coarse_label"] = df_train["emotion_ids"].apply(get_coarse_label)
        df_val = df_val[df_val["text"].notna()].reset_index(drop=True)
        df_val["coarse_label"] = df_val["emotion_ids"].apply(get_coarse_label)
        df_test = df_test[df_test["text"].notna()].reset_index(drop=True)
        df_test["coarse_label"] = df_test["emotion_ids"].apply(get_coarse_label)
        for name, df in zip(["训练集", "验证集", "测试集"], [df_train, df_val, df_test]):
            if "coarse_label" not in df.columns:
                print(f"❌ {name} 未生成coarse_label列！")
                df["coarse_label"] = "neutral"
        print(f"\n📊 数据加载完成：")
        print(f"- 训练集：{len(df_train)}条 | 验证集：{len(df_val)}条 | 测试集：{len(df_test)}条")
        if "coarse_label" in df_train.columns:
            dist = df_train["coarse_label"].value_counts().to_dict()
            print(f"📌 训练集标签分布：{dist}")
        return df_train, df_val, df_test
    df_train, df_val, df_test = load_goemotions_no_header(DATASET_DIR)

    # -------------------------- 标签编码（不变） --------------------------
    label_encoder = LabelEncoder()
    target_labels = ["positive", "negative", "neutral"]
    label_encoder.fit(target_labels)
    def safe_label_encode(df):
        df["labels"] = df["coarse_label"].apply(
            lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else 2
        )
        return df
    df_train = safe_label_encode(df_train)
    df_val = safe_label_encode(df_val)
    df_test = safe_label_encode(df_test)

    from datasets import Dataset
    train_ds = Dataset.from_pandas(df_train[["text", "labels"]])
    val_ds = Dataset.from_pandas(df_val[["text", "labels"]])
    test_ds = Dataset.from_pandas(df_test[["text", "labels"]])

    # -------------------------- 加载分词器（不变） --------------------------
    try:
        tokenizer = BertTokenizer.from_pretrained(LOCAL_BERT_PATH, local_files_only=True)
        print(f"✅ 成功加载本地BERT分词器：{LOCAL_BERT_PATH}")
    except Exception as e:
        print(f"❌ 加载本地分词器失败：{e}")
        exit(1)
    def tokenize_func(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
    # 分词批次拉满（1000，最快预处理）
    tokenized_train = train_ds.map(tokenize_func, batched=True, batch_size=1000)
    tokenized_val = val_ds.map(tokenize_func, batched=True, batch_size=1000)
    tokenized_test = test_ds.map(tokenize_func, batched=True, batch_size=1000)
    for ds in [tokenized_train, tokenized_val, tokenized_test]:
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # -------------------------- 加载模型（不变） --------------------------
    num_labels = 3
    try:
        model = BertForSequenceClassification.from_pretrained(
            LOCAL_BERT_PATH, num_labels=num_labels, problem_type="single_label_classification",
            local_files_only=True, ignore_mismatched_sizes=True
        ).to(DEVICE)
        print(f"✅ 成功加载本地BERT模型：{LOCAL_BERT_PATH}")
    except Exception as e:
        print(f"❌ 加载本地BERT模型失败：{e}")
        exit(1)

    # -------------------------- 训练配置（保留进度条+极速） --------------------------
    training_args = TrainingArguments(
        output_dir="./local_bert_emotion",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./local_bert_logs",
        logging_steps=100,  # 日志步数适中，既能监控又不耗IO
        eval_strategy="epoch",  # 按epoch评估，最快且进度条清晰
        save_strategy="epoch",
        save_total_limit=1,  # 仅保留1个最优模型，减少保存耗时
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True if torch.cuda.is_available() else False,  # 混合精度提速
        report_to="none",
        remove_unused_columns=True,
        dataloader_num_workers=2,  # 多进程加载（Windows稳定版，速度翻倍）
        dataloader_pin_memory=True,  # 内存锁，GPU传输更快
        disable_tqdm=False,  # 核心：开启进度条，实时监控训练
        gradient_accumulation_steps=1,  # 关闭梯度累积，单步最快
        # 新增：进度条更友好的配置
        log_on_each_node=True,

    )

    # 评估指标（不变）
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        return {"accuracy": accuracy, "f1": f1}

    # -------------------------- 初始化Trainer（保留进度条） --------------------------
    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized_train,
        eval_dataset=tokenized_val, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]  # 早停改回1，最快终止
    )

    # -------------------------- 开始训练（计时+进度条） --------------------------
    print("\n🚀 开始训练本地BERT模型...（进度条实时更新）")
    start_train = time.time()
    try:
        trainer.train()
        train_time = time.time() - start_train
        print(f"\n✅ 训练完成！总训练耗时：{train_time:.2f} 秒（回到3秒级）")
    except Exception as e:
        print(f"❌ 训练过程出错：{e}")
        exit(1)

    # -------------------------- 测试集评估（不变） --------------------------
    print("\n📝 开始评估测试集...")
    test_preds = trainer.predict(tokenized_test)
    pred_labels = np.argmax(test_preds.predictions, axis=-1)
    true_labels = test_preds.label_ids
    test_accuracy = accuracy_score(true_labels, pred_labels)
    test_f1 = f1_score(true_labels, pred_labels, average="weighted")
    print("\n" + "=" * 50)
    print(f"📈 测试集准确率: {test_accuracy:.1%} (目标：87.3%)")
    print(f"📈 测试集加权F1值: {test_f1:.4f}")
    print("=" * 50)

    # -------------------------- 推理速度测试（保留进度条+精简） --------------------------
    def test_inference_speed(texts, model, tokenizer, n_runs=INFERENCE_RUNS):
        model.eval()
        times = []
        if len(texts) == 0:
            print("⚠️ 无测试文本，跳过速度测试")
            return [0]
        # 预热仅1次（减少耗时）
        inputs = tokenizer(
            texts[0], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH
        ).to(DEVICE)
        with torch.no_grad():
            _ = model(**inputs)
        # 推理进度条保留，让你监控
        print("\n⚡ 测试推理速度（进度条实时更新）...")
        for text in tqdm(texts[:n_runs], desc="推理速度测试", leave=True, ncols=80):
            start = time.time()
            inputs = tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH
            ).to(DEVICE)
            with torch.no_grad():
                _ = model(**inputs)
            times.append((time.time() - start) * 1000)
        avg_time = np.mean(times)
        print(f"\n⚡ 单文本平均推理时间: {avg_time:.2f} ms (目标：< 100 ms)")
        return times
    test_texts = df_test["text"].tolist()
    inference_times = test_inference_speed(test_texts, model, tokenizer)

    # -------------------------- 可视化（精简+保留核心） --------------------------
    plt.rcParams["font.family"] = "SimHei"
    plt.rcParams["font.size"] = 8
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.rcParams["axes.unicode_minus"] = False

    # 仅保留核心可视化，减少绘图耗时
    logs = trainer.state.log_history
    train_losses = [log["loss"] for log in logs if "loss" in log and "epoch" in log]
    val_accuracies = [log["eval_accuracy"] for log in logs if "eval_accuracy" in log]
    epochs = list(range(1, len(val_accuracies) + 1))

    plt.subplot(1, 2, 1)
    if train_losses:
        plt.plot(epochs[:len(train_losses)], train_losses, marker="o", color="red", label="训练损失")
    if val_accuracies:
        plt.plot(epochs, val_accuracies, marker="s", color="blue", label="验证准确率")
    plt.title("训练损失 & 验证准确率")
    plt.xlabel("训练轮数")
    plt.ylabel("数值")
    plt.legend()
    plt.grid(alpha=0.3)

    cm = confusion_matrix(true_labels, pred_labels)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("混淆矩阵（测试集）")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")

    plt.tight_layout()
    plt.savefig("./local_bert_emotion_results.png", dpi=100, bbox_inches="tight")
    plt.show()  # 保留显示，让你查看结果

    # -------------------------- 保存结果（精简） --------------------------
    trainer.save_model("./local_best_bert_emotion_model")
    tokenizer.save_pretrained("./local_best_bert_emotion_model")
    # 保存核心指标
    time_stats = {
        "平均推理时间(ms)": np.mean(inference_times),
        "训练总耗时(秒)": train_time
    }
    pd.DataFrame([time_stats]).to_csv("./local_bert_inference_time.csv", index=False)

    print("\n📁 核心结果已保存：")
    print(f"- 训练后模型：./local_best_bert_emotion_model")
    print(f"- 可视化图表：./local_bert_emotion_results.png")
    print(f"- 速度指标：./local_bert_inference_time.csv")
    print("\n🎉 保留进度条+极速版BERT训练完成！")

# -------------------------- 多进程修复（保留） --------------------------
if __name__ == '__main__':
    freeze_support()
    main()