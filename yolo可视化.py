import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ultralytics import YOLO

# -------------------------- 基础配置 --------------------------
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 中文显示
plt.rcParams["axes.unicode_minus"] = False  # 负号显示
plt.rcParams['figure.dpi'] = 100  # 图片清晰度

# ======== 你的实际路径（无需修改）========
csv_path = r"D:\树莓派\face_detection-检测+分类+大模型\face_detection-检测+分类+大模型\训练结果-7classes\train\results.csv"
model_path = r"D:\树莓派\face_detection-检测+分类+大模型\face_detection-检测+分类+大模型\yolo11s-emotion.pt"
data_yaml_path = r"D:\树莓派\face_detection-检测+分类+大模型\face_detection-检测+分类+大模型\data.yaml"


# -------------------------- 图1：训练指标趋势图（4个子图） --------------------------
def plot_training_metrics(csv_path):
    # 读取数据
    df = pd.read_csv(csv_path, skip_blank_lines=True)

    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Metrics', fontsize=16)

    # 1. train/loss
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['train/loss'], 'b-', label='results', linewidth=1)
    # 5轮滑动平均平滑曲线
    smooth_train_loss = df['train/loss'].rolling(window=5, center=True).mean()
    ax1.plot(df['epoch'], smooth_train_loss, 'orange', linestyle=':', label='smooth', linewidth=1)
    ax1.set_title('train/loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # 2. metrics/accuracy_top1
    ax2 = axes[0, 1]
    ax2.plot(df['epoch'], df['metrics/accuracy_top1'], 'b-', label='results', linewidth=1)
    smooth_top1 = df['metrics/accuracy_top1'].rolling(window=5, center=True).mean()
    ax2.plot(df['epoch'], smooth_top1, 'orange', linestyle=':', label='smooth', linewidth=1)
    ax2.set_title('metrics/accuracy_top1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # 3. val/loss
    ax3 = axes[1, 0]
    ax3.plot(df['epoch'], df['val/loss'], 'b-', label='results', linewidth=1)
    smooth_val_loss = df['val/loss'].rolling(window=5, center=True).mean()
    ax3.plot(df['epoch'], smooth_val_loss, 'orange', linestyle=':', label='smooth', linewidth=1)
    ax3.set_title('val/loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()

    # 4. metrics/accuracy_top5
    ax4 = axes[1, 1]
    ax4.plot(df['epoch'], df['metrics/accuracy_top5'], 'b-', label='results', linewidth=1)
    smooth_top5 = df['metrics/accuracy_top5'].rolling(window=5, center=True).mean()
    ax4.plot(df['epoch'], smooth_top5, 'orange', linestyle=':', label='smooth', linewidth=1)
    ax4.set_title('metrics/accuracy_top5')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()

    # 保存图片
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()


# -------------------------- 图2：混淆矩阵图（修复格式报错） --------------------------
def plot_confusion_matrix(model_path, data_yaml_path):
    # 加载表情分类模型
    model = YOLO(model_path)
    # 评估验证集并生成混淆矩阵（禁用自动归一化，确保是原始计数）
    results = model.val(data=data_yaml_path, save_json=True, conf=0.5)

    # 获取混淆矩阵并处理数据类型（核心修复：兼容整数/浮点数）
    conf_matrix = results.confusion_matrix.matrix
    # 转换为整数（混淆矩阵本质是样本计数，浮点数是因为归一化，转整数还原真实计数）
    conf_matrix = np.round(conf_matrix).astype(int)
    # 7类表情名称
    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']

    # 绘制混淆矩阵热力图（修复fmt参数：d适配整数）
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',  # 整数格式（已转int，不再报错）
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count', 'shrink': 0.8},
        annot_kws={'size': 11}
    )

    # 标签和标题设置
    plt.xlabel('True', fontsize=12)
    plt.ylabel('Predicted', fontsize=12, rotation=0)
    plt.title('Confusion Matrix', fontsize=16, pad=20)

    # 保存图片
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


# -------------------------- 执行生成 --------------------------
if __name__ == '__main__':
    # 生成训练指标趋势图
    plot_training_metrics(csv_path)

    # 生成混淆矩阵图
    plot_confusion_matrix(model_path, data_yaml_path)

    print("✅ 两张图已成功生成：")
    print("1. training_metrics.png（训练指标趋势）")
    print("2. confusion_matrix.png（混淆矩阵）")