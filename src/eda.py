import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import load_data
import seaborn as sns

# 路径设置
TRAIN_DATA_PATH = "../data/train.csv"
FIGURES_PATH = "../report/figures/"


def perform_eda():
    print(">>> 开始探索性数据分析 (EDA)...")
    X, y = load_data(TRAIN_DATA_PATH)

    print(f"数据形状: {X.shape}")
    print(f"类别数量: {len(np.unique(y))}")

    # 1. 类别分布分析
    plt.figure(figsize=(15, 6))
    sns.countplot(x=y)

    plt.title("Class Distribution")
    plt.xlabel("Class ID")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(FIGURES_PATH, "class_distribution.png"))
    print("1. 类别分布图已保存。")

    # 2. 数据稀疏性分析
    zero_ratio = np.sum(X == 0) / X.size
    print(f"2. 数据总稀疏度 (零元素占比): {zero_ratio:.2%}")

    # 3. 特征均值和标准差分布
    feature_means = np.mean(X, axis=0)
    feature_stds = np.std(X, axis=0)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(feature_means, bins=50, color="skyblue", edgecolor="black")
    plt.title("Distribution of Feature Means")

    plt.subplot(1, 2, 2)
    plt.hist(feature_stds, bins=50, color="salmon", edgecolor="black")
    plt.title("Distribution of Feature Stds")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "feature_stats_dist.png"))
    print("3. 特征统计分布图已保存。")

    # 4. 特征相关性热图 (仅取前20个特征示例)
    plt.figure(figsize=(10, 8))
    corr = pd.DataFrame(X[:, :20]).corr()
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap (First 20 Features)")
    plt.savefig(os.path.join(FIGURES_PATH, "feature_correlation.png"))
    print("4. 特征相关性图已保存。")


if __name__ == "__main__":
    if not os.path.exists(FIGURES_PATH):
        os.makedirs(FIGURES_PATH)
    perform_eda()
