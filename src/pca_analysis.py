import numpy as np
import matplotlib.pyplot as plt
import os
from utils import load_data, standardize


class ManualPCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None

    def fit(self, X):
        # 1. 中心化数据
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. 使用 SVD 分解 (X = U S V^T)
        # 协方差矩阵 C = (X^T X) / (n-1)
        # 在 SVD 中，V 的列就是主成分
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # 3. 计算解释方差比
        explained_variance = (S**2) / (len(X) - 1)
        self.explained_variance_ratio = explained_variance / np.sum(explained_variance)

        # 4. 保留前 n_components 个成分
        if self.n_components is not None:
            self.components = Vt[: self.n_components].T
        else:
            self.components = Vt.T

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def run_pca_analysis():
    print(">>> 开始 PCA 降维分析...")
    X, y = load_data("../data/train.csv")
    X_std, _, _ = standardize(X)

    pca = ManualPCA()
    pca.fit(X_std)

    # 计算累计贡献率
    cumulative_variance = np.cumsum(pca.explained_variance_ratio)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        marker="o",
        linestyle="--",
    )
    plt.axhline(y=0.95, color="r", linestyle="-")
    plt.text(0, 0.96, "95% Explained Variance", color="r", fontsize=12)
    plt.title("Cumulative Explained Variance by PCA Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)

    figures_path = "../report/figures/"
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)
    plt.savefig(os.path.join(figures_path, "pca_variance.png"))
    print("PCA 累计贡献率图已保存。")

    # 找到保留 95% 方差所需的组件数
    n_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"保留 95% 方差所需的组件数: {n_95}")


if __name__ == "__main__":
    run_pca_analysis()
