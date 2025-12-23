import pandas as pd
import numpy as np
import os


def load_data(file_path):
    """加载数据，返回特征矩阵 X 和标签向量 y"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    data = pd.read_csv(file_path, header=None)

    # 假设最后一列是标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    return X, y


def load_test_data(file_path):
    """加载测试数据，返回特征矩阵 X"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    data = pd.read_csv(file_path, header=None)
    return data.values


def standardize(X_train, X_test=None):
    """标准化数据 (Z-score normalization)"""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # 防止除以 0
    std[std == 0] = 1.0

    X_train_std = (X_train - mean) / std

    if X_test is not None:
        X_test_std = (X_test - mean) / std
        return X_train_std, X_test_std

    return X_train_std, mean, std


def stratified_train_test_split(X, y, test_size=0.2, random_state=42):
    """手动实现分层抽样划分训练集/验证集"""
    np.random.seed(random_state)

    train_indices = []
    val_indices = []

    classes = np.unique(y)
    for c in classes:
        # 获取属于当前类别的所有索引
        idx = np.where(y == c)[0]
        # 随机打乱当前类别的索引
        np.random.shuffle(idx)

        # 计算划分点
        split_idx = int(len(idx) * (1 - test_size))

        train_indices.extend(idx[:split_idx])
        val_indices.extend(idx[split_idx:])

    # 将索引转换为 numpy 数组并再次随机打乱整体顺序
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]


def k_fold_stratified(X, y, k=5, random_state=42):
    """手动实现分层 K 折交叉验证索引生成"""
    np.random.seed(random_state)

    # 初始化 k 个 fold 的索引列表
    folds = [[] for _ in range(k)]

    classes = np.unique(y)
    for c in classes:
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)

        # 将当前类别的索引均匀分配到 k 个 fold 中
        for i, index_val in enumerate(idx):
            folds[i % k].append(index_val)

    # 结果生成器：每次返回训练索引和验证索引
    for i in range(k):
        val_idx = np.array(folds[i])
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        # 打乱索引顺序
        np.random.shuffle(val_idx)
        np.random.shuffle(train_idx)

        yield train_idx, val_idx
