import numpy as np
import pandas as pd
import os
from utils import load_data, load_test_data, standardize, stratified_train_test_split
from models_manual import ManualLogisticRegression
from pca_analysis import ManualPCA

# 路径配置
DATA_DIR = "../data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_PATH = "submission.csv"


def main():
    # 1. 加载与分割数据
    print("Step 1: Loading data...")
    X_all, y_all = load_data(TRAIN_PATH)
    X_test_final = load_test_data(TEST_PATH)

    # 划分训练集和验证集 (为了本地评估)
    X_train, X_val, y_train, y_val = stratified_train_test_split(
        X_all, y_all, test_size=0.2
    )

    # 2. 标准化
    print("Step 2: Standardizing data...")
    X_train_std, X_val_std = standardize(X_train, X_val)
    _, X_test_std = standardize(X_train, X_test_final)

    # 3. PCA 降维 (可选，根据分析决定保留多少组件)
    # 假设我们通过分析决定保留 200 维
    print("Step 3: Applying PCA...")
    pca = ManualPCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train_std)
    X_val_pca = pca.transform(X_val_std)
    X_test_pca = pca.transform(X_test_std)

    # 4. 训练手动实现的模型 (逻辑回归)
    print("Step 4: Training Manual Logistic Regression...")
    model = ManualLogisticRegression(learning_rate=0.1, epochs=100, batch_size=128)
    model.fit(X_train_pca, y_train)

    # 5. 模型评估
    train_acc = model.score(X_train_pca, y_train)
    val_acc = model.score(X_val_pca, y_val)
    print(f"\nManual Logistic Regression Results:")
    print(f"Training Accuracy: {train_acc:.4%}")
    print(f"Validation Accuracy: {val_acc:.4%}")

    # 6. 生成 Kaggle 提交文件
    print("\nStep 6: Generating submission...")
    test_preds = model.predict(X_test_pca)

    # 格式要求: ID, Category (具体根据 sample_submission.csv 调整)
    submission = pd.DataFrame(
        {"Id": np.arange(len(test_preds)), "Category": test_preds.astype(int)}
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
