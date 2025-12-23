# CS3339 机器学习期末项目 - 图像特征分类

本项目是上海交通大学 CS3339 机器学习课程的期末项目。任务是对高维（512维）图像特征进行 100 类的分类任务。

## 项目概况
- **任务**：100 类图像分类。
- **数据**：
    - 训练集：19,573 个样本。
    - 测试集：10,000 个样本。
    - 特征：每张图像提取好的 512 维特征向量。
- **核心要求**：实现至少 3 种统计学习方法（至少 2 种非深度学习），基于 Numpy 手动实现核心算法。

## 项目结构
```text
.
├── data/               # 数据集 (train.csv, test.csv, sample_submission.csv)
├── report/             # LaTeX 报告及图表
│   └── figures/        # EDA 和分析生成的图表
├── src/                # 源代码
│   ├── eda.py          # 探索性数据分析
│   ├── pca_analysis.py  # PCA 降维分析
│   ├── utils.py        # 工具函数 (数据加载, 标准化, 分层抽样, 交叉验证)
│   ├── models_manual.py # 手动实现的模型 (Logistic Regression, Linear SVM)
│   └── main.py         # 主程序 (模型训练与预测)
└── README.md           # 项目说明文档
```

## 当前进度
- [x] **探索性数据分析 (EDA)**：
    - 发现数据稀疏度为 15.73%。
    - 识别出轻微的类别不平衡问题。
    - 确定了 Z-score 标准化的必要性。
- [x] **降维分析 (PCA)**：
    - 手动实现 PCA，分析显示保留约 80-100 个主成分即可覆盖 95% 以上的方差。
- [x] **模型实现**：
    - [x] 多分类逻辑回归 (Softmax Regression) - 手动实现。
    - [x] 线性支持向量机 (Linear SVM, OvR 策略) - 手动实现。
    - [ ] 第三种模型（待实现）。
- [x] **评估方法**：
    - [x] 分层抽样 (Stratified Split)。
    - [x] 分层 K 折交叉验证 (Stratified K-Fold)。

## 如何运行
1. **安装依赖**：
   ```bash
   pip install numpy pandas matplotlib seaborn
   ```
2. **运行 EDA**：
   ```bash
   cd src && python eda.py
   ```
3. **运行 PCA 分析**：
   ```bash
   cd src && python pca_analysis.py
   ```
4. **运行主程序**（目前包含逻辑回归训练）：
   ```bash
   cd src && python main.py
   ```

