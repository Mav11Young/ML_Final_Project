import numpy as np

class ManualLogisticRegression:
    """
    手动实现的多分类逻辑回归 (Softmax Regression)
    使用梯度下降优化 Cross Entropy Loss
    """
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=64, l2_reg=0.01):
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.weights = None
        self.bias = None

    def _softmax(self, z):
        # 减去最大值防止指数爆炸
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y, num_classes):
        one_hot = np.zeros((y.size, num_classes))
        one_hot[np.arange(y.size), y.astype(int)] = 1
        return one_hot

    def fit(self, X, y):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))
        y_encoded = self._one_hot(y, num_classes)

        # 初始化参数
        self.weights = np.zeros((num_features, num_classes))
        self.bias = np.zeros((1, num_classes))

        for epoch in range(self.epochs):
            # 随机打乱数据
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y_encoded[indices]

            for i in range(0, num_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                # 前向传播
                z = np.dot(X_batch, self.weights) + self.bias
                probs = self._softmax(z)

                # 计算梯度
                dz = probs - y_batch
                dw = (1 / len(X_batch)) * np.dot(X_batch.T, dz) + self.l2_reg * self.weights
                db = (1 / len(X_batch)) * np.sum(dz, axis=0, keepdims=True)

                # 更新参数
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            if (epoch + 1) % 10 == 0:
                # 计算训练损失
                z_full = np.dot(X, self.weights) + self.bias
                probs_full = self._softmax(z_full)
                loss = -np.mean(np.sum(y_encoded * np.log(probs_full + 1e-15), axis=1))
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        probs = self._softmax(z)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

class ManualLinearSVM:
    """
    手动实现的线性支持向量机 (Linear SVM)
    使用 One-vs-Rest (OvR) 策略处理多分类
    使用 Hinge Loss 和 随机梯度下降 (SGD)
    """
    def __init__(self, learning_rate=0.001, epochs=50, lambda_param=0.01, batch_size=64):
        self.lr = learning_rate
        self.epochs = epochs
        self.lambda_param = lambda_param
        self.batch_size = batch_size
        self.models = [] # 存储 100 个类别的 (weights, bias)

    def fit(self, X, y):
        num_samples, num_features = X.shape
        classes = np.unique(y)
        num_classes = len(classes)
        self.models = []

        print(f"开始训练 Linear SVM (OvR), 共 {num_classes} 个分类器...")
        
        for idx, c in enumerate(classes):
            # 将多分类转为二分类: 当前类为 1, 其他类为 -1
            y_binary = np.where(y == c, 1, -1)
            
            w = np.zeros(num_features)
            b = 0
            
            for epoch in range(self.epochs):
                # 随机打乱
                indices = np.random.permutation(num_samples)
                X_shuff = X[indices]
                y_shuff = y_binary[indices]
                
                for i in range(0, num_samples, self.batch_size):
                    X_batch = X_shuff[i:i+self.batch_size]
                    y_batch = y_shuff[i:i+self.batch_size]
                    
                    # 向量化计算 Hinge Loss 条件
                    condition = y_batch * (np.dot(X_batch, w) + b) < 1
                    
                    # 计算梯度
                    dw = 2 * self.lambda_param * w
                    db = 0
                    
                    # 只有满足条件的样本贡献梯度
                    if np.any(condition):
                        dw -= np.dot(y_batch[condition], X_batch[condition]) / len(y_batch)
                        db -= np.sum(y_batch[condition]) / len(y_batch)
                    
                    # 更新参数
                    w -= self.lr * dw
                    b -= self.lr * db
            
            self.models.append((w, b))
            if (idx + 1) % 20 == 0:
                print(f"已完成 {idx + 1}/{num_classes} 个分类器的训练")

    def predict(self, X):
        # 计算每个分类器的得分 (decision function)
        scores = []
        for w, b in self.models:
            score = np.dot(X, w) + b
            scores.append(score)
        
        # scores 形状为 (num_classes, num_samples), 转置为 (num_samples, num_classes)
        scores = np.array(scores).T
        return np.argmax(scores, axis=1)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

