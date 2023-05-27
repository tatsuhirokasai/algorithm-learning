from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# データセットの読み込み
digits = load_digits()

# データを訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# モデルの初期化と訓練
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1)
mlp.fit(X_train, y_train)

# テストデータに対する予測
predictions = mlp.predict(X_test)

# 予測の精度を計算
print("Accuracy: ", accuracy_score(y_test, predictions))
