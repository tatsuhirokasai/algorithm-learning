from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# データをロード
digits = load_digits()
X = digits.data
y = digits.target

# データを標準化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの作成と訓練
# solver(最適化のアルゴリズム): 'sgd'（確率的勾配降下法）-> 'adam'（自動学習率調整アルゴリズム）
# learning_rate_init(初期の学習率): 0.1->0.001
# early_stopping(早期停止):-> True (過学習防止)
# random_state(シード値): 1->42
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=200, early_stopping=True, verbose=True, random_state=42)
mlp.fit(X_train, y_train)

# テストデータでの予測と精度の計算
accuracy = mlp.score(X_test, y_test)
print("Accuracy: ", accuracy)
