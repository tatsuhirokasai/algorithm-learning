from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# digits データセットをロード
digits = load_digits()

# データを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# SVCを初期化(デフォルトRBF)
clf = SVC(gamma='auto', random_state=42)

# モデルを訓練
clf.fit(X_train, y_train)

# テストセットに対する予測
y_pred = clf.predict(X_test)

# モデルの精度を評価
print("Accuracy:", accuracy_score(y_test, y_pred))
