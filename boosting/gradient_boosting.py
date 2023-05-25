from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# digits データセットをロードします
digits = load_digits()

# データを訓練セットとテストセットに分割します
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# GradientBoostingClassifierを初期化します
clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

# モデルを訓練します
clf.fit(X_train, y_train)

# テストセットに対する予測を行います
y_pred = clf.predict(X_test)

# モデルの精度を評価します
print("Accuracy:", accuracy_score(y_test, y_pred))
