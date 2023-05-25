from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# データの読み込み
digits = datasets.load_digits()

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# モデルの作成
model = svm.SVC(kernel='poly', degree=3, random_state=42)

# モデルの訓練
model.fit(X_train, y_train)

# テストデータでの予測
y_pred = model.predict(X_test)

# 正解率の計算
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)
