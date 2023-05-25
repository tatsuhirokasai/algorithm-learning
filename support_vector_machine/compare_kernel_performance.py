from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# データの読み込み
digits = datasets.load_digits()

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# パラメータグリッドの設定
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4]},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

# グリッドサーチの設定
svc = svm.SVC(random_state=42)
clf = GridSearchCV(svc, param_grid, cv=5)

# モデルの訓練
clf.fit(X_train, y_train)

# ベストパラメータの表示
print("Best parameters: ", clf.best_params_)

# テストデータでの予測
y_pred = clf.predict(X_test)

# 正解率の計算
accuracy = clf.score(X_test, y_test)

print('Accuracy:', accuracy)