from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# アヤメのデータセットを読み込む
iris = load_iris()
X = iris.data
y = iris.target

# データを訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ロジスティック回帰モデルを作成（ソフトマックス関数を使用）
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# モデルを訓練データにフィットさせる
model.fit(X_train, y_train)

# テストデータを用いて予測
predictions = model.predict(X_test)

# 結果を表示
print(predictions)
