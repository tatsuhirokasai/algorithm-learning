from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# アヤメのデータセットを読み込み
iris = load_iris()
x = iris.data
y = iris.target

# データを訓練データとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ロジスティック回帰モデルを作成
model = LogisticRegression(max_iter=1000)

# モデルを訓練データにフィット
model.fit(x_train, y_train)

# テストデータを用いて予測
predictions = model.predict(x_test)

# 結果を表示
print(predictions)
