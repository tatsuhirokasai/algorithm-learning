from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import datasets
import matplotlib.pyplot as plt

# データを生成（scikit-learnのデータセットを使用）
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# データを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 線形回帰モデルを生成し、訓練データを使って学習
model = LinearRegression()
model.fit(X_train, y_train)

# テストデータを使って予測
predictions = model.predict(X_test)

# 結果をプロット
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, predictions, color='blue', linewidth=3)
#plt.show()

# 画像として保存
plt.savefig('linear_regression_result.png')
