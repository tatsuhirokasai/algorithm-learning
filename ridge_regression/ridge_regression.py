from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# データをロード
housing = fetch_california_housing()
X, y = housing.data, housing.target

# データを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# リッジ回帰モデルを作成
ridge = Ridge(alpha=1.0)

# モデルを訓練データにフィット
ridge.fit(X_train, y_train)

# テストデータを用いて予測を行う
y_pred = ridge.predict(X_test)

# 平均二乗誤差（MSE）を計算
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error: ', mse)