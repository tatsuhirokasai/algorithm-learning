from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# データをロード
housing = fetch_california_housing()
x, y = housing.data, housing.target

# データを訓練セットとテストセットに分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# リッジ回帰モデルを作成
ridge = Ridge(alpha=1.0)

# モデルを訓練データにフィット
ridge.fit(x_train, y_train)

# テストデータを用いて予測を行う
y_pred = ridge.predict(x_test)

# 平均二乗誤差（MSE）を計算
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error: ', mse)