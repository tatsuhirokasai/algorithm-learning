from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# digitsのデータセットを読み込む
digits = load_digits()

# データを訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# ランダムフォレストのモデルを作成
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# モデルを訓練データにフィットさせる
clf.fit(X_train, y_train)

# テストデータを用いて予測
y_pred = clf.predict(X_test)

# ストデータの正解率を計算
print("Accuracy:", accuracy_score(y_test, y_pred))
