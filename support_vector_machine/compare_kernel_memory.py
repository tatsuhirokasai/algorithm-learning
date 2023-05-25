from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import time
import tracemalloc

# データセットの読み込み
digits = datasets.load_digits()

# モデルの初期化
kernels = ['linear', 'poly', 'rbf']
for kernel in kernels:
    print(f'Processing {kernel} kernel:')
    svc = SVC(kernel=kernel, gamma='scale')

    # メモリ使用量と時間の計測開始
    tracemalloc.start()
    start_time = time.time()

    # モデルの訓練と交差検証
    scores = cross_val_score(svc, digits.data, digits.target, cv=5)
    print(f"Accuracy: {scores.mean()} (+/- {scores.std() * 2})")

    # メモリ使用量と時間の計測終了
    elapsed_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Current memory usage: {current / 10**6}MB; Peak : {peak / 10**6}MB")
    print(f"Elapsed time: {elapsed_time} seconds")
    print("----" * 10)
