import numpy as np

def bootstrap_mean_confidence_interval(data, n_resamples=1000, alpha=0.05):
    resampled_means = []

    for _ in range(n_resamples):
        resampled_data = np.random.choice(data, len(data), replace=True)
        resampled_mean = np.mean(resampled_data)
        resampled_means.append(resampled_mean)

    lower = np.percentile(resampled_means, alpha / 2 * 100)
    upper = np.percentile(resampled_means, (1 - alpha / 2) * 100)

    return lower, upper

if __name__ == "__main__":
    data = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    n_resamples = 1000
    alpha = 0.05

    lower, upper = bootstrap_mean_confidence_interval(data, n_resamples, alpha)
    print(f"標本データ: {data}")
    print(f"ブートストラップ法による {100 * (1 - alpha)}% 信頼区間: ({lower:.2f}, {upper:.2f})")
