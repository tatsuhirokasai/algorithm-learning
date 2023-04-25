import random

def randomized_quick_sort(arr, low, high):
    if low < high:
        pivot_index = randomized_partition(arr, low, high)
        randomized_quick_sort(arr, low, pivot_index - 1)
        randomized_quick_sort(arr, pivot_index + 1, high)

def randomized_partition(arr, low, high):
    pivot_index = random.randint(low, high - 1)
    arr[low], arr[pivot_index] = arr[pivot_index], arr[low]
    return partition(arr, low, high)

def partition(arr, low, high):
    pivot = arr[low]
    left = low + 1
    right = high

    while True:
        while left <= right and arr[left] <= pivot:
            left = left + 1
        while arr[right] >= pivot and right >= left:
            right = right - 1

        if right < left:
            break
        else:
            arr[left], arr[right] = arr[right], arr[left]

    arr[low], arr[right] = arr[right], arr[low]
    return right

if __name__ == "__main__":
    arr = [29, 10, 14, 37, 13, 25, 3, 20, 19, 15]
    print(f"ソート前: {arr}")
    randomized_quick_sort(arr, 0, len(arr) - 1)
    print(f"ラスベガス法（ランダム化クイックソート）適用後: {arr}")
