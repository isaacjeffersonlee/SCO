from typing import List
import random


def merge(left: List, right: List) -> List:
    left_idx = right_idx = 0
    merged = []
    for i in range(len(left) + len(right)):
        if left_idx >= len(left):
            merged.extend(right[right_idx:])
            break
        elif right_idx >= len(right):
            merged.extend(left[left_idx:])
            break
        elif left[left_idx] <= right[right_idx]:
            merged.append(left[left_idx])
            left_idx += 1
        else:
            merged.append(right[right_idx])
            right_idx += 1

    return merged


# O(N * log_2(N))
def merge_sort(l: List) -> List:
    N = len(l)
    if len(l) == 1:
        return l
    mid_idx = N // 2
    left = merge_sort(l[:mid_idx])
    right = merge_sort(l[mid_idx:])
    return merge(left, right)


if __name__ == "__main__":
    for i in range(10000):
        test_list = [
            random.randint(-1000, 1000) for i in range(random.randint(1, 1000))
        ]
        sorted_list = merge_sort(test_list)
        assert sorted_list == sorted(test_list)

    print("All tests passed!")
