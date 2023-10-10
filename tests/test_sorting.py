import pytest
import sco.sorting
import random


def test_merge_sort():
    for i in range(1000):
        test_list = [random.randint(-1000, 1000) for i in range(random.randint(1, 100))]
        sorted_list = sco.sorting.merge_sort(test_list)
        assert sorted_list == sorted(test_list)
