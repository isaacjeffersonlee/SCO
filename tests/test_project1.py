import pytest
from sco import project1


def test_part2():
    S = "GTCATGGCTGCATAG"
    T = "TCATG"
    m = 3  # => 'TCA', 'CAT', 'ATG'
    assert project1.part2(S, T, m) == [[1], [2, 10], [3]]
    S = "TTTGTGGGTA"
    T = "GGTAAACCCTTTAAACC"
    m = 1  # => 'G', 'T', 'A', 'C'
    assert project1.part2(S, T, m) == [[3, 5, 6, 7], [0, 1, 2, 4, 8], [9], []]
    S = "T"
    T = "AGCTAG"
    m = 2  # => 'AG', 'GC', 'CT', 'TA', 'AG'
    assert project1.part2(S, T, m) == []
    S = "A"
    T = "A"
    m = 1
    assert project1.part2(S, T, m) == [[0]]
    S = "ATC"
    T = "AG"
    m = 3
    assert project1.part2(S, T, m) == []
