"""
Code for Scientific Computation Project 1
Please add college id here
CID: 01859216
"""

C2B = {"A": 0, "C": 1, "G": 2, "T": 3}


# ===== Code for Part 1=====#
def part1(Xin, istar):
    """
    Sort list of integers in non-decreasing order
    Input:
    Xin: List of N integers
    istar: integer between 0 and N-1 (0<=istar<=N-1)
    Output:
    X: Sorted list
    """
    X = Xin.copy()
    for i, x in enumerate(X[1:], 1):
        if i <= istar:
            ind = 0
            for j in range(i - 1, -1, -1):
                if x >= X[j]:
                    ind = j + 1
                    break
        else:
            a = 0
            b = i - 1
            while a <= b:
                c = (a + b) // 2
                if X[c] < x:
                    a = c + 1
                else:
                    b = c - 1
            ind = a

        X[ind + 1 : i + 1] = X[ind:i]
        X[ind] = x

    return X


def part1_time(inputs=None):
    """Examine dependence of walltimes of part1 function on N and istar
    You may modify the input/output as needed.
    """

    # Add code here for part 1, question 2

    return None  # Modify if needed


# ===== Code for Part 2=====#


def part2_naive(S, T, m):
    """Find locations in S of all length-m sequences in T
    Input:
    S,T: length-n and length-l gene sequences provided as strings

    (Naive implementation), takes about 9.66s for sequence from test_sequence.txt
    with T = "ATCGATCTGTTACGC" and m = 3

    Output:
    L: A list of lists where L[i] is a list containing all locations
    in S where the length-m sequence starting at T[i] can be found.
    """
    n, l = len(S), len(T)
    # Handle Edge Cases
    if m > len(T) or m > n:
        return []

    # First get all unique m-length contiguous substrings of T
    # Time complexity: O(l)
    subsequences = []
    for i in range(l - m + 1):
        t = T[i : i + m]
        if t not in subsequences:  # O(len(subsequences))
            subsequences.append(t)

    L = []
    # Match subsequences of T against S
    # Time complexity: O(l * nm)
    for subsequence in subsequences:
        subsequence_L = []
        for i in range(n - m + 1):  # O(n)
            if S[i : i + m] == subsequence:  # O(m)
                subsequence_L.append(i)
        L.append(subsequence_L)

    return L


def char2base4(S):
    """Convert gene string to list of ints."""
    return [C2B[s] for s in S]


def heval(L, base, prime):
    """Convert list L to base-10 number mod prime
    where base specifies the base of L
    """
    f = 0
    for l in L[:-1]:
        f = base * (l + f)

    return (f + (L[-1])) % prime


def part2(S, T, m):
    """Find locations in S of all length-m sequences in T
    Input:
    S,T: length-n and length-l gene sequences provided as strings

    For each sequence of m consecutive ‘letters’ in T, you should find all
    locations where the sequence can be found in S. For example, if
    S = GTCATGGCTGCATAG, T = TCATG, and m = 3, the search should identify the
    locations 1, 2, 10, and 3. Specifically, the function should return a
    list of lists, L, where L[i] is a list containing all locations in S where
    the ith length-m sequence in T can be found (for the example above,
    L = [[1],[2,10],[3]]). If the ith sequence in T is not found anywhere in S,
    you should have, L[i] = [].

    Output:
    L: A list of lists where L[i] is a list containing all locations
    in S where the length-m sequence starting at T[i] can be found.
    """
    n, l = len(S), len(T)
    # Handle Edge Cases
    if m > len(T) or m > n:
        return []

    subseqs = []
    # O(l)
    for i in range(l - m + 1):
        t = T[i:i + m]
        if t not in subseqs:
            subseqs.append(t)

    X = char2base4(S)  # O(n)
    base = 4
    q = 7919
    L = []
    # O(len(subseqs) * m)
    hashtable = {heval(char2base4(P), base, q): k for k, P in enumerate(subseqs)}
    for k, P in enumerate(subseqs):
        i = 0
        imatch = []
        hi = heval(X[:m], base, q)  # O(m)
        if hashtable.get(hi, -1) == k:
            if S[i:i + m] == P:  # O(m)
                imatch.append(i)

        bm = (4**m) % q
        for i in range(1, n - m + 1):
            # Update rolling hash
            hi = (hi * 4 - int(X[i - 1]) * bm + int(X[i - 1 + m])) % q
            if hashtable.get(hi, -1) == k:
                if S[i:i + m] == P:
                    imatch.append(i)

        L.append(imatch)

    return L


if __name__ == "__main__":
    import time

    with open("test_sequence.txt", "r") as f:
        sequence = f.read()

    S = sequence
    T = sequence[:100]
    # T = "ATCGATCTGTTACGC"
    # S = "GTCATGGCTGCATAG"
    # T = "TCATGCAT"
    m = 10  # => 'TCA', 'CAT', 'ATG'
    start_time = time.perf_counter()
    naive_solution = part2_naive(S, T, m)
    end_time = time.perf_counter()
    print(f"Naive Time taken: {end_time - start_time:.10f}s")
    start_time = time.perf_counter()
    rk_solution = part2(S, T, m)
    end_time = time.perf_counter()
    print(f"Rabin-Karp Time taken: {end_time - start_time:.10f}s")
    print(f"Naive and Rabin-Karp Solutions Agree? {naive_solution == rk_solution}")
