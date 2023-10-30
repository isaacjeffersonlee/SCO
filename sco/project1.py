"""
Code for Scientific Computation Project 1
Please add college id here
CID: 01859216
"""

import time
import matplotlib.pyplot as plt
import seaborn as sns
import random

sns.set()

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
    num_runs = 10  # Number of benchmark runs to generate raw timings
    # First fix N and examine the istar Vs. time
    N_max = 1000
    istar_times = []
    istar_vals = range(N_max)
    for istar in istar_vals:
        N = N_max
        raw_times = []
        for i in range(num_runs):
            # Generate a random list of integers, of length N
            Xin = random.sample(range(-10 * N_max, 10 * N_max), N)
            start_time = time.perf_counter()
            part1(Xin, istar)
            end_time = time.perf_counter()
            raw_times.append(end_time - start_time)

        avg_time = sum(raw_times) / len(raw_times)
        istar_times.append(avg_time)

    istar_strs = ("0", "N // 2", "N - 1")
    istar_timings_map = {s: [] for s in istar_strs}
    N_vals = range(N_max + 1)
    for N in N_vals:
        for istar_str, istar in zip(istar_strs, (0, N // 2, N - 1)):
            raw_times = []
            for i in range(num_runs):
                Xin = random.sample(range(-10 * N_max, 10 * N_max), N)
                start_time = time.perf_counter()
                part1(Xin, istar)
                end_time = time.perf_counter()
                raw_times.append(end_time - start_time)

            avg_time = sum(raw_times) / len(raw_times)
            istar_timings_map[istar_str].append(avg_time)

    fig, ax = plt.subplots(nrows=2, ncols=1)
    g = sns.lineplot(x=istar_vals, y=istar_times, ax=ax[0])
    ax[0].set_xlabel("istar")
    ax[0].set_ylabel("wall time (seconds)")
    ax[0].set_title(f"wall times for different istar values for N={N_max}")
    for istar_str, N_times in istar_timings_map.items():
        g = sns.lineplot(x=N_vals, y=N_times, ax=ax[1], label=f"istar={istar_str}")

    ax[1].set_xlabel("N")
    ax[1].set_ylabel("wall time (seconds)")
    ax[1].set_title("wall times for different N values")
    plt.suptitle("part1 wall times")
    plt.show()


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
    subseqs = []
    for i in range(l - m + 1):
        t = T[i : i + m]
        if t not in subseqs:  # O(len(subsequences))
            subseqs.append(t)

    d = {P: [] for P in subseqs}
    # Match subsequences of T against S
    # Time complexity: O(l * nm)
    for P in subseqs:
        for i in range(n - m + 1):  # O(n)
            if S[i : i + m] == P:  # O(m)
                d[P].append(i)

    L = [d[T[i : i + m]] for i in range(l - m + 1)]
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

    Multi-pattern Rabin-Karp Implementation.

    Output:
    L: A list of lists where L[i] is a list containing all locations
    in S where the length-m sequence starting at T[i] can be found.
    """
    n, l = len(S), len(T)
    # Handle Edge Cases
    if m > len(T) or m > n:
        return []

    # First we find all unique contiguous length m subsequences of T.
    subseqs = []
    for i in range(l - m + 1):
        t = T[i:i + m]
        if t not in subseqs:  # Uniqueness check, preserving ordering.
            subseqs.append(t)

    # Create a map that maps each hash to a list of subsequence numbers, k
    # which have that hash. E.g if T = "CATCATCAT" and m = 3 then the unique
    # subsequences in T are given by: "CAT" => k = 0, "ATC" => k = 1, "TCA" => k = 2.
    # But of course we can have hash collisions, i.e suppose our hash for
    # "CAT" was the same as the hash for "ATC" and was equal to h, and then the hash
    # for "TCA" is h'. So hash_to_k_map will map each unique hash to a list of matching
    # indices. E.g in this example we have: hash_to_k_map = {h: [0, 1], h': [2]}.
    base, q = 4, 7919
    hash_to_k_map = {}
    for k, P in enumerate(subseqs):
        h = heval(char2base4(P), base, q)
        if h not in hash_to_k_map:
            hash_to_k_map[h] = [k]
        else:  # Handle collisions
            hash_to_k_map[h].append(k)
    # Appending -1 to X is a workaround to avoid indexing errors for hi, without
    # having to use branching logic, which would be less efficient.
    X = char2base4(S)
    X.append(-1)
    bm = (4**m) % q
    # Initialize a map from each unique subsequence in T to a list
    # of indices in S where that subsequence appears.
    subseq_to_i_map = {P: [] for P in subseqs}
    hi = heval(X[:m], base, q)
    for i in range(n - m + 1):
        if hash_to_k_map.get(hi, -1) != -1:
            possible_k = hash_to_k_map[hi]
            for k in possible_k:
                P = subseqs[k]
                if S[i:i + m] == P:
                    subseq_to_i_map[P].append(i)
                    break
        # Update rolling hash
        hi = (hi * 4 - X[i] * bm + X[i + m]) % q
    # Finally we apply our map over all (not necessarily unique) contiguous
    # length m subsequences of T, giving us L, which we return.
    # Doing it like this means we don't have to repeat calculations
    # for duplicate contiguous subsequences of T.
    return [subseq_to_i_map[T[i:i + m]] for i in range(l - m + 1)]


if __name__ == "__main__":
    with open("test_sequence.txt", "r") as f:
        sequence = f.read()

    S = sequence[:40000]
    T = sequence[:20000]
    m = 1000
    start_time = time.perf_counter()
    naive_solution = part2_naive(S, T, m)
    end_time = time.perf_counter()
    print(f"Naive Time taken: {end_time - start_time:.10f}s")
    start_time = time.perf_counter()
    rk_solution = part2(S, T, m)
    end_time = time.perf_counter()
    print(f"Rabin-Karp Time taken: {end_time - start_time:.10f}s")
    print(f"Naive and Rabin-Karp Solutions Agree? {naive_solution == rk_solution}")

    # part1_time()
