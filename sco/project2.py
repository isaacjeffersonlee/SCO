"""
Code for Scientific Computation Project 2
Please add college id here
CID:
"""
import heapq
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# use scipy in part 2 as needed
from scipy.integrate import solve_ivp
from scipy.ndimage import shift


# ===== Codes for Part 1=====#
def searchGPT(graph, source, target):
    # Initialize distances with infinity for all nodes except the source
    distances = {node: float("inf") for node in graph}
    distances[source] = 0

    # Initialize a priority queue to keep track of nodes to explore
    priority_queue = [(0, source)]  # (distance, node)

    # Initialize a dictionary to store the parent node of each node in the shortest path
    parents = {}

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # If the current node is the target, reconstruct the path and return it
        if current_node == target:
            path = []
            while current_node in parents:  # Reconstruct path
                path.insert(0, current_node)
                current_node = parents[current_node]
            path.insert(0, source)
            return current_distance, path

        # If the current distance is greater than the known distance, skip this node
        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = max(distances[current_node], weight["weight"])
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                parents[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return float("inf")  # No path exists


def searchPKR(G, s, x):
    """Input:
    G: weighted NetworkX graph with n nodes (numbered as 0,1,...,n-1)
    s: an integer corresponding to a node in G
    x: an integer corresponding to a node in G
    """
    Fdict = {}  # Explored finalized nodes
    Mdict = {}
    Mlist = []
    dmin = float("inf")
    n = len(G)
    G.add_node(n)
    heapq.heappush(Mlist, [0, s])
    Mdict[s] = Mlist[0]
    found = False

    while len(Mlist) > 0:
        dmin, nmin = heapq.heappop(Mlist)
        if nmin == x:
            found = True
            break

        Fdict[nmin] = dmin

        for m, en, wn in G.edges(nmin, data="weight"):
            if en in Fdict:
                pass
            elif en in Mdict:
                dcomp = max(dmin, wn)
                if dcomp < Mdict[en][0]:
                    l = Mdict.pop(en)
                    l[1] = n
                    lnew = [dcomp, en]
                    heapq.heappush(Mlist, lnew)
                    Mdict[en] = lnew
            else:
                dcomp = max(dmin, wn)
                lnew = [dcomp, en]
                heapq.heappush(Mlist, lnew)
                Mdict[en] = lnew

    return dmin


def searchPKR2(G, s, x):
    """Input:
    G: weighted NetworkX graph with n nodes (numbered as 0,1,...,n-1)
    s: an integer corresponding to a node in G
    x: an integer corresponding to a node in G
    Output: Should produce equivalent output to searchGPT given same input
    """
    Fdict = {}  # Keep track of visited nodes
    Mdict = {}  # node: (node, distance) key-value pairs
    Mlist = []  # Priority queue
    dmin = float("inf")
    n = len(G)
    G.add_node(n)
    heapq.heappush(Mlist, [0, s])
    Mdict[s] = Mlist[0]
    found = False
    parents = {}

    while len(Mlist) > 0:
        dmin, nmin = heapq.heappop(Mlist)
        if nmin == x:
            found = True
            break

        Fdict[nmin] = dmin

        for m, en, wn in G.edges(nmin, data="weight"):
            if en in Fdict:  # Stops us from going backwards
                pass
            elif en in Mdict:
                dcomp = max(dmin, wn)
                if dcomp < Mdict[en][0]:
                    l = Mdict.pop(en)
                    l[1] = n
                    lnew = [dcomp, en]
                    parents[en] = m
                    heapq.heappush(Mlist, lnew)
                    Mdict[en] = lnew
            else:
                dcomp = max(dmin, wn)
                lnew = [dcomp, en]
                parents[en] = m
                heapq.heappush(Mlist, lnew)
                Mdict[en] = lnew

    # Reconstruct path
    path = [0 for i in range(len(parents) + 1)]  # Pre-allocate path array
    current_node = x
    i = len(parents)
    while current_node in parents:
        path[i] = current_node
        current_node = parents[current_node]
        i -= 1

    path[i] = s
    return dmin, path


# ===== Code for Part 2=====#
def part2q1(y0, tf=1, Nt=5000):
    """
    Part 2, question 1
    Simulate system of n nonlinear ODEs

    Input:
    y0: Initial condition, size n array
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)
    seed: ensures same intial condition is generated with each simulation
    Output:
    tarray: size Nt+1 array
    yarray: Nt+1 x n array containing y at
            each time step including the initial condition.
    """
    import numpy as np

    # Set up parameters, arrays
    n = y0.size
    tarray = np.linspace(0, tf, Nt + 1)
    yarray = np.zeros((Nt + 1, n))
    yarray[0, :] = y0
    beta = 0.04 / np.pi**2
    alpha = 1 - 2 * beta

    def RHS(t, y):
        """
        Compute RHS of model
        """
        dydt = np.zeros_like(y)
        for i in range(1, n - 1):
            dydt[i] = alpha * y[i] - y[i] ** 3 + beta * (y[i + 1] + y[i - 1])

        dydt[0] = alpha * y[0] - y[0] ** 3 + beta * (y[1] + y[-1])
        dydt[-1] = alpha * y[-1] - y[-1] ** 3 + beta * (y[0] + y[-2])

        return dydt

    # Compute numerical solutions
    dt = tarray[1]
    for i in range(Nt):
        yarray[i + 1, :] = yarray[i, :] + dt * RHS(0, yarray[i, :])

    return tarray, yarray


def part2q1new(y0, tf=40, Nt=800):
    """
    Part 2, question 1
    Simulate system of n nonlinear ODEs

    Input:
    y0: Initial condition, size n array
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)
    seed: ensures same intial condition is generated with each simulation
    Output:
    tarray: size Nt+1 array
    yarray: Nt+1 x n array containing y at
            each time step including the initial condition.
    """
    beta = 10000 / np.pi**2
    alpha = 1 - 2 * beta

    def RHS(t, y):
        y_prev = shift(y, 1, cval=0)
        y_next = shift(y, -1, cval=0)
        dydt = alpha * y - y**3 + beta * (y_next + y_prev)
        dydt[0] = alpha * y[0] - y[0] ** 3 + beta * (y[1] + y[-1])
        dydt[-1] = alpha * y[-1] - y[-1] ** 3 + beta * (y[0] + y[-2])
        return dydt

    n = y0.size
    tarray = np.linspace(0, tf, Nt + 1)
    yarray = np.zeros((Nt + 1, n))
    yarray[0, :] = y0
    sol = solve_ivp(RHS, t_span=[0, tf], y0=y0, t_eval=tarray, rtol=1e-6, atol=1e-6)
    assert np.array_equal(sol.t, tarray)
    return tarray, sol.y.T


def part2q1new_decoupled(y0, tf=40, Nt=800):
    """
    Part 2, question 1
    Simulate system of n nonlinear ODEs

    Input:
    y0: Initial condition, size n array
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)
    seed: ensures same intial condition is generated with each simulation
    Output:
    tarray: size Nt+1 array
    yarray: Nt+1 x n array containing y at
            each time step including the initial condition.
    """

    # Set up parameters, arrays
    n = y0.size
    tarray = np.linspace(0, tf, Nt + 1)
    yarray = np.zeros((Nt + 1, n))
    yarray[0, :] = y0
    dydt = lambda t, y: y - y**3
    sol = solve_ivp(dydt, t_span=[0, tf], y0=y0, t_eval=tarray, rtol=1e-6, atol=1e-6)
    assert np.array_equal(sol.t, tarray)
    return tarray, sol.y.T


def part2q2_initial_exploration(y0A, y0B):
    dy0A = np.diff(y0A)
    dy0B = np.diff(y0B)
    tarray, yarray_A = part2q1new(y0=y0A, tf=40, Nt=800)
    tarray, yarray_B = part2q1new(y0=y0B, tf=40, Nt=800)
    tarray, yarray_A_decoupled = part2q1new_decoupled(y0=y0A, tf=40, Nt=800)
    tarray, yarray_B_decoupled = part2q1new_decoupled(y0=y0B, tf=40, Nt=800)
    fig, ax = plt.subplots(figsize=(30, 40), nrows=6, ncols=1)
    ax[0].plot(tarray, yarray_A)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("$y_i(t)$")
    ax[0].set_title("$y_{0A}$ Solutions")
    ax[1].plot(tarray, yarray_B)
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("$y_i(t)$")
    ax[1].set_title("$y_{0B}$ Solutions")
    ax[2].plot(tarray, yarray_A_decoupled)
    ax[2].set_xlabel("t")
    ax[2].set_ylabel("$y_i(t)$")
    ax[2].set_title("$y_{0A}$ Decoupled Solutions")
    ax[3].plot(tarray, yarray_B_decoupled)
    ax[3].set_xlabel("t")
    ax[3].set_ylabel("$y_i(t)$")
    ax[3].set_title("$y_{0B}$ Decoupled Solutions")
    ax[4].scatter(x=range(len(y0A)), y=y0A, s=2, c="blue", label="$y$")
    ax[4].scatter(x=range(len(dy0A)), y=dy0A, s=2, c="red", label="$dy$")
    ax[4].set_xlabel("index i")
    ax[4].set_ylabel("$y_{0A}$[i]")
    ax[4].set_title("$y_{0A}$ Distribution")
    ax[5].scatter(x=range(len(y0B)), y=y0B, s=2, c="blue", label="$y$")
    ax[5].scatter(x=range(len(dy0B)), y=dy0B, s=2, c="red", label="$dy$")
    ax[5].set_xlabel("index i")
    ax[5].set_ylabel("$y_{0B}$[i]")
    ax[5].set_title("$y_{0B}$ Distribution")
    plt.legend()
    plt.show()


def linearize(y_bar, beta):
    """
    Generate the matrix M such that:

       dỹ/dt = Mỹ

    Where yᵢ := ȳᵢ + ϵỹᵢ + O(ϵ²).
    """
    n = len(y_bar)
    M = np.zeros((n, n))
    np.fill_diagonal(M, 1 - 3 * y_bar**2 - 2 * beta)
    np.fill_diagonal(M[1:, :], beta)
    np.fill_diagonal(M[:, 1:], beta)
    M[0, -1] = M[-1, 0] = beta
    return M


def part2q2_linearization(y0A, y0B):
    tarray, yarray_B = part2q1new(y0=y0B, tf=40, Nt=800)
    beta = 10000 / np.pi**2
    y_star = yarray_B[-1, :]  # Non-trivial Equilibrium point
    M = linearize(y_star, beta)
    evals, evecs = np.linalg.eigh(M)
    c_A = np.linalg.solve(
        evecs, y0A - y_star
    )  # Solve for constants using initial conditions y0A
    c_B = np.linalg.solve(
        evecs, y0B - y_star
    )  # Solve for constants using initial conditions y0B
    # Define y_tilde(t)
    y_tilde_A = lambda t: evecs @ (c_A * np.exp(evals * t))
    y_tilde_B = lambda t: evecs @ (c_B * np.exp(evals * t))
    # Map our perturbation solution functions to t values
    y_tilde_A_array = tuple(map(y_tilde_A, tarray))
    y_tilde_B_array = tuple(map(y_tilde_B, tarray))
    fig, ax = plt.subplots(figsize=(30, 20), nrows=2, ncols=1)
    ax[0].plot(tarray, y_tilde_A_array)
    ax[0].set_xlabel("$t$")
    ax[0].set_ylabel("$\\tilde{y}(t)$")
    ax[0].set_title("Perturbation solutions for $y_{0A}$")
    ax[1].plot(tarray, y_tilde_B_array)
    ax[1].set_xlabel("$t$")
    ax[1].set_ylabel("$\\tilde{y}(t)$")
    ax[1].set_title("Perturbation solutions for $y_{0B}$")
    plt.show()


def part2q2():  # add input variables if needed
    """
    Add code used for part 2 question 2.
    Code to load initial conditions is included below
    """

    data = np.load("project2.npy")
    y0A = data[0, :]
    y0B = data[1, :]
    # Make sure to close the plot window to allow the second part to run
    part2q2_initial_exploration(y0A, y0B)
    part2q2_linearization(y0A, y0B)


def part2q3(tf=10, Nt=1000, mu=0.2, seed=1):
    """
    Input:
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf
    mu: model parameter
    seed: ensures same random numbers are generated with each simulation

    Output:
    tarray: size Nt+1 array
    X size n x Nt+1 array containing solution
    """

    # Set initial condition
    y0 = np.array([0.3, 0.4, 0.5])
    np.random.seed(seed)
    n = y0.size  # must be n=3
    Y = np.zeros(
        (Nt + 1, n)
    )  # may require substantial memory if Nt, m, and n are all very large
    Y[0, :] = y0

    Dt = tf / Nt
    tarray = np.linspace(0, tf, Nt + 1)
    beta = 0.04 / np.pi**2
    alpha = 1 - 2 * beta

    def RHS(t, y):
        """
        Compute RHS of model
        """
        dydt = np.array([0, 0, 0])
        dydt[0] = alpha * y[0] - y[0] ** 3 + beta * (y[1] + y[2])
        dydt[1] = alpha * y[1] - y[1] ** 3 + beta * (y[0] + y[2])
        dydt[2] = alpha * y[2] - y[2] ** 3 + beta * (y[0] + y[1])

        return dydt

    dW = np.sqrt(Dt) * np.random.normal(size=(Nt, n))

    # Iterate over Nt time steps
    for j in range(Nt):
        y = Y[j, :]
        F = RHS(0, y)
        Y[j + 1, 0] = y[0] + Dt * F[0] + mu * dW[j, 0]
        Y[j + 1, 1] = y[1] + Dt * F[1] + mu * dW[j, 1]
        Y[j + 1, 2] = y[2] + Dt * F[2] + mu * dW[j, 2]

    return tarray, Y


def part2q3Analyze():  # add input variables as needed
    """
    Code for part 2, question 3
    """

    # add code for generating figures and any other relevant calculations here
    mu = 0.0001
    tarray, yarray_1 = part2q3(tf=40, Nt=800, mu=mu)
    fig, ax = plt.subplots(figsize=(30, 20), nrows=2, ncols=1)
    ax[0].plot(tarray, yarray_1)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("$y_i(t)$")
    ax[0].set_title(f"$y$ Solutions for $\\mu$={mu}")

    mu = 0.5
    tarray, yarray_2 = part2q3(tf=40, Nt=800, mu=mu)
    ax[1].plot(tarray, yarray_2)
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("$y_i(t)$")
    ax[1].set_title(f"$y$ Solutions for $\\mu$={mu}")

    plt.show()

    return None  # modify as needed
