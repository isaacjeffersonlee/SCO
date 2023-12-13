"""Scientific Computation Project 3
Your CID here: 01859216
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, solve_banded
from scipy.spatial.distance import pdist
from scipy import sparse
import time

# use scipy as needed

# ===== Code for Part 1=====#


def plot_field(lat, lon, u, time, levels=20):
    """
    Generate contour plot of u at particular time
    Use if/as needed
    Input:
    lat,lon: latitude and longitude arrays
    u: full array of wind speed data
    time: time at which wind speed will be plotted (index between 0 and 364)
    levels: number of contour levels in plot
    """
    plt.figure()
    plt.contourf(lon, lat, u[time, :, :], levels)
    plt.axis("equal")
    plt.grid()
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.show()

    return None


def plot_stacked_fields(nrows, ncols, lat, lon, u, stepsize=1, levels=20):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5, 50))
    t = 0
    for i in range(nrows):
        for j in range(ncols):
            t += stepsize
            contour = ax[i, j].contourf(lon, lat, u[t, :, :], levels)
            ax[i, j].set_title(f"t = {t}")
            if i == nrows - 1:
                ax[i, j].set_xlabel("Longitude")
            if j == 0:
                ax[i, j].set_ylabel("Latitude")
            # ax[i].axis("equal")
            # ax[i].grid()

    # Adjust the vertical space (hspace) between subplots
    plt.subplots_adjust(hspace=1.5)
    plt.suptitle("Wind speed across time")


def plot_field_animated(lat, lon, u, levels=20):
    """
    Generate contour plot of u at different times (animated)

    Input:
    lat, lon: latitude and longitude arrays
    u: full array of wind speed data with shape (num_times, num_latitudes, num_longitudes)
    levels: number of contour levels in the plot
    """
    fig, ax = plt.subplots()
    contour = ax.contourf(lon, lat, u[0, :, :], levels)
    ax.set_title("Wind Speed at Time 0")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.axis("equal")
    plt.grid()

    def update(frame):
        ax.clear()
        contour = ax.contourf(lon, lat, u[frame, :, :], levels)
        ax.set_title(f"Wind Speed at Time {frame}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.axis("equal")
        plt.grid()

    animation = FuncAnimation(fig, update, frames=u.shape[0], interval=200, blit=False)
    plt.show()


def pca(X):
    m, n = X.shape
    X = X - X.mean(axis=0)
    C = np.cov(X, rowvar=False)
    evals, evecs = eigh(C)
    idx = np.argsort(evals)[::-1]
    evecs, evals = evecs[:, idx], evals[idx]
    explained_variance_ratio = evals / np.sum(evals)
    return X @ evecs, explained_variance_ratio


def pca_svd(X):
    def _svd_flip(U, Vt):
        """Sign correction to ensure deterministic output from SVD."""
        max_abs_cols = np.argmax(np.abs(U), axis=0)
        signs = np.sign(U[max_abs_cols, range(U.shape[1])])
        U *= signs
        Vt *= signs[:, np.newaxis]
        return U, Vt

    n, m = X.shape
    X = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    U, Vt = _svd_flip(U, Vt)
    explained_variance = (S**2) / (n - 1)
    total_var = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_var
    return X @ Vt.T, explained_variance_ratio


def plot_explained_variances_through_space(u):
    explained_longitudinal_variances = [pca(u[t, :, :])[1] for t in range(365)]
    plt.title(
        f"Explained Variance for each longitudinal component across time ({len(explained_longitudinal_variances[0])})"
    )
    plt.xlabel("t")
    plt.ylabel("Explained Longitudinal Variance %")
    plt.plot(explained_longitudinal_variances)
    plt.show()

    explained_latitudinal_variances = [pca(u[t, :, :].T)[1] for t in range(365)]
    plt.title(
        f"Explained Variance for each latitudinal component across time ({len(explained_latitudinal_variances[0])})"
    )
    plt.xlabel("t")
    plt.ylabel("Explained Latitudinal Variance %")
    plt.plot(explained_latitudinal_variances)


def plot_explained_variance_through_time(u, lon, lat):
    _, explained_variance = pca(u.reshape(365, 16 * 144).T)
    fig, ax = plt.subplots()
    ax.plot(np.cumsum(explained_variance), marker="x", ms=5.0)
    ax.set_xlabel("n")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Cumulative Explained Variance of first n PC through Time")


def plot_first_n_principal_components_through_time(u, lon, lat, n):
    Z, _ = pca(u.reshape(365, 16 * 144).T)
    nrows = ncols = int(np.sqrt(n))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    c = 0
    for i in range(nrows):
        for j in range(ncols):
            zc = -Z[:, c].reshape((16, 144))  # Sign of PC is arbitrary
            ax[i, j].contourf(lon, lat, zc, 200)
            if j != 0:
                ax[i, j].set_yticks([])  # Turn off y ticks for non first columns
            if j == 0:
                ax[i, j].set_ylabel("Latitude")
            if i != nrows - 1:
                ax[i, j].set_xticks([])  # Turn off xticks for non last rows
            if i == nrows - 1:
                ax[i, j].set_xlabel("Longitude")
            ax[i, j].set_title(f"PC: {c}")
            c += 1

    plt.subplots_adjust(hspace=1.2)
    plt.suptitle(f"First {n} principal components")


def plot_fourier_spectra(lat, lon, u):
    C = np.log(np.sum(np.abs(np.fft.fftshift(np.fft.fft2(u))) ** 2, axis=0))
    plt.figure()
    plt.contourf(lon, lat, C, 200)


def part1():  # add input if needed
    """
    Code for part 1
    """

    # --- load data ---#
    d = np.load("data1.npz")
    lat = d["lat"]
    lon = d["lon"]
    u = d["u"]
    # -------------------------------------#
    plot_field_animated(lat, lon, u, levels=20)
    plot_stacked_fields(10, 5, lat, lon, u, levels=20, stepsize=1)
    plot_explained_variances_through_space(u)
    plot_explained_variance_through_time(u, lon, lat)
    plot_first_n_principal_components_through_time(u, lon, lat, n=100)
    plot_fourier_spectra(lat, lon, u)
    plt.show()

    # get the list of frequencies

    return None  # modify if needed


# ===== Code for Part 2=====#
def part2(f, method=2):
    """
    Question 2.1 i)
    Input:
        f: m x n array
        method: 1 or 2, interpolation method to use
    Output:
        fI: interpolated data (using method)
    """

    m, n = f.shape
    fI = np.zeros((m - 1, n))  # use/modify as needed

    if method == 1:
        fI = 0.5 * (f[:-1, :] + f[1:, :])
    else:
        # Coefficients for method 2
        alpha = 0.3
        a = 1.5
        b = 0.1

        # coefficients for near-boundary points
        a_bc, b_bc, c_bc, d_bc = (5 / 16, 15 / 16, -5 / 16, 1 / 16)

        # add code here
        # Idea: A @ fI = B @ f
        ab = np.zeros((3, m - 1))
        ab[0, 2:] = ab[2, :-2] = alpha
        ab[1, :] = 1
        # Construct B
        B = np.zeros((m - 1, m))
        np.fill_diagonal(B[1:-1, 0 : m - 4 + 1], b / 2)
        np.fill_diagonal(B[1:-1, 1 : m - 4 + 2], a / 2)
        np.fill_diagonal(B[1:-1, 2 : m - 4 + 3], a / 2)
        np.fill_diagonal(B[1:-1, 3 : m - 4 + 4], b / 2)
        B[0, 0:4] = [a_bc, b_bc, c_bc, d_bc]
        B[-1, -4:] = [a_bc, b_bc, c_bc, d_bc]
        B = sparse.csr_matrix(B)
        # Solve A @ fI = B @ f for fI
        fI = solve_banded((1, 1), ab, B @ f, overwrite_ab=True, overwrite_b=True)

    return fI  # modify as needed


def plot_implicit_wavenumber_analysis(a, b, c, alpha, beta):
    kh = np.linspace(0, np.pi, 1000)
    method_1_kh_prime = np.sin(kh)
    A = a * np.sin(kh) + (b / 2) * np.sin(2 * kh) + (c / 3) * np.sin(3 * kh)
    B = 1 + 2 * alpha * np.cos(kh) + 2 * beta * np.cos(2 * kh)
    method_2_kh_prime = A / B
    fig, ax = plt.subplots()
    ax.plot(kh, kh, label="exact")
    ax.plot(kh, method_1_kh_prime, label="method 1")
    ax.plot(kh, method_2_kh_prime, label="method 2")
    ax.set_xlabel("kh")
    ax.set_ylabel("kh'")
    ax.legend()


def plot_walltime_vs_m():
    # Asymptotic time complexity experimental analysis
    m_vals = range(10, 30_000, 1000)
    times = {"method_1": [], "method_2": []}

    for m in m_vals:
        n = m  # "n is of the same magnitude as m."
        f = np.random.rand(m, n)
        start_1 = time.perf_counter()
        fI_1 = part2(f, method=1)
        end_1 = time.perf_counter()
        times["method_1"].append(end_1 - start_1)
        start_2 = time.perf_counter()
        fI_2 = part2(f, method=2)
        end_2 = time.perf_counter()
        times["method_2"].append(end_2 - start_2)

    fig, ax = plt.subplots(figsize=(20, 10))
    coeffs_1 = np.polyfit(m_vals, times["method_1"], 2)
    a_1, b_1, c_1 = coeffs_1
    f_1 = np.poly1d(coeffs_1)
    z_1 = f_1(m_vals)
    coeffs_2 = np.polyfit(m_vals, times["method_2"], 2)
    a_2, b_2, c_2 = coeffs_2
    f_2 = np.poly1d(coeffs_2)
    z_2 = f_2(m_vals)

    ax.plot(m_vals, times["method_1"], label="Method 1", marker="o")
    ax.plot(
        m_vals,
        z_1,
        label=f"Method 1 fitted quadratic: ${a_1:.10f}m^2 + {b_1:.10f}m + {c_1:.10f}$",
    )
    ax.plot(m_vals, times["method_2"], label="Method 2", marker="x")
    ax.plot(
        m_vals,
        z_2,
        label=f"Method 2 fitted quadratic: ${a_2:.10f}m^2 + {b_2:.10f}m + {c_2:.10f}$",
    )
    ax.set_xlabel("m")
    ax.set_ylabel("Wall Time (seconds)")
    ax.legend()


def part2_analyze():
    d = np.load("data1.npz")
    # lat = d["lat"]
    # lon = d["lon"]
    # u = d["u"]
    # t = 0
    # f = u[t]
    # m, n = f.shape  # arbitrary grid sizes
    # y = np.linspace(0, 1, m)
    # x = np.linspace(0, 1, n)
    # xg, yg = np.meshgrid(x, y)
    # dy = y[1] - y[0]
    # yI = y[:-1] + dy / 2  # grid for interpolated data
    # xg, ygI = np.meshgrid(x, yI)

    # Wavenumber analysis
    alpha = 0.3
    beta = 0.0
    a = 1.5
    b = 0.1
    c = 0.0
    plot_implicit_wavenumber_analysis(a, b, c, alpha, beta)
    # plot_walltime_vs_m()

    plt.show()

    return None


# ===== Code for Part 3=====#
def part3q1(y0, alpha, beta, b, c, tf=200, Nt=800, err=1e-6, method="RK45"):
    """
    Part 3 question 1
    Simulate system of 2n nonlinear ODEs

    Input:
    y0: Initial condition, size 2*n array
    alpha,beta,b,c: model parameters
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)

    Output:
    tarray: size Nt+1 array
    yarray: Nt+1 x 2*n array containing y at
            each time step including the initial condition.
    """

    # Set up parameters, arrays

    n = y0.size // 2
    tarray = np.linspace(0, tf, Nt + 1)
    yarray = np.zeros((Nt + 1, 2 * n))
    yarray[0, :] = y0

    def RHS(t, y):
        """
        Compute RHS of model
        """
        # add code here
        u = y[:n]
        v = y[n:]
        r2 = u**2 + v**2
        nu = r2 * u
        nv = r2 * v
        cu = np.roll(u, 1) + np.roll(u, -1)
        cv = np.roll(v, 1) + np.roll(v, -1)

        dydt = alpha * y
        dydt[:n] += beta * (cu - b * cv) - nu + c * nv + b * (1 - alpha) * v
        dydt[n:] += beta * (cv + b * cu) - nv - c * nu - b * (1 - alpha) * u

        return dydt

    sol = solve_ivp(
        RHS,
        (tarray[0], tarray[-1]),
        y0,
        t_eval=tarray,
        method=method,
        atol=err,
        rtol=err,
    )
    yarray = sol.y.T
    return tarray, yarray


def part3_analyze(display_1=False, display_2=False, display_3=False):
    """
    Part 3 question 1: Analyze dynamics generated by system of ODEs
    """

    def solve_u(c, n, tf=200, Nt=800):
        # Set parameters
        beta = (25 / np.pi) ** 2
        alpha = 1 - 2 * beta
        b = -1.5
        # Set initial conidition
        L = (n - 1) / np.sqrt(beta)
        k = 40 * np.pi / L
        a0 = np.linspace(0, L, n)
        A0 = np.sqrt(1 - k**2) * np.exp(1j * a0)
        y0 = np.zeros(2 * n)
        y0[:n] = 1 + 0.2 * np.cos(4 * k * a0) + 0.3 * np.sin(7 * k * a0) + 0.1 * A0.real
        t, y = part3q1(
            y0, alpha, beta, b, c, tf=20, Nt=2, method="RK45"
        )  # for transient, modify tf and other parameters as needed
        y0 = y[-1, :]
        t, y = part3q1(y0, alpha, beta, b, c, tf, Nt, method="RK45", err=1e-6)
        u, v = y[:, :n], y[:, n:]
        return t, u

    # -------------------------------------------#
    n = 4000
    i_vals = np.arange(100, n - 100 + 1)  # Only use results for 100 <= i <= n - 100

    # qualitative analysis for different c values.
    if display_1:
        for c in (0.5, 0.7, 0.9, 1.1, 1.3, 1.5):
            t, u = solve_u(c, n)
            u = u[:, i_vals]
            plt.figure()
            plt.contourf(i_vals, t, u, 20)
            plt.xlabel("i")
            plt.ylabel("t")
            plt.title(f"$u_i(t)$ for c={c}")
            plt.show()

    # c=1.3 analysis
    c = 1.3
    tf = 200
    t, u = solve_u(c, n, tf=tf)
    u = u[:, i_vals]

    # PCA Analysis
    # Calculate explained variances for timespans of 25
    # This quite intensive. Run if you dare.
    if display_2:
        delta_t = 25
        explained_variances = []
        t_vals = range(0, tf, delta_t)
        explained_variances = [pca(u[t : t + delta_t, :])[1] for t in t_vals]
        plt.title(
            f"Explained Variance between times $i*{delta_t}$ and $i*{delta_t}+{delta_t}$"
        )
        plt.xlabel("i")
        plt.ylabel("Explained Variance")
        plt.plot(explained_variances)
        plt.show()

    # Compute Correlation sum
    if display_3:
        D = pdist(u)
        eps_vals = np.linspace(np.min(D), np.max(D), num=100)[::-1]
        print(eps_vals)
        C_vals = []
        for i, eps in enumerate(eps_vals):
            D = D[D < eps]
            C_vals.append(D.size)

        # Re-order back to ascending eps for easier plotting
        eps_vals = eps_vals[::-1]
        C_vals = C_vals[::-1]
        # Fit least squares line of best fit to the center of the dataj
        coeffs = np.polyfit(np.log(eps_vals[5:50]), np.log(C_vals[5:50]), 1)
        f = np.poly1d(coeffs)
        z = f(np.log(eps_vals[5:50]))
        fig, ax = plt.subplots()
        ax.plot(np.log(eps_vals), np.log(C_vals), label="ln $C(\\epsilon)$", marker="x")
        ax.plot(
            np.log(eps_vals[5:50]),
            z,
            label=f"Least squares fit, $ln C(\\epsilon) = {coeffs[0]:.2f}ln \\epsilon + {coeffs[1]:.2f}$",
            marker="o",
        )
        ax.set_xlabel("ln$\\epsilon$")
        ax.set_ylabel("ln$C(\\epsilon$)")
        plt.legend()
        plt.show()

    return None


def part3q2(x, c=1.0):
    """
    Code for part 3, question 2
    """
    # Set parameters
    beta = (25 / np.pi) ** 2
    alpha = 1 - 2 * beta
    b = -1.5
    n = 4000

    # Set initial conidition
    L = (n - 1) / np.sqrt(beta)
    k = 40 * np.pi / L
    a0 = np.linspace(0, L, n)
    A0 = np.sqrt(1 - k**2) * np.exp(1j * a0)
    y0 = np.zeros(2*n)
    y0[:n] = 1 + 0.2 * np.cos(4 * k * a0) + 0.3 * np.sin(7 * k * a0) + 0.1 * A0.real

    # Compute solution
    t, y = part3q1(
        y0, alpha, beta, b, c, tf=20, Nt=2, method="RK45"
    )  # for transient, modify tf and other parameters as needed
    y0 = y[-1, :]
    t, y = part3q1(y0, alpha, beta, b, c, method="RK45", err=1e-6)
    A = y[:, :n]

    # Analyze code here
    l1, v1 = np.linalg.eigh(A.T.dot(A))  # Right singular vectors of A
    v2 = A.dot(v1)  # Left singular vectors of A
    A2 = (v2[:, :x]).dot((v1[:, :x]).T)
    e = np.sum((A2.real - A) ** 2)

    return A2.real, e


if __name__ == "__main__":
    x = None  # Included so file can be imported
    part2_analyze()
