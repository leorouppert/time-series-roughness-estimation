import os

import numpy as np
import scipy.special as special
from tqdm import tqdm

from src.constants import DATA_FOLDER


def generate_Z(H, eta, n_paths, n_points, T):
    time = np.linspace(0, 1, n_points)[1:]
    N_pts = n_points - 1

    sigma = np.empty((N_pts, N_pts))
    c = (eta**2) * (2 * H)

    # Exploit symmetry
    i_upper, j_upper = np.triu_indices(N_pts, k=1)

    s = time[i_upper]
    t = time[j_upper]

    off_diag_vals = c * (
        np.power(t - s, H - 0.5)
        / (H + 0.5)
        * np.power(s, 0.5 + H)
        * special.hyp2f1(0.5 - H, 0.5 + H, 1.5 + H, -s / (t - s))
    )

    sigma[i_upper, j_upper] = off_diag_vals
    sigma[j_upper, i_upper] = off_diag_vals

    np.fill_diagonal(sigma, (eta**2) * np.power(time, 2 * H))

    L = np.linalg.cholesky(sigma)

    gaussian = np.random.normal(loc=0.0, scale=1.0, size=(n_paths, N_pts))
    Z = np.zeros((n_paths, n_points))

    Z[:, 1:] = gaussian @ L.T

    return Z * np.power(T, H)


def generate_OU(X0, mu, theta, sigma, n_paths, n_points, T):
    t = np.linspace(0.0, T, n_points)[1:]

    m = X0 * np.exp(-theta * t) + mu * (1 - np.exp(-theta * t))

    S, T_grid = np.meshgrid(t, t)
    cov_matrix = (sigma**2 / (2 * theta)) * (
        np.exp(-theta * np.abs(T_grid - S)) - np.exp(-theta * (T_grid + S))
    )

    L = np.linalg.cholesky(cov_matrix)

    Z = np.random.normal(0, 1, size=(n_paths, n_points - 1))

    X = m[:, None].T + Z @ L.T

    X0_array = np.full((n_paths, 1), X0)
    X_full = np.hstack((X0_array, X))

    return X_full


def create_data(n_samples, n_points, T, n_H, eta=1):
    assert n_samples % n_H == 0, "n_samples should be a multiple of n_H"
    paths_per_H = n_samples // n_H

    h = np.random.uniform(0, 1, n_H)
    Z = np.zeros((n_samples, n_points))
    for i in tqdm(range(n_H)):
        Z[paths_per_H * i : paths_per_H * (i + 1), :] = generate_Z(
            h[i], eta, paths_per_H, n_points, T
        )
    H = np.repeat(h, paths_per_H)

    print("Saving data...")

    np.savetxt(
        os.path.join(DATA_FOLDER, str(n_samples) + "_" + str(n_points) + "_Z.csv"),
        Z,
        delimiter=",",
        fmt="%f",
    )
    np.savetxt(
        os.path.join(DATA_FOLDER, str(n_samples) + "_" + str(n_points) + "_H.csv"),
        H,
        delimiter=",",
        fmt="%f",
    )


def load_data(nb_train, n_points):
    print("Loading data...")
    Z = np.loadtxt(
        os.path.join(DATA_FOLDER, str(nb_train) + "_" + str(n_points) + "_Z.csv"),
        delimiter=",",
    )
    H = np.loadtxt(
        os.path.join(DATA_FOLDER, str(nb_train) + "_" + str(n_points) + "_H.csv"),
        delimiter=",",
    )
    return Z, H
