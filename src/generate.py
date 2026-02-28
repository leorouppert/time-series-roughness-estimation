import os

import numpy as np
import scipy.special as special
from tqdm import tqdm

from src.constants import DATA_FOLDER


def generate_Z(H, eta, N_paths, N_points, T):
    time = np.linspace(0, 1, N_points)[1:]
    N_pts = N_points - 1

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

    gaussian = np.random.normal(loc=0.0, scale=1.0, size=(N_paths, N_pts))
    Z = np.zeros((N_paths, N_points))

    Z[:, 1:] = gaussian @ L.T

    return Z * np.power(T, H)


def create_data(n_samples, N_points, T, n_H, eta=1):
    h = np.random.uniform(0, 1, n_H)
    Z = np.zeros((n_samples, N_points))
    for i in tqdm(range(n_H)):
        Z[n_samples // n_H * i : n_samples // n_H * (i + 1), :] = generate_Z(
            h[i], eta, n_samples // n_H, N_points, T
        )
    H = [h[i] for _ in range(n_samples // n_H) for i in range(n_H)]
    np.savetxt(
        os.path.join(DATA_FOLDER, str(n_samples) + "_" + str(N_points) + "_Z.csv"),
        Z,
        delimiter=",",
        fmt="%f",
    )
    np.savetxt(
        os.path.join(DATA_FOLDER, str(n_samples) + "_" + str(N_points) + "_H.csv"),
        H,
        delimiter=",",
        fmt="%f",
    )


def load_data(nb_train, N_points):
    Z = np.loadtxt(
        os.path.join(DATA_FOLDER, str(nb_train) + "_" + str(N_points) + "_Z.csv"),
        delimiter=",",
    )
    H = np.loadtxt(
        os.path.join(DATA_FOLDER, str(nb_train) + "_" + str(N_points) + "_H.csv"),
        delimiter=",",
    )
    return Z, H
