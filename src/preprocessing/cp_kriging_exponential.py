#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pump Efficiency Analysis with B-spline + Exponential Kriging
(수식만 로그로 출력)

Author: Yongbeen Kim
Date: 2025-09-29
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# -----------------------------
# Constants
# -----------------------------
RHO = 1000.0  # Water density [kg/m³]
G = 9.81      # Gravity [m/s²]

# -----------------------------
# Input Data
# -----------------------------
rpm = np.full(20, 900)
p_in = np.array([
    1.262, 1.262, 1.212, 0.858, 0.454, 0.0, -0.303, -0.555, -0.909, -1.262,
    -1.464, -1.767, -1.969, -2.020, -2.322, -2.423, -2.474, -2.474, -2.575, -2.575
]) * 1000
p_out = np.array([
    21.48, 20.78, 19.64, 18.15, 17.17, 15.45, 14.54, 13.91, 12.77, 11.86,
    11.16, 10.25, 10.02, 9.74, 9.16, 9.04, 9.24, 9.14, 9.06, 9.06
]) * 1000
Q = np.array([
    0.0527, 0.1191, 0.2793, 0.4258, 0.5449, 0.6641, 0.7168, 0.7695, 0.8242, 0.9023,
    0.9160, 0.9570, 0.9824, 1.0098, 1.0352, 1.0762, 1.0625, 1.0625, 1.0762, 1.0625
]) / 1000
v_in = np.array([
    0.1216, 0.2747, 0.6439, 0.9817, 1.2563, 1.5310, 1.6526, 1.7742, 1.9003, 2.0804,
    2.1119, 2.2065, 2.2650, 2.3281, 2.3866, 2.4812, 2.4496, 2.4496, 2.4812, 2.4496
])
v_out = np.array([
    0.2192, 0.4953, 1.1612, 1.7702, 2.2655, 2.7609, 2.9801, 3.1993, 3.4267, 3.7515,
    3.8084, 3.9789, 4.0844, 4.1981, 4.3037, 4.4742, 4.4174, 4.4174, 4.4742, 4.4174
])
he = np.full(20, 0.075)
torque = np.array([
    0.0402, 0.1098, 0.1345, 0.1484, 0.1561, 0.2041, 0.2041, 0.2242, 0.1994, 0.2535,
    0.2473, 0.2597, 0.2674, 0.2891, 0.2736, 0.2922, 0.3061, 0.2953, 0.3138, 0.3308
])

# -----------------------------
# Functions
# -----------------------------
def compute_efficiency(rpm, torque, p_in, p_out, Q, v_in, v_out, he):
    omega = rpm * np.pi / 30.0
    w_shaft = torque * omega
    ha = (p_out - p_in) / (RHO * G) + he + (v_out**2 - v_in**2) / (2 * G)
    w_hydraulic = RHO * G * Q * ha
    eta = np.clip((w_hydraulic / w_shaft) * 100, 0, 100)
    bep_idx = np.argmax(eta)
    return eta, Q[bep_idx], eta[bep_idx]

def cubic_spline_interpolation(x, y, x_new):
    n = len(x)
    y_new = np.zeros_like(x_new)
    for i, xi in enumerate(x_new):
        if xi <= x[0]:
            j = 0
        elif xi >= x[-1]:
            j = n - 2
        else:
            j = np.searchsorted(x, xi) - 1
        t = (xi - x[j]) / (x[j + 1] - x[j])
        y_new[i] = (1 - t) * y[j] + t * y[j + 1]
    return y_new

def exponential_variogram(h, nugget, sill, vrange):
    """
    γ(h) = nugget + sill * (1 - exp(-h / range))
    """
    return nugget + sill * (1 - np.exp(-h / vrange))

def ordinary_kriging_1d(x_data, y_data, x_pred, nugget, sill, vrange, variogram_func):
    n = len(x_data)
    cov_matrix = np.array([[sill - variogram_func(abs(xi - xj), 0, sill, vrange)
                            for xj in x_data] for xi in x_data])
    cov_matrix += 1e-10 * np.eye(n)
    cov_aug = np.zeros((n + 1, n + 1))
    cov_aug[:n, :n] = cov_matrix
    cov_aug[-1, :-1] = 1.0
    cov_aug[:-1, -1] = 1.0
    cov_aug[-1, -1] = 0.0
    y_pred, var_pred = [], []
    for xp in x_pred:
        k = np.array([sill - variogram_func(abs(xp - xi), 0, sill, vrange) for xi in x_data])
        weights = np.linalg.solve(cov_aug, np.append(k, 1.0))
        y_pred.append(np.dot(weights[:-1], y_data))
        var_pred.append(sill - np.dot(weights[:-1], k) + weights[-1])
    return np.array(y_pred), np.array(var_pred)

def semivariogram(x, y, n_lags=10):
    h_list, gamma_list = [], []
    n = len(x)
    for i in range(n - 1):
        for j in range(i + 1, n):
            h_list.append(abs(x[j] - x[i]))
            gamma_list.append(0.5 * (y[j] - y[i]) ** 2)
    h_list = np.array(h_list)
    gamma_list = np.array(gamma_list)
    lag_edges = np.linspace(0, h_list.max(), n_lags + 1)
    gamma_avg, lag_center = [], []
    for k in range(n_lags):
        mask = (h_list >= lag_edges[k]) & (h_list < lag_edges[k + 1])
        if np.any(mask):
            gamma_avg.append(np.mean(gamma_list[mask]))
            lag_center.append((lag_edges[k] + lag_edges[k + 1]) / 2)
    return np.array(lag_center), np.array(gamma_avg)

def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    eta, bep_Q, bep_eta = compute_efficiency(rpm, torque, p_in, p_out, Q, v_in, v_out, he)

    q_unique, unique_idx = np.unique(Q, return_index=True)
    eta_unique = eta[unique_idx]
    q_dense = np.linspace(q_unique.min(), q_unique.max(), 50)
    eta_spline_dense = cubic_spline_interpolation(q_unique, eta_unique, q_dense)

    nugget = 0.0
    sill = np.var(eta_unique)
    vrange = (q_unique.max() - q_unique.min()) * 0.1

    # 🔹 수식만 로그로 출력
    logger.info(exponential_variogram.__doc__.strip())
    logger.info(f"Parameters: nugget={nugget:.4f}, sill={sill:.4f}, range={vrange:.4f}")

    # -----------------------------
    # Plot B-spline + Kriging
    # -----------------------------
    points_list = [5, 10, 15, 25]
    colors = ["purple", "blue", "green", "orange"]

    plt.figure(figsize=(12, 6))
    plt.plot(Q, eta, "ro-", label="Original Data")
    plt.scatter(bep_Q, bep_eta, color="red", s=100, zorder=5)
    plt.text(bep_Q, bep_eta + 1,
             f"BEP\n(Q={bep_Q:.4f}, η={bep_eta:.2f}%)",
             color="red", fontweight="bold", ha="center")

    for n_pts, color in zip(points_list, colors):
        q_spline = np.linspace(q_unique.min(), q_unique.max(), n_pts)
        eta_spline = cubic_spline_interpolation(q_unique, eta_unique, q_spline)
        eta_pred_bs, _ = ordinary_kriging_1d(q_spline, eta_spline, q_dense, nugget, sill, vrange, exponential_variogram)
        rmse = compute_rmse(eta_spline_dense, eta_pred_bs)
        logger.info(f"Sample size={n_pts}, RMSE={rmse:.4f}")
        plt.plot(q_dense, eta_pred_bs, color=color, lw=2, label=f"B-spline + Kriging ({n_pts} pts)")

    plt.xlabel("Flow Rate Q [m³/s]")
    plt.ylabel("Efficiency η [%]")
    plt.title("Pump Efficiency: B-spline + Exponential Kriging")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
