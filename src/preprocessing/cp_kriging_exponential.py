#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pump Efficiency Analysis with Natural Cubic Spline + Exponential Kriging + Adjustable Noise

Author: Yongbeen Kim
Date: 2025-09-30
"""

import logging
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# -----------------------------
# Configuration / Constants
# -----------------------------
CONFIG = {
    "RHO": 1000.0,             # Water density [kg/m³]
    "G": 9.81,                 # Gravity [m/s²]
    "KRIGING_NOISE": 1e-12,    # Noise for Kriging covariance
    "N_LAGS": 10,              # Semivariogram lag bins
    "PLOT_POINTS": [10, 15, 25, 30],
    "PLOT_COLORS": ["purple", "blue", "green", "orange"],
    "DENSE_POINTS": 50         # Interpolation points
}

# -----------------------------
# Input Data
# -----------------------------
RPM = np.full(20, 900)
P_IN = np.array([...]) * 1000.0  # 생략 가능
P_OUT = np.array([...]) * 1000.0
Q = np.array([...]) / 1000.0
V_IN = np.array([...])
V_OUT = np.array([...])
HE = np.full(20, 0.075)
TORQUE = np.array([...])

# -----------------------------
# Functions
# -----------------------------
def compute_efficiency(rpm: np.ndarray, torque: np.ndarray, p_in: np.ndarray,
                       p_out: np.ndarray, q: np.ndarray, v_in: np.ndarray,
                       v_out: np.ndarray, he: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Calculate pump efficiency and Best Efficiency Point (BEP).

    Returns:
        eta (np.ndarray): Efficiency [%]
        bep_q (float): Flow rate at BEP
        bep_eta (float): Efficiency at BEP
    """
    omega = rpm * np.pi / 30.0
    w_shaft = torque * omega
    ha = (p_out - p_in) / (CONFIG["RHO"] * CONFIG["G"]) + he + (v_out**2 - v_in**2) / (2 * CONFIG["G"])
    w_hydraulic = CONFIG["RHO"] * CONFIG["G"] * q * ha
    eta = np.clip((w_hydraulic / w_shaft) * 100.0, 0.0, 100.0)
    bep_idx = np.argmax(eta)
    return eta, q[bep_idx], eta[bep_idx]


def cubic_spline_natural(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """
    Natural cubic spline interpolation.
    """
    n = len(x)
    h = np.diff(x)
    b = np.diff(y) / h

    # Solve second derivatives
    A = np.zeros((n, n))
    rhs = np.zeros(n)
    A[0, 0] = A[-1, -1] = 1.0  # natural boundary

    for i in range(1, n-1):
        A[i, i-1] = h[i-1]
        A[i, i] = 2*(h[i-1] + h[i])
        A[i, i+1] = h[i]
        rhs[i] = 6*(b[i] - b[i-1])

    M = np.linalg.solve(A, rhs)

    # Interpolation
    y_new = np.zeros_like(x_new)
    for idx, xi in enumerate(x_new):
        i = np.searchsorted(x, xi) - 1
        i = np.clip(i, 0, n-2)
        hi = x[i+1] - x[i]
        xi_diff = xi - x[i]

        y_new[idx] = (
            (M[i+1]/(6*hi)) * xi_diff**3 +
            (M[i]/(6*hi)) * (hi - xi_diff)**3 +
            (y[i] - M[i]*hi**2/6) * (hi - xi_diff)/hi +
            (y[i+1] - M[i+1]*hi**2/6) * xi_diff/hi
        )
    return y_new


def exponential_variogram(h: float, nugget: float, sill: float, vrange: float) -> float:
    """Exponential variogram model."""
    return nugget + sill * (1.0 - np.exp(-h / vrange))


def ordinary_kriging_1d(x_data: np.ndarray, y_data: np.ndarray, x_pred: np.ndarray,
                         nugget: float, sill: float, vrange: float, variogram_func,
                         noise: float = CONFIG["KRIGING_NOISE"]) -> Tuple[np.ndarray, np.ndarray]:
    """1D ordinary Kriging with adjustable noise."""
    n = len(x_data)
    cov_matrix = np.array([[sill - variogram_func(abs(xi - xj), 0.0, sill, vrange)
                            for xj in x_data] for xi in x_data])
    cov_matrix += noise * np.eye(n)

    cov_aug = np.zeros((n+1, n+1))
    cov_aug[:n, :n] = cov_matrix
    cov_aug[-1, :-1] = 1.0
    cov_aug[:-1, -1] = 1.0
    cov_aug[-1, -1] = 0.0

    y_pred, var_pred = np.zeros(len(x_pred)), np.zeros(len(x_pred))
    for i, xp in enumerate(x_pred):
        k = np.array([sill - variogram_func(abs(xp - xi), 0.0, sill, vrange) for xi in x_data])
        sol = np.linalg.solve(cov_aug, np.append(k, 1.0))
        weights, lagrange = sol[:-1], sol[-1]
        y_pred[i] = np.dot(weights, y_data)
        var_pred[i] = sill - np.dot(weights, k) + lagrange

    return y_pred, var_pred


def semivariogram(x: np.ndarray, y: np.ndarray, n_lags: int = CONFIG["N_LAGS"]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute experimental semivariogram."""
    h_list, gamma_list = [], []
    n = len(x)
    for i in range(n-1):
        for j in range(i+1, n):
            h_list.append(abs(x[j]-x[i]))
            gamma_list.append(0.5*(y[j]-y[i])**2)
    h_arr, gamma_arr = np.array(h_list), np.array(gamma_list)
    lag_edges = np.linspace(0.0, h_arr.max(), n_lags+1)
    gamma_avg, lag_center = [], []
    for k in range(n_lags):
        mask = (h_arr >= lag_edges[k]) & (h_arr < lag_edges[k+1])
        if np.any(mask):
            gamma_avg.append(np.mean(gamma_arr[mask]))
            lag_center.append((lag_edges[k]+lag_edges[k+1])/2.0)
    return np.array(lag_center), np.array(gamma_avg)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred)**2))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res / ss_tot


def plot_efficiency(Q: np.ndarray, eta: np.ndarray, bep_q: float, bep_eta: float,
                    q_dense: np.ndarray, eta_dense: np.ndarray,
                    points_list: list, colors: list,
                    nugget_fit: float, sill_fit: float, range_fit: float):
    """Plot efficiency curves and Kriging predictions."""
    plt.figure(figsize=(12,6))
    plt.plot(Q, eta, "ro-", label="Original Data")
    plt.scatter(bep_q, bep_eta, color="red", s=100, zorder=5)
    plt.text(bep_q, bep_eta+1, f"BEP\n(Q={bep_q:.4f}, η={bep_eta:.2f}%)",
             color="red", fontweight="bold", ha="center")

    q_unique = np.unique(Q)
    eta_unique = eta[np.unique(Q, return_index=True)[1]]

    for n_pts, color in zip(points_list, colors):
        q_spline = np.linspace(q_unique.min(), q_unique.max(), n_pts)
        eta_spline = cubic_spline_natural(q_unique, eta_unique, q_spline)
        eta_pred, _ = ordinary_kriging_1d(q_spline, eta_spline, q_dense,
                                          nugget_fit, sill_fit, range_fit,
                                          exponential_variogram)
        rmse = compute_rmse(eta_dense, eta_pred)
        r2 = compute_r2(eta_dense, eta_pred)
        logger.info(f"Sample size={n_pts}, RMSE={rmse:.4f}, R²={r2:.4f}")
        plt.plot(q_dense, eta_pred, color=color, lw=2,
                 label=f"Cubic Spline + Kriging ({n_pts} pts)")

    plt.xlabel("Flow Rate Q [m³/s]")
    plt.ylabel("Efficiency η [%]")
    plt.title("Pump Efficiency: Natural Cubic Spline + Exponential Kriging")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    eta, bep_q, bep_eta = compute_efficiency(RPM, TORQUE, P_IN, P_OUT, Q, V_IN, V_OUT, HE)
    q_dense = np.linspace(Q.min(), Q.max(), CONFIG["DENSE_POINTS"])
    eta_dense = cubic_spline_natural(np.unique(Q), eta[np.unique(Q, return_index=True)[1]], q_dense)

    lag, gamma_exp = semivariogram(np.unique(Q), eta[np.unique(Q, return_index=True)[1]])
    nugget_fit = gamma_exp[0]
    sill_fit = gamma_exp.max()
    range_fit = lag[np.argmax(gamma_exp >= 0.95 * sill_fit)]

    logger.info(f"=== Variogram Estimated Parameters ===\nnugget={nugget_fit:.4f}, sill={sill_fit:.4f}, range={range_fit:.4f}")

    plot_efficiency(Q, eta, bep_q, bep_eta, q_dense, eta_dense,
                    CONFIG["PLOT_POINTS"], CONFIG["PLOT_COLORS"],
                    nugget_fit, sill_fit, range_fit)
