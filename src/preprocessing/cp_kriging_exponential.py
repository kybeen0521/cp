
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# --------------------------------------------------
# Configuration
# --------------------------------------------------
RHO = 1000.0  # kg/m³
G = 9.81      # m/s²

# --------------------------------------------------
# Input Data (원본 제공)
# --------------------------------------------------
data = {
    "Pump Speed n [rpm]": [900]*20,
    "Water Temperature T [°C]": [
        25.1, 25.45, 25.5, 25.3, 25.25, 25.35, 25.15, 25.2, 25.1, 25.4,
        25.45, 25.3, 25.3, 24.9, 24.95, 25.55, 25.35, 25.15, 25.2, 25.25
    ],
    "Inlet Pressure Pin [kPa]": [
        1.262, 1.262, 1.212, 0.858, 0.454, 0.0, -0.303, -0.555, -0.909, -1.262,
        -1.464, -1.767, -1.969, -2.020, -2.322, -2.423, -2.474, -2.474, -2.575, -2.575
    ],
    "Flow Rate Q [l/s]": [
        0.0527, 0.1191, 0.2793, 0.4258, 0.5449, 0.6641, 0.7168, 0.7695, 0.8242, 0.9023,
        0.9160, 0.9570, 0.9824, 1.0098, 1.0352, 1.0762, 1.0625, 1.0625, 1.0762, 1.0625
    ],
    "Inlet Velocity Vin [m/s]": [
        0.1216, 0.2747, 0.6439, 0.9817, 1.2563, 1.5310, 1.6526, 1.7742, 1.9003, 2.0804,
        2.1119, 2.2065, 2.2650, 2.3281, 2.3866, 2.4812, 2.4496, 2.4496, 2.4812, 2.4496
    ],
    "Outlet Velocity Vout [m/s]": [
        0.2192, 0.4953, 1.1612, 1.7702, 2.2655, 2.7609, 2.9801, 3.1993, 3.4267, 3.7515,
        3.8084, 3.9789, 4.0844, 4.1981, 4.3037, 4.4742, 4.4174, 4.4174, 4.4742, 4.4174
    ],
    "Elevation Head He [m]": [0.075]*20,
    "Outlet Pressure Pout [kPa]": [
        21.48, 20.78, 19.64, 18.15, 17.17, 15.45, 14.54, 13.91, 12.77, 11.86,
        11.16, 10.25, 10.02, 9.74, 9.16, 9.04, 9.24, 9.14, 9.06, 9.06
    ],
    "Motor Torque t [Nm]": [
        0.0402, 0.1098, 0.1345, 0.1484, 0.1561, 0.2041, 0.2041, 0.2242, 0.1994, 0.2535,
        0.2473, 0.2597, 0.2674, 0.2891, 0.2736, 0.2922, 0.3061, 0.2953, 0.3138, 0.3308
    ],
}

df = pd.DataFrame(data)

# --------------------------------------------------
# Efficiency 계산
# --------------------------------------------------
Q = df["Flow Rate Q [l/s]"].values / 1000.0  # m³/s
rpm = df["Pump Speed n [rpm]"].values
torque = df["Motor Torque t [Nm]"].values
p_in = df["Inlet Pressure Pin [kPa]"].values * 1000.0  # Pa
p_out = df["Outlet Pressure Pout [kPa]"].values * 1000.0
v_in = df["Inlet Velocity Vin [m/s]"].values
v_out = df["Outlet Velocity Vout [m/s]"].values
he = df["Elevation Head He [m]"].values

omega = rpm * np.pi / 30.0
w_shaft = torque * omega

ha = (p_out - p_in)/(RHO*G) + he + (v_out**2 - v_in**2)/(2*G)
w_hydraulic = RHO * G * Q * ha

eta = np.clip((w_hydraulic / w_shaft) * 100, 0, 100)
df["eta [%]"] = eta

# --------------------------------------------------
# BEP 계산 (원본 데이터 기준)
# --------------------------------------------------
bep_idx = np.argmax(eta)
bep_Q = Q[bep_idx]
bep_eta = eta[bep_idx]

# --------------------------------------------------
# 유니크 원본 데이터 (중복 제거)
# --------------------------------------------------
q_unique, unique_idx = np.unique(Q, return_index=True)
eta_unique = eta[unique_idx]

# --------------------------------------------------
# Ordinary Kriging (1D Gaussian)
# --------------------------------------------------
def ordinary_kriging_1d(x_data, y_data, x_pred, sill=None, range_factor=0.1):
    n = len(x_data)
    if sill is None:
        sill = np.var(y_data)
    range_ = (x_data.max() - x_data.min()) * range_factor

    cov_matrix = np.array([[sill * np.exp(-abs(x_data[i]-x_data[j])/range_) for j in range(n)] for i in range(n)])
    cov_matrix += 1e-10 * np.eye(n)

    cov_aug = np.zeros((n+1, n+1))
    cov_aug[:n,:n] = cov_matrix
    cov_aug[-1,:-1] = 1.0
    cov_aug[:-1,-1] = 1.0
    cov_aug[-1,-1] = 0.0

    y_pred = []
    for x in x_pred:
        k = np.array([sill * np.exp(-abs(x - xi)/range_) for xi in x_data])
        k_aug = np.append(k, 1.0)
        weights = np.linalg.solve(cov_aug, np.append(y_data, 1.0))
        y_pred.append(np.dot(weights[:-1], k))
    return np.clip(np.array(y_pred), 0, 100)

# --------------------------------------------------
# B-spline + Kriging 예측 (10, 15, 50, 100 pts)
# --------------------------------------------------
points_list = [10, 15, 50, 100]
colors = ["purple", "blue", "green", "orange"]

spline_func = make_interp_spline(q_unique, eta_unique, k=3)

plt.figure(figsize=(10,6))
plt.plot(Q, eta, "ro-", label="Original Data")
plt.scatter(bep_Q, bep_eta, color="red", s=100, zorder=5)
plt.text(bep_Q, bep_eta+1, f"BEP\n(Q={bep_Q:.4f}, η={bep_eta:.2f}%)",
         color="red", fontweight="bold", ha="center")

for i, n_pts in enumerate(points_list):
    q_spline = np.linspace(q_unique.min(), q_unique.max(), n_pts)
    eta_spline = spline_func(q_spline)
    eta_pred = ordinary_kriging_1d(q_spline, eta_spline, q_spline)
    plt.plot(q_spline, eta_pred, color=colors[i], lw=2, label=f"Kriging Prediction ({n_pts} pts B-spline)")

plt.xlabel("Flow Rate Q [l/s]")
plt.ylabel("Efficiency η [%]")
plt.title("Pump Efficiency Curve with Kriging Prediction (Multiple B-spline Points)")
plt.ylim(0, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
