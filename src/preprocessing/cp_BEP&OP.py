import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Pipe & Fluid Configuration
# ----------------------------
class PipeConfig:
    D = 0.02
    L = 2.0
    EPS = 0.00015
    KL_SUM = 1.8
    Z_DIFF = 2.0
    G = 9.81
    A = np.pi * (D/2)**2

class FluidProperties:
    RHO = 998.2
    MU = 0.001003

# ----------------------------
# Utility Functions
# ----------------------------
def velocity(Q, config):
    return 4 * Q / (np.pi * config.D**2)

def reynolds(rho, mu, Q, config):
    V = velocity(Q, config)
    return rho * V * config.D / mu

def friction_factor(Re, config):
    if Re < 2000:
        return 64.0 / Re
    return (-1.8 * np.log10((config.EPS / (3.7 * config.D)) + (5.74 / (Re**0.9)))) ** -2

def calc_system_head(Q, fluid, config):
    H_sys = []
    for q in Q:
        Re = reynolds(fluid.RHO, fluid.MU, q, config)
        f = friction_factor(Re, config)
        k = (f * config.L / config.D + config.KL_SUM) / (2 * config.G * config.A**2)
        H_sys.append(config.Z_DIFF + k * q**2)
    return np.array(H_sys)

def calc_actual_head(p1, p2, v1, v2, z_diff=PipeConfig.Z_DIFF, rho=FluidProperties.RHO, g=PipeConfig.G):
    velocity_term = (v2**2 - v1**2) / (2 * g)
    ramda = rho * g / 1000
    return (p2 - p1)/ramda + z_diff + velocity_term

def find_intersection(x, y1, y2):
    for i in range(len(x)-1):
        if (y1[i] - y2[i]) * (y1[i+1] - y2[i+1]) <= 0:
            x0 = x[i] + (x[i+1]-x[i])*(y2[i]-y1[i])/((y1[i+1]-y2[i+1])-(y1[i]-y2[i]))
            y0 = y1[i] + (y1[i+1]-y1[i])*(x0-x[i])/(x[i+1]-x[i])
            return x0, y0
    return None, None

def calc_efficiency(Q, H, torque, n_rpm):
    omega = 2*np.pi*n_rpm/60  # rad/s
    power_hydraulic = FluidProperties.RHO * PipeConfig.G * Q * H
    power_shaft = torque * omega
    return (power_hydraulic / power_shaft * 100).clip(upper=100)

# ----------------------------
# Example Data
# ----------------------------
data = {
    "Pump Speed n [rpm]": [900]*18,
    "Flow Rate Q [l/s]": [0.0527,0.1191,0.2793,0.4258,0.5449,0.6641,0.7168,0.7695,0.8242,0.9023,0.9160,0.9570,0.9824,1.0098,1.0352,1.0762,1.0625,1.0625],
    "Inlet Pressure Pin [kPa]": [1.262,1.262,1.212,0.858,0.454,0.0,-0.303,-0.555,-0.909,-1.262,-1.464,-1.767,-1.969,-2.02,-2.322,-2.423,-2.474,-2.474],
    "Outlet Pressure Pout [kPa]": [0.1216,0.2747,0.6439,0.9817,1.2563,1.5310,1.6526,1.7742,1.9003,2.0804,2.1119,2.2065,2.2650,2.3281,2.3866,2.4812,2.4496,2.4496],
    "Inlet Velocity Vin [m/s]": [0.2192]*18,
    "Outlet Velocity Vout [m/s]": [0.075]*18,
    "Motor Torque t [Nm]": [0.0402,0.1098,0.1345,0.1484,0.1561,0.2041,0.2041,0.2242,0.1994,0.2535,0.2473,0.2597,0.2674,0.2891,0.2736,0.2922,0.3061,0.2953]
}
df = pd.DataFrame(data)
df["Q_m3s"] = df["Flow Rate Q [l/s]"]/1000

# ----------------------------
# 계산
# ----------------------------
config = PipeConfig()
fluid = FluidProperties()

df["ha"] = calc_actual_head(df["Inlet Pressure Pin [kPa]"], df["Outlet Pressure Pout [kPa]"],
                            df["Inlet Velocity Vin [m/s]"], df["Outlet Velocity Vout [m/s]"])
df["H_system"] = calc_system_head(df["Q_m3s"], fluid, config)
df["efficiency"] = calc_efficiency(df["Q_m3s"], df["ha"], df["Motor Torque t [Nm]"], df["Pump Speed n [rpm]"])

# OP 계산
Q_op, H_op = find_intersection(df["Q_m3s"].values, df["ha"].values, df["H_system"].values)

# BEP 계산
idx_bep = df["efficiency"].idxmax()
Q_bep = df.loc[idx_bep,"Q_m3s"]
H_bep = df.loc[idx_bep,"ha"]
eff_bep = df.loc[idx_bep,"efficiency"]

# ----------------------------
# Figure with two Y axes
# ----------------------------
fig, ax1 = plt.subplots(figsize=(10,6))

color_head = "blue"
ax1.set_xlabel("Flow rate Q [m³/s]")
ax1.set_ylabel("Head H [m]", color=color_head)
ax1.plot(df["Q_m3s"], df["ha"], "-o", color=color_head, label="Actual Head")
ax1.plot(df["Q_m3s"], df["H_system"], "-o", color="green", label="System Curve")
ax1.tick_params(axis='y', labelcolor=color_head)
ax1.grid(True)

if Q_op is not None:
    ax1.scatter(Q_op, H_op, color="red", s=100, label="Operating Point (OP)")
    ax1.annotate(f"OP\n({Q_op:.4f},{H_op:.2f})", xy=(Q_op,H_op), xytext=(Q_op,H_op+0.5),
                 arrowprops=dict(arrowstyle="->", color="red"))

ax1.scatter(Q_bep, H_bep, color="purple", s=100, label=f"BEP ({eff_bep:.1f}%)")
ax1.annotate(f"BEP\n({eff_bep:.1f}%)", xy=(Q_bep,H_bep), xytext=(Q_bep,H_bep+0.5),
             arrowprops=dict(arrowstyle="->", color="purple"))

# Efficiency를 오른쪽 Y축으로
ax2 = ax1.twinx()
color_eff = "orange"
ax2.set_ylabel("Efficiency (%)", color=color_eff)
ax2.plot(df["Q_m3s"], df["efficiency"], "-s", color=color_eff, alpha=0.7, label="Efficiency")
ax2.tick_params(axis='y', labelcolor=color_eff)

# Legend 통합
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

plt.title("Pump Head Curve, System Curve, Efficiency, OP & BEP")
fig.tight_layout()
plt.show()
