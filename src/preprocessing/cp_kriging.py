# 분석 스크립트: 데이터 불러오기, ha/Wshaft 계산, Gaussian Process(Kriging 유사) 보간, 시스템 곡선과 교점(운전점) 찾기, 시각화
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

# helper for displaying dataframe in the UI
try:
    from caas_jupyter_tools import display_dataframe_to_user
except Exception:
    display_dataframe_to_user = None

# -------------------------------
# 1) 파일 선택 다이얼로그
# -------------------------------
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()  # tkinter 기본 창 숨기기

file_path = filedialog.askopenfilename(
    title="Select Excel file",
    filetypes=[("Excel files", "*.xlsx *.xls")]
)

if not file_path:
    raise FileNotFoundError("파일을 선택하지 않았습니다.")

# -------------------------------
# 2) 데이터 불러오기
# -------------------------------
df = pd.read_excel(file_path, sheet_name="Sheet1")
df.columns = df.columns.str.strip()

# -------------------------------
# 3) 기본 상수 및 계산
# -------------------------------
RHO = 998.2  # kg/m3
G = 9.81     # m/s2

# 컬럼 자동 탐색
q_col = next(c for c in df.columns if "Flow" in c and "Q" in c)
p_in_col = next(c for c in df.columns if "Inlet Pressure" in c or "Pin" in c)
p_out_col = next(c for c in df.columns if "Outlet Pressure" in c or "Pout" in c)
v_in_col = next(c for c in df.columns if "Inlet Velocity" in c or "Vin" in c)
v_out_col = next(c for c in df.columns if "Outlet Velocity" in c or "Vout" in c)
he_col = next(c for c in df.columns if "Elevation Head" in c or "He" in c)
torque_col = next(c for c in df.columns if "Motor Torque" in c or "t [Nm]" in c)
rpm_col = next(c for c in df.columns if "Pump Speed" in c or "n [rpm]" in c)

# Q 단위 변환 (L/s → m3/s)
df["q_m3s"] = df[q_col] / 1000.0

# ha 계산
ramda = RHO * G / 1000.0  # kPa 환산계수
df["ha_m"] = (df[p_out_col] - df[p_in_col]) / ramda + df[he_col] + (df[v_out_col]**2 - df[v_in_col]**2) / (2*G)

# shaft power, hydraulic power, efficiency
df["p_in_Pa"] = df[p_in_col] * 1000.0
df["p_out_Pa"] = df[p_out_col] * 1000.0
df["w_hydraulic_W"] = RHO * G * df["q_m3s"] * df["ha_m"]
df["omega"] = df[rpm_col] * np.pi / 30.0
df["w_shaft_W"] = df[torque_col] * df["omega"]
df["efficiency_pct"] = (df["w_hydraulic_W"] / df["w_shaft_W"]).replace([np.inf, -np.inf], np.nan) * 100.0
df["efficiency_pct"] = df["efficiency_pct"].clip(lower=0.0, upper=100.0)

# 정렬
df = df.sort_values("q_m3s").reset_index(drop=True)

# Summary 출력
summary_cols = ["q_m3s", "ha_m", "efficiency_pct", "w_hydraulic_W", "w_shaft_W"]
summary = df[summary_cols].copy()
summary.columns = ["Q_m3s", "ha_m", "efficiency_pct", "W_hydraulic_W", "W_shaft_W"]

if display_dataframe_to_user:
    display_dataframe_to_user("Pump experimental summary", summary)
else:
    print(summary.head(10).to_string(index=False))

# -------------------------------
# 4) Gaussian Process (Kriging 유사) 보간
# -------------------------------
X = df["q_m3s"].values.reshape(-1,1)
y_ha = df["ha_m"].values
y_eff = df["efficiency_pct"].values

kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.01, length_scale_bounds=(1e-4, 1.0)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6,1e1))
gpr_ha = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True).fit(X, y_ha)
gpr_eff = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True).fit(X, np.nan_to_num(y_eff, nan=0.0))

Q_grid = np.linspace(df["q_m3s"].min() * 0.95, df["q_m3s"].max() * 1.05, 500).reshape(-1,1)
ha_pred, ha_std = gpr_ha.predict(Q_grid, return_std=True)
eff_pred, eff_std = gpr_eff.predict(Q_grid, return_std=True)

# -------------------------------
# 5) 시스템 곡선 계산
# -------------------------------
D = 0.02; L = 2.0; EPS = 0.00015; KL_SUM = 1.8; Z_DIFF = 2.0; A = np.pi*(D/2)**2
MU = 0.001003

def velocity_from_Q(q):
    return 4*q/(np.pi*D**2)

def reynolds(rho, mu, q):
    V = velocity_from_Q(q)
    return rho * V * D / mu

def friction_factor(Re):
    if Re < 2000:
        return 64.0/Re
    return (-1.8 * np.log10((EPS/(3.7*D)) + (5.74/(Re**0.9)))) ** -2

H_sys = np.array([Z_DIFF + ((friction_factor(reynolds(RHO, MU, q)) * L / D + KL_SUM) / (2*G*A**2)) * q**2 for q in Q_grid.flatten()])

# -------------------------------
# 6) Operating Point 찾기 (선형 보간)
# -------------------------------
def find_intersection(x, y1, y2):
    for i in range(len(x)-1):
        if (y1[i] - y2[i]) * (y1[i+1] - y2[i+1]) <= 0:
            x0 = x[i] + (x[i+1]-x[i]) * (y2[i]-y1[i]) / ((y1[i+1]-y2[i+1]) - (y1[i]-y2[i]))
            y0 = y1[i] + (y1[i+1]-y1[i]) * (x0-x[i]) / (x[i+1]-x[i])
            return float(x0), float(y0)
    return None, None

Q_op, H_op = find_intersection(Q_grid.flatten(), ha_pred, H_sys)

# -------------------------------
# 7) 최대 효율 BEP
# -------------------------------
idx_eff_max = np.argmax(eff_pred)
Q_eff_max = float(Q_grid[idx_eff_max])
eff_max = float(eff_pred[idx_eff_max])
ha_at_effmax = float(ha_pred[idx_eff_max])
Hsys_at_effmax = float(H_sys[idx_eff_max])

# -------------------------------
# 8) 효율 곡선 시각화
# -------------------------------
plt.figure(figsize=(10,6))
plt.plot(df["q_m3s"], df["efficiency_pct"], "o", label="Measured efficiency", markersize=6)
plt.plot(Q_grid.flatten(), eff_pred, "-", label="GPR (eff)", linewidth=2)
plt.fill_between(Q_grid.flatten(), eff_pred-1.96*eff_std, eff_pred+1.96*eff_std, alpha=0.2)
plt.scatter([Q_eff_max], [eff_max], color="green", s=80, label=f"Max Eff: {eff_max:.2f}% at Q={Q_eff_max:.4f}")
plt.xlabel("Flow rate Q [m³/s]"); plt.ylabel("Efficiency [%]")
plt.title("Efficiency Curve (Measured + GPR)")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# -------------------------------
# 9) 결과 요약 출력 및 자동 저장
# -------------------------------
result = {
    "Q_op_m3s": Q_op,
    "H_op_m": H_op,
    "Q_eff_max_m3s": Q_eff_max,
    "eff_max_pct": eff_max
}
result_df = pd.DataFrame([result])

if display_dataframe_to_user:
    display_dataframe_to_user("Kriging analysis results", result_df)
else:
    print("\nResult summary:\n", result_df.to_string(index=False))

out_csv = os.path.splitext(file_path)[0] + "_result.csv"
result_df.to_csv(out_csv, index=False)
print(f"Results saved to {out_csv}")
