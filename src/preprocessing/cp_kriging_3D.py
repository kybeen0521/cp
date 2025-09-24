from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import filedialog

# -------------------------------
# 1) 파일 선택 및 데이터 불러오기
# -------------------------------
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select Excel file", filetypes=[("Excel files", "*.xlsx *.xls")])
if not file_path:
    raise FileNotFoundError("파일을 선택하지 않았습니다.")

df = pd.read_excel(file_path, sheet_name="Sheet1")
df.columns = df.columns.str.strip()

# -------------------------------
# 2) 컬럼 자동 탐색
# -------------------------------
q_col = next(c for c in df.columns if "Flow" in c and "Q" in c)
v_in_col = next(c for c in df.columns if "Inlet Velocity" in c or "Vin" in c)
p_in_col = next(c for c in df.columns if "Inlet Pressure" in c or "Pin" in c)
p_out_col = next(c for c in df.columns if "Outlet Pressure" in c or "Pout" in c)
he_col = next(c for c in df.columns if "Elevation Head" in c or "He" in c)
v_out_col = next(c for c in df.columns if "Outlet Velocity" in c or "Vout" in c)

# -------------------------------
# 3) 기본 상수 및 ha 계산
# -------------------------------
RHO = 998.2
G = 9.81
df["q_m3s"] = df[q_col] / 1000.0
ramda = RHO * G / 1000.0
df["ha_m"] = (df[p_out_col] - df[p_in_col]) / ramda + df[he_col] + (df[v_out_col]**2 - df[v_in_col]**2) / (2*G)

# -------------------------------
# 4) Gaussian Process (2D 입력)
# -------------------------------
X_train = df[["q_m3s", v_in_col]].values  # 2D 입력: Q, Vin
y_train = df["ha_m"].values

kernel = C(1.0) * RBF(length_scale=[0.01,0.01]) + WhiteKernel(noise_level=1e-3)
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(X_train, y_train)

# -------------------------------
# 5) 3D 그리드 생성 및 예측
# -------------------------------
q_grid = np.linspace(df["q_m3s"].min(), df["q_m3s"].max(), 50)
v_grid = np.linspace(df[v_in_col].min(), df[v_in_col].max(), 50)
Q_grid, V_grid = np.meshgrid(q_grid, v_grid)
X_pred = np.column_stack([Q_grid.ravel(), V_grid.ravel()])

ha_pred, ha_std = gpr.predict(X_pred, return_std=True)
HA = ha_pred.reshape(Q_grid.shape)

# -------------------------------
# 6) 3D surface plot
# -------------------------------
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Q_grid, V_grid, HA, cmap='viridis', alpha=0.8)
ax.set_xlabel("Flow rate Q [m³/s]")
ax.set_ylabel("Inlet Velocity Vin [m/s]")
ax.set_zlabel("ha [m]")
ax.set_title("3D Surface of ha (Q vs Vin)")
fig.colorbar(surf, shrink=0.5, aspect=10, label="ha [m]")
plt.show()
