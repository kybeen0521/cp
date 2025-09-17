import logging
from pathlib import Path
from tkinter import Tk, filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Logging & Plot Configuration
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
plt.rcParams.update({
    "font.family": "Times New Roman",
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

# ----------------------------
# Pipe & Fluid Configuration
# ----------------------------
class PipeConfig:
    D: float = 0.02
    L: float = 2.0
    EPS: float = 0.00015
    KL_SUM: float = 1.8
    Z_DIFF: float = 2.0
    G: float = 9.81
    A: float = np.pi * (D / 2) ** 2

class FluidProperties:
    RHO: float = 998.2
    MU: float = 0.001003

# ----------------------------
# Utility Functions
# ----------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.replace("\n", "", regex=False)
                  .str.replace("\t", "", regex=False)
                  .str.replace("\xa0", "", regex=False)
                  .str.strip()
    )
    return df

def velocity(Q: float, config: PipeConfig) -> float:
    return 4 * Q / (np.pi * config.D**2)

def reynolds(rho: float, mu: float, Q: float, config: PipeConfig) -> float:
    V = velocity(Q, config)
    return rho * V * config.D / mu

def friction_factor(Re: float, config: PipeConfig) -> float:
    if Re < 2000:
        return 64.0 / Re
    return (-1.8 * np.log10((config.EPS / (3.7 * config.D)) + (5.74 / (Re**0.9)))) ** -2

def calc_system_head(Q: pd.Series, fluid: FluidProperties, config: PipeConfig) -> pd.Series:
    H_sys = []
    for q in Q:
        Re = reynolds(fluid.RHO, fluid.MU, q, config)
        f = friction_factor(Re, config)
        k = (f * config.L / config.D + config.KL_SUM) / (2 * config.G * config.A**2)
        H_sys.append(config.Z_DIFF + k * q**2)
    return pd.Series(H_sys, index=Q.index)

def calc_actual_head(p1, p2, v1, v2, z_diff=PipeConfig.Z_DIFF, rho=FluidProperties.RHO, g=PipeConfig.G) -> pd.Series:
    velocity_term = (v2**2 - v1**2) / (2 * g)
    ramda = rho * g / 1000
    return (p2 - p1) / ramda + z_diff + velocity_term

def select_file() -> Path:
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Excel or CSV file",
        filetypes=[("Excel or CSV files", "*.xlsx *.xls *.csv")]
    )
    return Path(file_path) if file_path else None

def find_intersection(x, y1, y2):
    """
    두 곡선 y1, y2의 교점을 선형 보간법으로 계산
    """
    for i in range(len(x)-1):
        if (y1[i] - y2[i]) * (y1[i+1] - y2[i+1]) <= 0:
            x0 = x[i] + (x[i+1]-x[i]) * (y2[i]-y1[i]) / ((y1[i+1]-y2[i+1]) - (y1[i]-y2[i]))
            y0 = y1[i] + (y1[i+1]-y1[i]) * (x0-x[i]) / (x[i+1]-x[i])
            return x0, y0
    return None, None

# ----------------------------
# Main Workflow
# ----------------------------
def main():
    # 1. 파일 선택
    data_file = select_file()
    if not data_file:
        logging.error("No file selected!")
        return
    logging.info(f"Selected file: {data_file}")

    # 2. 파일 읽기
    ext = data_file.suffix.lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(data_file)
    elif ext == ".csv":
        try:
            df = pd.read_csv(data_file, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(data_file, encoding="cp949")
    else:
        logging.error(f"Unsupported file type: {ext}")
        return

    df = clean_columns(df)

    # 3. 컬럼 탐색
    try:
        q_col = next(c for c in df.columns if "Flow" in c and "Q" in c)
        p1_col = next(c for c in df.columns if "Inlet" in c and "Pressure" in c)
        p2_col = next(c for c in df.columns if "Outlet" in c and "Pressure" in c)
        v1_col = next(c for c in df.columns if "Inlet" in c and "Velocity" in c)
        v2_col = next(c for c in df.columns if "Outlet" in c and "Velocity" in c)
    except StopIteration:
        logging.error("Required columns not found!")
        return

    # 4. Actual Head 계산
    df["Q_m3s"] = df[q_col] / 1000
    df["ha"] = calc_actual_head(df[p1_col], df[p2_col], df[v1_col], df[v2_col])
    df_sorted = df.sort_values("Q_m3s").reset_index(drop=True)

    # 5. System Curve 계산
    config = PipeConfig()
    fluid = FluidProperties()
    df_sorted["H_system"] = calc_system_head(df_sorted["Q_m3s"], fluid, config)

    # 6. Operating Point 계산 (교점)
    Q_op, H_op = find_intersection(df_sorted["Q_m3s"], df_sorted["ha"], df_sorted["H_system"])
    if Q_op is None:
        logging.warning("No intersection found between Actual Head and System Curve.")
    else:
        logging.info(f"Operating Point: ({Q_op:.4f}, {H_op:.2f})")

    # 7. 그래프 출력 (OP 텍스트 겹침 방지, 좌표 형태)
    plt.figure(figsize=(10,6))
    plt.plot(df_sorted["Q_m3s"], df_sorted["ha"], "-o", label="Actual Head", color="blue")
    plt.plot(df_sorted["Q_m3s"], df_sorted["H_system"], "-o", label="System Curve", color="green")

    if Q_op is not None:
        plt.scatter(Q_op, H_op, color="red", s=100, label="Operating Point")
        offset_y = 0.5
        plt.annotate(
            f"OP ({Q_op:.4f}, {H_op:.2f})",  # 좌표 형태
            xy=(Q_op, H_op),
            xytext=(Q_op, H_op + offset_y),
            textcoords='data',
            fontsize=10,
            color='red',
            arrowprops=dict(arrowstyle="->", color="red", lw=1),
            ha='center'
        )

    plt.xlabel("Flow rate Q [m³/s]")
    plt.ylabel("Head H [m]")
    plt.title("Actual Head Curve & System Curve with Operating Point")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
