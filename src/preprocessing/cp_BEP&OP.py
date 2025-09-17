import logging
from pathlib import Path
from tkinter import Tk, filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Configuration
# --------------------------------------------------
RHO = 1000.0
G = 9.81

plt.rcParams.update({
    "font.family": "Times New Roman",
    "figure.figsize": (10, 12),
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (df.columns.str.replace("\n", "", regex=False)
                            .str.replace("\t", "", regex=False)
                            .str.replace("\xa0", "", regex=False)
                            .str.strip())
    return df

def find_flow_column(df: pd.DataFrame):
    return next((c for c in df.columns if "Flow" in c and "Q" in c), None)

def calc_efficiency(df, torque_col, rpm_col, p_in_col, p_out_col, v1_col, v2_col, he_col, flow_col):
    df["q_m3s"] = df[flow_col] / 1000
    df["p_in_Pa"] = df[p_in_col] * 1000
    df["p_out_Pa"] = df[p_out_col] * 1000
    df["ha"] = ((df["p_out_Pa"] - df["p_in_Pa"]) / (RHO * G) +
                df[he_col] + (df[v2_col]**2 - df[v1_col]**2) / (2*G))
    df["omega"] = df[rpm_col] * np.pi / 30
    df["w_hydraulic"] = RHO * G * df["q_m3s"] * df["ha"]
    df["w_shaft"] = df[torque_col] * df["omega"]
    df["efficiency"] = (df["w_hydraulic"] / df["w_shaft"] * 100).clip(0, 100)
    return df

def find_stable_bep(df: pd.DataFrame) -> pd.Series:
    max_eff = df["efficiency"].max()
    candidates = df[df["efficiency"] >= 0.99 * max_eff].copy()
    candidates["y_diff"] = candidates["efficiency"].diff().abs().fillna(0) + candidates["efficiency"].diff(-1).abs().fillna(0)
    return candidates.loc[candidates["y_diff"].idxmin()]

def calc_system_head(Q: pd.Series, Z_DIFF=2.0, L=2.0, D=0.02, EPS=0.00015, KL_SUM=1.8):
    A = np.pi * (D/2)**2
    G_CONST = 9.81
    RHO_WATER = 998.2
    MU_WATER = 0.001003
    H_sys = []

    for q in Q:
        if q == 0:
            H_sys.append(Z_DIFF)
            continue
        V = 4*q/(np.pi*D**2)
        Re = RHO_WATER * V * D / MU_WATER
        if Re < 2000:
            f = 64/Re
        else:
            f = (-1.8*np.log10(EPS/(3.7*D) + 5.74/Re**0.9))**-2
        k = (f*L/D + KL_SUM)/(2*G_CONST*A**2)
        H_sys.append(Z_DIFF + k*q**2)
    return pd.Series(H_sys, index=Q.index)

def find_intersection(x, y1, y2):
    for i in range(len(x)-1):
        if (y1[i] - y2[i]) * (y1[i+1] - y2[i+1]) <= 0:
            # Linear interpolation
            denom = ((y1[i+1]-y2[i+1]) - (y1[i]-y2[i]))
            if denom == 0:
                continue
            x0 = x[i] + (x[i+1]-x[i]) * (y2[i]-y1[i]) / denom
            y0 = y1[i] + (y1[i+1]-y1[i]) * (x0-x[i]) / (x[i+1]-x[i])
            return x0, y0
    return None, None

def select_file() -> Path:
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Excel or CSV file",
        filetypes=[("Excel or CSV files", "*.xlsx *.xls *.csv")]
    )
    return Path(file_path) if file_path else None

# --------------------------------------------------
# Main Workflow
# --------------------------------------------------
def main():
    data_file = select_file()
    if not data_file:
        logging.error("No file selected. Exiting.")
        return
    logging.info(f"Selected file: {data_file}")

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

    try:
        torque_col = next(c for c in df.columns if "Torque" in c)
        rpm_col = next(c for c in df.columns if "Speed" in c or "RPM" in c)
        flow_col = find_flow_column(df)
        p1_col = next(c for c in df.columns if "Inlet" in c and "Pressure" in c)
        p2_col = next(c for c in df.columns if "Outlet" in c and "Pressure" in c)
        v1_col = next(c for c in df.columns if "Inlet" in c and "Velocity" in c)
        v2_col = next(c for c in df.columns if "Outlet" in c and "Velocity" in c)
        he_col = next(c for c in df.columns if "Elevation" in c or "He" in c)
    except StopIteration:
        logging.error(f"Required columns not found. Available columns:\n{list(df.columns)}")
        return

    # ---------------------
    # Calculations
    # ---------------------
    df = calc_efficiency(df, torque_col, rpm_col, p1_col, p2_col, v1_col, v2_col, he_col, flow_col)
    df_sorted = df.sort_values("q_m3s").reset_index(drop=True)
    bep_row = find_stable_bep(df_sorted)
    df_sorted["H_system"] = calc_system_head(df_sorted["q_m3s"])
    Q_op, H_op = find_intersection(df_sorted["q_m3s"], df_sorted["ha"], df_sorted["H_system"])

    # -----------------------------
    # Subplots (BEP/Efficiency + Head/OP)
    # -----------------------------
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,12))

    # Efficiency subplot
    ax1.plot(df_sorted["q_m3s"], df_sorted["efficiency"], "-o", color="green", label="Efficiency")
    ax1.scatter(bep_row["q_m3s"], bep_row["efficiency"], color="red", s=100, label="BEP")
    ax1.set_xlabel("Flow rate Q [m³/s]")
    ax1.set_ylabel("Efficiency [%]")
    ax1.set_title("Pump Efficiency Curve with BEP")
    ax1.legend()
    ax1.grid(True)

    # Head subplot (OP)
    ax2.plot(df_sorted["q_m3s"], df_sorted["ha"], "-o", color="blue", label="Actual Head")
    ax2.plot(df_sorted["q_m3s"], df_sorted["H_system"], "-o", color="green", label="System Curve")
    if Q_op is not None:
        ax2.scatter(Q_op, H_op, color="red", s=100, label="Operating Point")
        offset_y = 0.5
        ax2.annotate(
            f"OP ({Q_op:.4f}, {H_op:.2f})",
            xy=(Q_op, H_op),
            xytext=(Q_op, H_op + offset_y),
            textcoords='data',
            fontsize=10,
            color='red',
            arrowprops=dict(arrowstyle="->", color="red", lw=1),
            ha='center'
        )
    ax2.set_xlabel("Flow rate Q [m³/s]")
    ax2.set_ylabel("Head H [m]")
    ax2.set_title("Actual Head Curve & System Curve with Operating Point")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
