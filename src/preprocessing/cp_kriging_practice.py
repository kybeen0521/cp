import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

# --------------------------------------------------
# Configuration
# --------------------------------------------------
RHO: float = 1000.0  # Water density [kg/m³]
G: float = 9.81  # Gravity [m/s²]

plt.rcParams.update({
    "font.family": "Times New Roman",
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.replace("\n", "", regex=False)
        .str.replace("\t", "", regex=False)
        .str.replace("\xa0", "", regex=False)
        .str.strip()
    )
    return df


def find_stable_bep(df: pd.DataFrame) -> pd.Series:
    max_eff = df["efficiency"].max()
    candidates = df[df["efficiency"] >= 0.99 * max_eff].copy()
    candidates["y_diff"] = candidates["efficiency"].diff().abs().fillna(0) + \
                           candidates["efficiency"].diff(-1).abs().fillna(0)
    stable_bep = candidates.loc[candidates["y_diff"].idxmin()]
    return stable_bep


def plot_efficiency(df: pd.DataFrame) -> None:
    plt.plot(
        df["q_m3s"],
        df["efficiency"],
        "-o",
        color="green",
        markersize=6,
        linewidth=2,
        markerfacecolor="lime",
        label="Efficiency",
    )

    # 안정적인 BEP 표시
    bep_row = find_stable_bep(df)
    bep_q = bep_row["q_m3s"]
    bep_eff = bep_row["efficiency"]

    plt.scatter(bep_q, bep_eff, color="red", s=100, zorder=5)
    texts = [plt.text(
        bep_q, bep_eff + 0.5, f"BEP\n(Q={bep_q:.4f}, Eff={bep_eff:.2f}%)",
        fontsize=10, color="red", fontweight="bold", ha="center"
    )]

    adjust_text(
        texts,
        only_move={"points": "y", "text": "y"},
        arrowprops=dict(arrowstyle="->", color="red", lw=1),
        expand_points=(1.2, 1.2),
    )

    plt.xlabel("Flow rate Q [m³/s]")
    plt.ylabel("Efficiency [%]")
    plt.title("Pump Efficiency Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    logging.info(f"Stable BEP: {bep_eff:.2f}% at Q = {bep_q:.4f} m³/s")


# --------------------------------------------------
# Main Workflow
# --------------------------------------------------
def calculate_efficiency(df: pd.DataFrame) -> None:
    df = clean_columns(df)

    # ----------------------------
    # Column detection
    # ----------------------------
    try:
        torque_col = next(c for c in df.columns if "Torque" in c)
        rpm_col = next(c for c in df.columns if "Speed" in c or "RPM" in c)
        flow_col = next(c for c in df.columns if "Flow" in c and "Q" in c)
        p1_col = next(c for c in df.columns if "Inlet" in c and "Pressure" in c)
        p2_col = next(c for c in df.columns if "Outlet" in c and "Pressure" in c)
        v1_col = next(c for c in df.columns if "Inlet" in c and "Velocity" in c)
        v2_col = next(c for c in df.columns if "Outlet" in c and "Velocity" in c)
        he_col = next(c for c in df.columns if "Elevation Head" in c)
    except StopIteration:
        logging.error(f"Required columns not found. Available columns:\n{list(df.columns)}")
        return

    # ----------------------------
    # Calculations (BEP)
    # ----------------------------
    df["p_in_Pa"] = df[p1_col] * 1000.0  # kPa → Pa
    df["p_out_Pa"] = df[p2_col] * 1000.0
    df["q_m3s"] = df[flow_col] / 1000.0  # L/s → m³/s

    df["ha"] = ((df["p_out_Pa"] - df["p_in_Pa"]) / (RHO * G)
                + df[he_col]
                + (df[v2_col] ** 2 - df[v1_col] ** 2) / (2 * G))

    df["w_hydraulic"] = RHO * G * df["q_m3s"] * df["ha"]
    df["omega"] = df[rpm_col] * np.pi / 30.0
    df["w_shaft"] = df[torque_col] * df["omega"]

    df["efficiency"] = (df["w_hydraulic"] / df["w_shaft"]) * 100.0
    df["efficiency"] = df["efficiency"].clip(lower=0, upper=100)

    df_sorted = df.sort_values("q_m3s")

    logging.info(
        "\n=== Efficiency Results ===\n"
        + df_sorted[["q_m3s", "efficiency"]].to_string(
            index=False, header=["Flow rate [m³/s]", "Efficiency [%]"]
        )
    )

    plot_efficiency(df_sorted)


# --------------------------------------------------
# Input Data
# --------------------------------------------------
if __name__ == "__main__":
    data = {
        "Pump Speed n [rpm]": [
            900, 900, 900, 900, 900, 900, 900, 900, 900, 900,
            900, 900, 900, 900, 900, 900, 900, 900, 900, 900
        ],
        "Water Temperature T [캜]": [
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
        "Elevation Head He [m]": [
            0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075,
            0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075
        ],
        "Outlet Pressure Pout [kPa]": [
            21.48, 20.78, 19.64, 18.15, 17.17, 15.45, 14.54, 13.91, 12.77, 11.86,
            11.16, 10.25, 10.02, 9.74, 9.16, 9.04, 9.24, 9.14, 9.06, 9.06
        ],
        "Motor Torque t [Nm]": [
            0.0402, 0.1098, 0.1345, 0.1484, 0.1561, 0.2041, 0.2041, 0.2242, 0.1994, 0.2535,
            0.2473, 0.2597, 0.2674, 0.2891, 0.2736, 0.2922, 0.3061, 0.2953, 0.3138, 0.3308
        ],
    }

    df_input = pd.DataFrame(data)

    calculate_efficiency(df_input)
