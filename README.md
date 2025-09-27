# Centrifugal Pump Data Analysis & Prediction Pipeline

## Overview
This repository contains a set of Python tools for analyzing centrifugal pump performance,  
including **actual head curves**, **shaft power curves**, **pump efficiency curves**, and **system curve calculations**.  
The pipeline was developed as part of practice project.  

The workflow is designed for **cleaning raw Excel/CSV data, calculating key performance metrics, and visualizing results**,  
with automatic logging for reproducibility.


📑 For more details, please refer to [CP Report](CPreport_KimYongBeen.pdf)
---

## Features
- Actual head calculation and visualization
- Shaft power computation from torque and RPM
- Pump efficiency curve calculation with maximum efficiency detection
- Pipe system head curve calculation based on fluid properties and pipe configuration
- Automatic column detection for flexible input files
- Interactive file selection via GUI

---

## Data Processing Flow
Raw Excel/CSV Data  

↓ Step 1: Actual Head Analysis → `actual_head_curve.py`  

↓ Step 2: Shaft Power Calculation → `shaft_power_curve.py`  

↓ Step 3: Pump Efficiency Analysis → `pump_efficiency_curve.py`  

↓ Step 4: System Curve Analysis → `system_curve.py` 


↓ Step 5-1: Pump Efficiency Kriging Prediction Analysis → `cp_kriging_gaussian.py`


↓ Step 5-2: Pump Efficiency Kriging Prediction Analysis → `cp_kriging_exponential.py`


↓ Output: Cleaned Data, Calculated Metrics, Plots, Logs

---

## 📝 Step Descriptions

### Step 1: Actual Head Analysis
- **Script:** `src/actual_head_curve.py`
- **Input:** Excel file containing `Flow`, `Inlet Pressure`, `Outlet Pressure`, `Inlet/Outlet Velocity`
- **Process:**
  - Clean column names
  - Calculate actual head: `ha = (p2 - p1)/ (ρg) + z_diff + (v2^2 - v1^2)/(2g)`
  - Detect local increases in head
  - Plot `Q` vs. `ha` with anomalies highlighted
- **Output:**  
  - Plots: `output/plots/*.png`
  - Logs: `output/logs/actual_head_log.txt`

### Step 2: Shaft Power Calculation
- **Script:** `src/shaft_power_curve.py`
- **Input:** Excel file with `Torque` and `RPM`
- **Process:**
  - Compute angular velocity: `ω = RPM * π / 30`
  - Calculate shaft power: `W_shaft = Torque * ω`
  - Sort by flow rate
- **Output:**  
  - Shaft power plots: `output/plots/*.png`
  - Logs: `output/logs/shaft_power_log.txt`

### Step 3: Pump Efficiency Analysis
- **Script:** `src/pump_efficiency_curve.py`
- **Input:** Excel file with `Torque`, `RPM`, `Flow`, `Inlet/Outlet Pressure`, `Inlet/Outlet Velocity`
- **Process:**
  - Compute hydraulic power: `W_hydraulic = ρ * g * Q * ha`
  - Compute shaft power
  - Efficiency: `η = W_hydraulic / W_shaft * 100`
  - Plot efficiency vs. flow rate and highlight maximum efficiency
- **Output:**  
  - Efficiency plots: `output/plots/*.png`
  - Logs: `output/logs/efficiency_log.txt`

### Step 4: Pipe System Curve Calculation
- **Script:** `src/system_curve.py`
- **Input:** Excel file with `Flow` rate data
- **Process:**
  - Calculate velocity: `V = 4 * Q / (π * D^2)`
  - Compute Reynolds number: `Re = ρ * V * D / μ`
  - Determine Darcy friction factor (laminar/Haaland)
  - Compute system head: `H_system = Z_DIFF + ((f*L/D + ΣKL)/(2*g*A^2)) * Q^2`
  - Plot system curve
- **Output:**  
  - System curve plots: `output/plots/system_curve.png`
  - Logs: `output/logs/system_curve_log.txt`

### Step 5-1: Pump Efficiency Kriging Prediction (Gaussian)
- **Script:** src/cp_kriging_gaussian.py
- **Input:** Unique Flow vs. Efficiency data from Step 3
- **Process:**
  - Generate B-spline interpolation of efficiency curve
  - Apply 1D Ordinary Kriging with Gaussian semivariogram
  - Predict efficiency at multiple points (e.g., 10, 15, 30, 50, 100 points)
  - Visualize predicted curves alongside original data and BEP

- **Output:** 
  - Kriging prediction plots: output/plots/kriging_gaussian_*.png
  - Logs: output/logs/kriging_gaussian_log.txt

### Step 5-2: Pump Efficiency Kriging Prediction (Exponential)
- **Script:** src/cp_kriging_exponential.py
- **Input:** Unique Flow vs. Efficiency data from Step 3
- **Process:**
  - Generate B-spline interpolation of efficiency curve
  - Apply 1D Ordinary Kriging with Gaussian semivariogram
  - Predict efficiency at multiple points (e.g., 10, 15, 30, 50, 100 points)
  - Visualize predicted curves alongside original data and BEP

- **Output:** 
  - Kriging prediction plots: output/plots/kriging_exponential_*.png
  - Logs: output/logs/kriging_gaussian_log.txt


---
## 📂 Project Directory Structure
```
data/
input/                  # Raw Excel/CSV files
output/
├─ plots/               # Generated plots from all steps
│  ├─ actual_head/
│  ├─ shaft_power/
│  ├─ efficiency/
│  ├─ system_curve/
│  └─ kriging/
│      ├─ gaussian/
│      └─ exponential/
├─ logs/                # Step-specific logs
│  ├─ actual_head_log.txt
│  ├─ shaft_power_log.txt
│  ├─ efficiency_log.txt
│  ├─ system_curve_log.txt
│  ├─ kriging_gaussian_log.txt
│  └─ kriging_exponential_log.txt

src/
├─ actual_head_curve.py
├─ shaft_power_curve.py
├─ pump_efficiency_curve.py
├─ system_curve.py
├─ cp_kriging_exponential.py    # Exponential Kriging prediction
├─ cp_kriging_gaussian.py       # Gaussian Kriging prediction
└─ utils/                       # Helper functions (clean_columns.py, calc_utils.py, etc.)
```
---


## 👤 Author
**Yongbeen Kim (김용빈)**  
Researcher, Intelligent Mechatronics Research Center, KETI


📅 Document last updated 2025.09.26

