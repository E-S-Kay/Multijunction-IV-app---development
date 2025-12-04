import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import linregress

# ---------------------------------------------------------
# Helper: Diode Equation Solver
# ---------------------------------------------------------
def diode_current(V, Jph, J0, n, Rs, Rsh, T):
    k = 1.380649e-23
    q = 1.602176634e-19
    Vt = n * k * T / q
    return Jph - J0 * (np.exp((V + Rs * 0) / Vt) - 1) - V / Rsh

def solve_J_at_V0(Jph, J0, n, Rs, Rsh, T):
    f = lambda J: J - (Jph - J0 * (np.exp((0 + Rs * J) / (n * 1.380649e-23 * T / 1.602e-19)) - 1) - 0 / Rsh)
    return Jph  # For stability: photocurrent dominates at V=0

# ---------------------------------------------------------
# IV calculation for a subcell using diode equation
# ---------------------------------------------------------
def calculate_subcell_iv(Jph, J0, n, Rs, Rsh, T, num_points=500):
    J = np.linspace(-0.1, Jph, num_points)  # current axis
    k = 1.380649e-23
    q = 1.602176634e-19
    Vt = n * k * T / q

    V = J*0.0
    for i, Ji in enumerate(J):
        f = lambda V: Jph - J0*(np.exp((V + Ji*Rs)/Vt) - 1) - (V + Ji*Rs)/Rsh - Ji
        try:
            V[i] = brentq(f, -1.5, 2.5)
        except:
            V[i] = np.nan

    # Remove NaN
    mask = ~np.isnan(V)
    return J[mask], V[mask]

# ---------------------------------------------------------
# Linear 2-point interpolation for Jsc of multi-junction
# ---------------------------------------------------------
def interpolate_jsc(J, Vstack):
    # values around V=0
    idx_above = np.where(Vstack > 0)[0]
    idx_below = np.where(Vstack <= 0)[0]

    if len(idx_above) == 0 or len(idx_below) == 0:
        return np.nan

    i1 = idx_below[-1]     # largest negative V
    i2 = idx_above[0]      # smallest positive V

    slope, intercept, _, _, _ = linregress([Vstack[i1], Vstack[i2]], [J[i1], J[i2]])
    return intercept  # at V=0

# ---------------------------------------------------------
# Extract solar cell parameters
# ---------------------------------------------------------
def extract_parameters(J, V):
    # Jsc
    idx_above = np.where(V > 0)[0]
    idx_below = np.where(V <= 0)[0]

    if len(idx_above) > 0 and len(idx_below) > 0:
        i1 = idx_below[-1]
        i2 = idx_above[0]
        slope, intercept, *_ = linregress([V[i1], V[i2]], [J[i1], J[i2]])
        Jsc = intercept
    else:
        Jsc = np.nan

    # Voc
    idx_above = np.where(J > 0)[0]
    idx_below = np.where(J <= 0)[0]
    if len(idx_above) > 0 and len(idx_below) > 0:
        j1 = idx_below[-1]
        j2 = idx_above[0]
        slope, intercept, *_ = linregress([J[j1], J[j2]], [V[j1], V[j2]])
        Voc = intercept
    else:
        Voc = np.nan

    # MPP
    P = J * V
    idx = np.argmax(P)
    Jmpp = J[idx]
    Vmpp = V[idx]
    PCE = P[idx]

    # FF
    if Voc != 0 and Jsc != 0:
        FF = (Vmpp * Jmpp) / (Voc * Jsc)
    else:
        FF = np.nan

    return Jsc, Voc, Jmpp, Vmpp, PCE, FF

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.title("Multijunction IV Calculator (1–4 Junctions)")

st.sidebar.header("Subcell Parameters")
num_cells = st.sidebar.selectbox("Number of Subcells", [1,2,3,4], index=1)

# Default values from user
default_params = [
    {"Jph":20, "J0":1e-11, "n":1.0,  "Rs":0.2, "Rsh":10000, "T":298},
    {"Jph":21, "J0":1e-15, "n":1.24, "Rs":0.2, "Rsh":1000,  "T":298},
    {"Jph":18, "J0":1e-12, "n":1.3,  "Rs":0.3, "Rsh":5000,  "T":298},
    {"Jph":17, "J0":1e-13, "n":1.4,  "Rs":0.3, "Rsh":8000,  "T":298},
]

params = []

for i in range(num_cells):
    st.sidebar.subheader(f"Subcell {i+1}")

    Jph = float(st.sidebar.text_input(f"Jph{i+1}", value=default_params[i]["Jph"]))
    J0  = float(st.sidebar.text_input(f"J0{i+1}",  value=default_params[i]["J0"]))
    n   = float(st.sidebar.text_input(f"n{i+1}",   value=default_params[i]["n"]))
    Rs  = float(st.sidebar.text_input(f"Rs{i+1}",  value=default_params[i]["Rs"]))
    Rsh = float(st.sidebar.text_input(f"Rsh{i+1}", value=default_params[i]["Rsh"]))
    T   = float(st.sidebar.text_input(f"T{i+1}",   value=default_params[i]["T"]))

    params.append(dict(Jph=Jph, J0=J0, n=n, Rs=Rs, Rsh=Rsh, T=T))

# ---------------------------------------------------------
# Compute IV curves
# ---------------------------------------------------------
J_all = []
V_all = []
for p in params:
    J, V = calculate_subcell_iv(**p)
    J_all.append(J)
    V_all.append(V)

# Common current grid
J_min = max(J.min() for J in J_all)
J_max = min(J.max() for J in J_all)
J_common = np.linspace(J_min, J_max, 600)

# Interpolate all subcells to common J grid
V_interp = [np.interp(J_common, J_all[i], V_all[i]) for i in range(num_cells)]

# Sum of voltages
V_stack = np.sum(V_interp, axis=0)

# ---------------------------------------------------------
# Compute parameters per subcell
# ---------------------------------------------------------
results = []

for i in range(num_cells):
    Jsc, Voc, Jmpp, Vmpp, PCE, FF = extract_parameters(J_all[i], V_all[i])
    results.append([f"Subcell {i+1}",
                    Jsc, Voc, Jmpp, Vmpp, PCE, FF])

# ---------------------------------------------------------
# Tandem parameters
# ---------------------------------------------------------
if num_cells > 1:
    Jsc_t = interpolate_jsc(J_common, V_stack)
    Voc_t = sum(results[i][2] for i in range(num_cells))

    # MPP
    P = J_common * V_stack
    idx = np.argmax(P)
    Jmpp_t = J_common[idx]
    Vmpp_t = V_stack[idx]
    PCE_t  = P[idx]

    FF_t = (Vmpp_t * Jmpp_t) / (Voc_t * Jsc_t)

    results.append(["Multijunction", Jsc_t, Voc_t, Jmpp_t, Vmpp_t, PCE_t, FF_t])

# ---------------------------------------------------------
# Display table
# ---------------------------------------------------------
df = pd.DataFrame(results, columns=["Cell","Jsc","Voc","Jmpp","Vmpp","PCE","FF"])
df_display = df.copy()

fmt_2 = ["Jsc","Jmpp","PCE"]
fmt_2v = ["Voc","Vmpp"]

for col in fmt_2:
    df_display[col] = df_display[col].map(lambda x: f"{x:.2f}")
for col in fmt_2v:
    df_display[col] = df_display[col].map(lambda x: f"{x:.2f}")

st.subheader("Results")
st.dataframe(df_display)

# ---------------------------------------------------------
# Export
# ---------------------------------------------------------
st.subheader("Export Data")
export_name = st.text_input("Export filename prefix:", "IV_export")

# -------- 1) Results Table TXT ----------
results_txt = df_display.to_string(index=False)

st.download_button(
    "Download Results (.txt)",
    data=results_txt,
    file_name=f"{export_name}_results.txt",
    mime="text/plain"
)

# -------- 2) IV Curve TXT ----------
iv_dict = {}

# Subcells: V_i, J_i
for i in range(num_cells):
    iv_dict[f"V_subcell{i+1}"] = V_all[i]
    iv_dict[f"J_subcell{i+1}"] = J_all[i]

# Multijunction
if num_cells > 1:
    iv_dict["V_multijunction"] = V_stack
    iv_dict["J_multijunction"] = J_common

iv_df = pd.DataFrame(iv_dict)
iv_txt = iv_df.to_string(index=False)

st.download_button(
    "Download IV (.txt)",
    data=iv_txt,
    file_name=f"{export_name}_IV.txt",
    mime="text/plain"
)

# -------- 3) Input Parameters TXT ----------
param_lines = []
for i, p in enumerate(params):
    param_lines.append(f"--- Subcell {i+1} ---")
    for key, value in p.items():
        param_lines.append(f"{key}: {value}")
    param_lines.append("")

param_txt = "\n".join(param_lines)

st.download_button(
    "Download Parameters (.txt)",
    data=param_txt,
    file_name=f"{export_name}_parameters.txt",
    mime="text/plain"
)
