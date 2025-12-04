import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy import stats

st.set_page_config(layout="wide")

# --------------------------------------------------------------------
# Helper: Diode equation
# --------------------------------------------------------------------
def diode_current(V, Jph, J0, n, Rs, Rp, T):
    k = 1.380649e-23
    q = 1.602176634e-19
    Vt = k * T / q
    return Jph - J0 * (np.exp((V + Rs * 0) / (n * Vt)) - 1) - (V / Rp)

def iv_curve(Jph, J0, n, Rs, Rp, T, points=400):
    V = np.linspace(0, 1.5, points)
    J = Jph - J0 * (np.exp((V + Jph*Rs) / (n * (1.380649e-23 * T / 1.602176634e-19))) - 1) - V / Rp
    return V, J

# --------------------------------------------------------------------
# Solve for Jsc (subcell) via diode equation at V=0
# --------------------------------------------------------------------
def solve_jsc_subcell(Jph, J0, n, Rs, Rp, T):
    def f(J):
        return Jph - J0 * (np.exp((0 + J * Rs) / (n * (1.380649e-23*T/1.602176634e-19))) - 1) - 0/Rp - J
    try:
        return brentq(f, -50, 50)
    except:
        return np.nan

# --------------------------------------------------------------------
# Solve for Voc (subcell)
# --------------------------------------------------------------------
def solve_voc_subcell(Jph, J0, n, Rs, Rp, T):
    def f(V):
        k = 1.380649e-23
        q = 1.602176634e-19
        Vt = k*T/q
        return Jph - J0 * (np.exp((V) / (n * Vt)) - 1) - V/Rp
    try:
        return brentq(f, 0, 2)
    except:
        return np.nan

# --------------------------------------------------------------------
# PCE, MPP
# --------------------------------------------------------------------
def compute_mpp(V, J):
    P = V * J
    idx = np.argmax(P)
    return V[idx], J[idx], P[idx]

# --------------------------------------------------------------------
# Tandem Jsc via 2-point linear interpolation
# --------------------------------------------------------------------
def jsc_tandem_interpolated(J, V_stack):
    sign = np.sign(V_stack)
    zeros = np.where(np.diff(sign))[0]
    if len(zeros) == 0:
        return np.nan
    i = zeros[0]
    slope, intercept, _, _, _ = stats.linregress([V_stack[i], V_stack[i+1]],
                                                 [J[i], J[i+1]])
    return intercept  # At V=0

# =====================================================================
# UI
# =====================================================================

st.title("Multi-Junction Solar Cell IV Simulator (1–4 Junctions)")

num_cells = st.sidebar.selectbox("Number of Junctions", [1, 2, 3, 4], index=1)

default_params = [
    # --- Subcell 1 defaults ---
    dict(Jph=20,   J0=1e-11, n=1.0,  Rs=0.2, Rp=10000, T=298),
    # --- Subcell 2 defaults ---
    dict(Jph=21,   J0=1e-15, n=1.24, Rs=0.2, Rp=1000,  T=298),
    # Defaults for subcell 3
    dict(Jph=18,   J0=1e-12, n=1.15, Rs=0.2, Rp=8000,  T=298),
    # Defaults for subcell 4
    dict(Jph=16,   J0=1e-13, n=1.20, Rs=0.2, Rp=5000,  T=298),
]

params = []
st.sidebar.markdown("### Subcell Parameters")
for i in range(num_cells):
    st.sidebar.markdown(f"#### Subcell {i+1}")
    d = default_params[i]

    Jph = float(st.sidebar.text_input(f"Jph{i+1} (mA/cm²)", value=str(d["Jph"])))
    J0  = float(st.sidebar.text_input(f"J0{i+1} (mA/cm²)",  value=str(d["J0"])))
    n   = float(st.sidebar.text_input(f"n{i+1}",            value=str(d["n"])))
    Rs  = float(st.sidebar.text_input(f"Rs{i+1} (Ω·cm²)",   value=str(d["Rs"])))
    Rp  = float(st.sidebar.text_input(f"Rp{i+1} (Ω·cm²)",   value=str(d["Rp"])))
    T   = float(st.sidebar.text_input(f"T{i+1} (K)",        value=str(d["T"])))

    params.append(dict(Jph=Jph, J0=J0, n=n, Rs=Rs, Rp=Rp, T=T))

# =====================================================================
# Simulate each subcell
# =====================================================================
V_all = []
J_all = []
results = []

for p in params:
    V, J = iv_curve(**p)
    V_all.append(V)
    J_all.append(J)

    Jsc = solve_jsc_subcell(**p)
    Voc = solve_voc_subcell(**p)
    Vmpp, Jmpp, Pmpp = compute_mpp(V, J)
    FF = (Vmpp * Jmpp) / (Voc * Jsc) if Voc != 0 else np.nan
    PCE = Vmpp * Jmpp / 100  # W/cm² on 100 mW/cm²

    results.append([Jsc, Voc, Vmpp, Jmpp, FF*100, PCE*100])

# =====================================================================
# Multi-junction stack
# =====================================================================
J_common = J_all[0]
for j in J_all[1:]:
    J_common = np.minimum(J_common, j)

V_stack = sum(V_all)

Jsc_tandem = jsc_tandem_interpolated(J_common, V_stack)
Vmpp_s, Jmpp_s, Pmpp_s = compute_mpp(V_stack, J_common)

Voc_sum = sum(r[1] for r in results)
FF_t = (Vmpp_s * Jmpp_s) / (Voc_sum * Jsc_tandem)
PCE_t = Vmpp_s * Jmpp_s / 100

results.append([
    Jsc_tandem,
    Voc_sum,
    Vmpp_s,
    Jmpp_s,
    FF_t*100,
    PCE_t*100
])

# =====================================================================
# Display table
# =====================================================================
df = pd.DataFrame(
    results,
    columns=["Jsc (mA/cm²)", "Voc (V)", "Vmpp (V)", "Jmpp (mA/cm²)", "FF (%)", "PCE (%)"]
)

labels = [f"Subcell {i+1}" for i in range(num_cells)] + ["Multi-Junction"]
df["Cell"] = labels
df = df[["Cell"] + df.columns[:-1].tolist()]

df_display = df.copy()
for col in ["Jsc (mA/cm²)", "Jmpp (mA/cm²)", "PCE (%)"]:
    df_display[col] = df_display[col].map(lambda x: f"{x:.2f}")
for col in ["Voc (V)", "Vmpp (V)", "FF (%)"]:
    df_display[col] = df_display[col].map(lambda x: f"{x:.2f}")

st.markdown("### Results")
st.dataframe(df_display)

# =====================================================================
# Export section
# =====================================================================
st.markdown("### Export Data")

default_name = "IV_export"
export_name = st.text_input("Export filename prefix:", value=default_name)

# ---- Results TXT ----
results_txt = df_display.to_string(index=False)

st.download_button(
    label="📄 Download Results Table (.txt)",
    data=results_txt,
    file_name=f"{export_name}_results.txt",
    mime="text/plain"
)

# ---- IV Curves TXT ----
iv_export_dict = {}
for i, V in enumerate(V_all):
    iv_export_dict[f"V_subcell{i+1} (V)"] = V
    iv_export_dict[f"J_subcell{i+1} (mA/cm²)"] = J_all[i]

iv_export_dict["V_multijunction (V)"] = V_stack
iv_export_dict["J_multijunction (mA/cm²)"] = J_common

df_iv_export = pd.DataFrame(iv_export_dict)
iv_txt = df_iv_export.to_string(index=False)

st.download_button(
    label="📄 Download IV Curves (.txt)",
    data=iv_txt,
    file_name=f"{export_name}_IV.txt",
    mime="text/plain"
)

# ---- Input Parameters TXT ----
param_output = []
for i, p in enumerate(params):
    param_output.append(f"--- Subcell {i+1} ---")
    param_output.append(f"Jph (mA/cm²): {p['Jph']}")
    param_output.append(f"J0 (mA/cm²): {p['J0']}")
    param_output.append(f"n: {p['n']}")
    param_output.append(f"Rs (Ω·cm²): {p['Rs']}")
    param_output.append(f"Rp (Ω·cm²): {p['Rp']}")
    param_output.append(f"T (K): {p['T']}")
    param_output.append("")

param_txt = "\n".join(param_output)

st.download_button(
    label="📄 Download Input Parameters (.txt)",
    data=param_txt,
    file_name=f"{export_name}_parameters.txt",
    mime="text/plain"
)
