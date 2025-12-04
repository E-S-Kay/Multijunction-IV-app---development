import numpy as np
import streamlit as st
from scipy.optimize import root_scalar, fsolve
from scipy import stats
import plotly.graph_objects as go
import pandas as pd

# ============================================================
# Helper functions
# ============================================================

def safe_exp(x):
    """Avoid overflow of exp()."""
    return np.exp(np.clip(x, -700, 700))

def diode_equation_V(V, J, cell):
    """Single-diode equation with series and shunt resistances."""
    q = 1.602176634e-19
    k = 1.380649e-23
    arg = q * (V + J * cell["Rs"]) / (cell["n"] * k * cell["T"])
    exp_term = safe_exp(arg)
    return J - (cell["Jph"] - cell["J0"] * (exp_term - 1.0) - (V + J * cell["Rs"]) / cell["Rsh"])

def estimate_Voc(cell):
    """Find Voc by solving J(V)=0."""
    try:
        sol = root_scalar(lambda V: diode_equation_V(V, 0.0, cell),
                          bracket=[-0.5, 2.0], method="bisect")
        if sol.converged:
            return sol.root
    except Exception:
        pass
    return 0.6  # fallback

def calculate_iv(Jph_mA, J0_mA, n, Rs, Rsh, T, J_common):
    """Calculate V(J) for a subcell using the diode equation."""
    Jph = float(Jph_mA)/1000
    J0  = float(J0_mA)/1000
    cell = {"Jph":Jph, "J0":J0, "n":float(n),
            "Rs":float(Rs), "Rsh":float(Rsh), "T":float(T)}

    # Voc
    Voc = estimate_Voc(cell)

    V_vals = np.zeros_like(J_common, dtype=float)
    V_prev = Voc

    for i, JmA in enumerate(J_common):
        J = float(JmA)/1000
        V_sol = None

        # try bisection
        try:
            sol = root_scalar(lambda V: diode_equation_V(V, J, cell),
                              bracket=[-1.0, Voc + 1.5], method="bisect")
            if sol.converged:
                V_sol = sol.root
        except:
            pass

        # fallback to fsolve
        if V_sol is None:
            try:
                sol = fsolve(lambda V: diode_equation_V(V, J, cell),
                             V_prev, maxfev=1000)
                V_sol = float(sol[0])
            except:
                V_sol = float(V_prev)

        V_vals[i] = V_sol
        V_prev = V_sol

    # P(J)
    P_plot = V_vals * J_common
    idx_mpp = int(np.nanargmax(P_plot))

    # Jsc: solve J(V=0)
    try:
        upper = max(1e-6, Jph_mA * 1.5)
        sol_j = root_scalar(lambda J: diode_equation_V(0.0, J/1000, cell),
                            bracket=[0.0, upper], method="bisect")
        Jsc_val = sol_j.root if sol_j.converged else np.nan
    except:
        Jsc_val = np.nan

    Vmpp = float(V_vals[idx_mpp])
    Jmpp = float(J_common[idx_mpp])
    Pmpp = float(P_plot[idx_mpp])

    return V_vals, P_plot, float(Voc), Vmpp, Jmpp, Pmpp, Jsc_val

def interpolate_Jsc_two_points_linreg(V, J):
    """Find zero-crossing current of multijunction curve using two-point linear regression."""
    V = np.asarray(V)
    J = np.asarray(J)

    mask = ((V[:-1] <= 0) & (V[1:] >= 0)) | ((V[:-1] >= 0) & (V[1:] <= 0))
    idxs = np.where(mask)[0]

    if idxs.size == 0:
        return np.nan

    i = idxs[0]
    V_pair = V[i:i+2]
    J_pair = J[i:i+2]

    slope, intercept, *_ = stats.linregress(V_pair, J_pair)
    return intercept  # J(V=0)

def calc_FF(Jsc, Voc, Jmpp, Vmpp):
    """Fill factor."""
    if Jsc is None or Voc is None:
        return np.nan
    if Jsc == 0 or Voc == 0:
        return np.nan
    return (Vmpp * Jmpp) / (Voc * Jsc)

def to_float(x, default=0.0):
    try:
        return float(x.replace(",", "."))  
    except:
        return default

def fmt(x, dec=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NaN"
    return f"{x:.{dec}f}"

# ============================================================
# UI setup
# ============================================================

st.set_page_config(page_title="Multijunction IV Simulator", layout="centered")
st.title("IV Curves: 1–4 Subcells (Single-Diode Model)")

st.markdown("""
Simulates 1–4 solar subcells using the **single-diode model**.  
Multijunction voltages are summed at a common current density.
""")

# number of subcells
num_cells = st.sidebar.selectbox("Number of subcells", [1,2,3,4], index=1)

# Colors
pastel = ["#AFCBFF", "#FFCBAF", "#CBAFFF", "#AFFFCB"]

# Collect input parameters
cells = []
params = []

for i in range(num_cells):
    st.sidebar.markdown(
        f"<div style='background:{pastel[i]};padding:8px;border-radius:6px'><b>Subcell {i+1}</b></div>",
        unsafe_allow_html=True
    )

    Jph = to_float(st.sidebar.text_input(f"Jph {i+1} [mA/cm²]", "30" if i==0 else "20"))
    J0  = to_float(st.sidebar.text_input(f"J0 {i+1} [mA/cm²]", "1e-10" if i==0 else "1e-12"))
    n   = to_float(st.sidebar.text_input(f"n {i+1}", "1.0"))
    Rs  = to_float(st.sidebar.text_input(f"Rs {i+1} [Ω·cm²]", "0.2"))
    Rsh = to_float(st.sidebar.text_input(f"Rsh {i+1} [Ω·cm²]", "1000"))
    T   = to_float(st.sidebar.text_input(f"T {i+1} [K]", "298"))

    cells.append({"Jph":Jph,"J0":J0,"n":n,"Rs":Rs,"Rsh":Rsh,"T":T})
    params.append({"Jph":Jph,"J0":J0,"n":n,"Rs":Rs,"Rsh":Rsh,"T":T})

# ============================================================
# IV calculation
# ============================================================

J_common = np.linspace(0, max([c["Jph"] for c in cells]), 800)

V_all = []
rows = []

for i,c in enumerate(cells):
    V, P, Voc, Vmpp, Jmpp, Pmpp, Jsc = calculate_iv(
        c["Jph"], c["J0"], c["n"], c["Rs"], c["Rsh"], c["T"],
        J_common
    )

    V_all.append(V)
    FF = calc_FF(Jsc, Voc, Jmpp, Vmpp)

    rows.append({
        "Label":f"Subcell {i+1}",
        "Jsc":Jsc, "Voc":Voc, "FF":FF,
        "PCE":Pmpp, "Jmpp":Jmpp, "Vmpp":Vmpp,
        "color":pastel[i]
    })

# Multijunction
if num_cells > 1:
    V_stack = np.sum(np.vstack(V_all), axis=0)
    P_stack = V_stack * J_common

    idx_mpp = int(np.nanargmax(P_stack))
    Voc_stack = float(V_stack[0])
    Vmpp_stack = float(V_stack[idx_mpp])
    Jmpp_stack = float(J_common[idx_mpp])
    Pmpp_stack = float(P_stack[idx_mpp])

    Jsc_stack = interpolate_Jsc_two_points_linreg(V_stack, J_common)
    FF_stack  = calc_FF(Jsc_stack, Voc_stack, Jmpp_stack, Vmpp_stack)

    rows.append({
        "Label":"Multijunction",
        "Jsc":Jsc_stack, "Voc":Voc_stack,
        "FF":FF_stack, "PCE":Pmpp_stack,
        "Jmpp":Jmpp_stack, "Vmpp":Vmpp_stack,
        "color":"white"
    })

# ============================================================
# Results table
# ============================================================

df = pd.DataFrame({
    "Cell":[r["Label"] for r in rows],
    "Jsc [mA/cm²]":[fmt(r["Jsc"],2) for r in rows],
    "Voc [V]":[fmt(r["Voc"],3) for r in rows],
    "FF [%]":[fmt(r["FF"]*100,2) if r["FF"]==r["FF"] else "NaN" for r in rows],
    "PCE (mW/cm²)":[fmt(r["PCE"],2) for r in rows],
    "Jmpp [mA/cm²]":[fmt(r["Jmpp"],2) for r in rows],
    "Vmpp [V]":[fmt(r["Vmpp"],3) for r in rows],
})

# Save color column separately
colors = [r["color"] for r in rows]

def highlight_row(row):
    return [f"background-color: {colors[row.name]}"] * len(row)

st.write("### Results")
st.dataframe(df.style.apply(highlight_row,axis=1).hide(axis="index"))

# ============================================================
# Plot
# ============================================================

fig = go.Figure()

# subcells
for i,V in enumerate(V_all):
    fig.add_trace(go.Scatter(
        x=V, y=J_common, mode="lines", name=f"Subcell {i+1}",
        line=dict(color=pastel[i], width=2)
    ))

# MJ
if num_cells > 1:
    fig.add_trace(go.Scatter(
        x=V_stack, y=J_common, mode="lines",
        name="Multijunction",
        line=dict(color="black",width=4)
    ))

fig.update_layout(
    title="IV Curves",
    xaxis_title="Voltage [V]",
    yaxis_title="Current density [mA/cm²]",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# EXPORT SECTION
# ============================================================

st.markdown("---")
st.markdown("## Export")

default_name = "IV_export"
export_name = st.text_input("Filename prefix (no extension):", value=default_name)

# --- Results table export ---
results_txt = df.to_string(index=False)
st.download_button(
    "📄 Download Results (.txt)",
    data=results_txt,
    file_name=f"{export_name}_results.txt",
    mime="text/plain"
)

# --- IV curves export ---
iv_data = {}
for i,V in enumerate(V_all):
    iv_data[f"V_subcell{i+1}(V)"] = V
    iv_data[f"J_subcell{i+1}(mA/cm²)"] = J_common

if num_cells > 1:
    iv_data[f"V_multijunction(V)"] = V_stack
    iv_data[f"J_multijunction(mA/cm²)"] = J_common

df_iv = pd.DataFrame(iv_data)
iv_txt = df_iv.to_string(index=False)

st.download_button(
    "📄 Download IV curves (.txt)",
    data=iv_txt,
    file_name=f"{export_name}_IV.txt",
    mime="text/plain"
)

# --- Input parameters export ---
param_lines = []
for i,p in enumerate(params):
    param_lines.append(f"--- Subcell {i+1} ---")
    for k,v in p.items():
        param_lines.append(f"{k}: {v}")
    param_lines.append("")

param_txt = "\n".join(param_lines)

st.download_button(
    "📄 Download Input Parameters (.txt)",
    data=param_txt,
    file_name=f"{export_name}_parameters.txt",
    mime="text/plain"
)

# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.markdown("""
Developed by **Eike Köhnen (HZB)**  
Contact: eike.koehnen@helmholtz-berlin.de
""")
