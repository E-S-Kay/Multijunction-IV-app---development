import numpy as np
import streamlit as st
from scipy.optimize import root_scalar, fsolve
from scipy import stats
import plotly.graph_objects as go
import pandas as pd

# -----------------------------
# Helper functions
# -----------------------------
def safe_exp(x):
    # Avoid overflow of exp() for large arguments
    return np.exp(np.clip(x, -700, 700))

def diode_equation_V(V, J, cell):
    # Single-diode equation with series and shunt resistances
    q = 1.602176634e-19
    k = 1.380649e-23
    arg = q * (V + J * cell["Rs"]) / (cell["n"] * k * cell["T"])
    exp_term = safe_exp(arg)
    return J - (cell["Jph"] - cell["J0"] * (exp_term - 1.0) - (V + J * cell["Rs"]) / cell["Rsh"])

def estimate_Voc(cell):
    # Open-circuit voltage at J = 0 (solve for V)
    try:
        sol = root_scalar(lambda V: diode_equation_V(V, 0.0, cell),
                          bracket=[-0.5, 2.0], method="bisect")
        if sol.converged:
            return sol.root
    except Exception:
        pass
    # Robust fallback
    return 0.6

def calculate_iv(Jph_mA, J0_mA, n, Rs, Rsh, T, J_common):
    # Compute V(J) and power for a single subcell along the common current axis
    Jph = float(Jph_mA) / 1000.0
    J0  = float(J0_mA) / 1000.0
    cell = {"Jph": Jph, "J0": J0, "n": float(n), "Rs": float(Rs), "Rsh": float(Rsh), "T": float(T)}

    Voc = estimate_Voc(cell)
    V_vals = np.zeros_like(J_common, dtype=float)
    V_prev = Voc

    for i, JmA in enumerate(J_common):
        J = float(JmA) / 1000.0
        V_sol = None
        try:
            sol = root_scalar(lambda V: diode_equation_V(V, J, cell),
                              bracket=[-1.0, Voc + 1.5], method="bisect")
            if sol.converged:
                V_sol = sol.root
        except Exception:
            pass
        if V_sol is None:
            try:
                sol = fsolve(lambda V: diode_equation_V(V, J, cell), V_prev, maxfev=1000)
                V_sol = float(sol[0])
            except Exception:
                V_sol = float(V_prev)
        V_vals[i] = V_sol
        V_prev = V_sol

    P_plot = V_vals * J_common
    idx_mpp = int(np.nanargmax(P_plot))

    # Short-circuit current density at V=0 (solve for J)
    try:
        upper = max(1e-6, Jph_mA * 1.5)
        sol_j = root_scalar(lambda J: diode_equation_V(0.0, J/1000.0, cell),
                            bracket=[0.0, upper], method="bisect")
        Jsc_val = float(sol_j.root) if sol_j.converged else np.nan
    except Exception:
        Jsc_val = np.nan

    Vmpp = float(V_vals[idx_mpp])
    Jmpp = float(J_common[idx_mpp])
    Pmpp = float(P_plot[idx_mpp])

    return V_vals, P_plot, float(Voc), Vmpp, Jmpp, Pmpp, Jsc_val

def interpolate_Jsc_two_points_linreg(V, J):
    # Estimate J(V=0) by linear interpolation across the sign change
    V = np.asarray(V, dtype=float)
    J = np.asarray(J, dtype=float)
    if V.size < 2:
        return np.nan
    mask = ((V[:-1] <= 0.0) & (V[1:] >= 0.0)) | ((V[:-1] >= 0.0) & (V[1:] <= 0.0))
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return np.nan
    idx = int(idxs[0])
    V_pair = V[idx:idx+2]
    J_pair = J[idx:idx+2]
    if np.isclose(V_pair[0], V_pair[1]):
        return float(J_pair[0])
    slope, intercept, _, _, _ = stats.linregress(V_pair, J_pair)
    return float(intercept)

def calc_FF(Jsc, Voc, Jmpp, Vmpp):
    # Fill factor = Pmpp / (Jsc * Voc)
    try:
        if np.isnan(Jsc) or Jsc == 0 or Voc == 0:
            return np.nan
        return (Jmpp * Vmpp) / (Jsc * Voc)
    except Exception:
        return np.nan

def to_float(text, default=0.0):
    # Robust float parsing with comma/point handling
    try:
        return float(text.strip().replace(",", "."))
    except Exception:
        return float(default)

def fmt(x, dec=2):
    # Safe number formatting with fixed decimals
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NaN"
    return f"{x:.{dec}f}"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Multijunction IV Simulator", layout="centered")

st.title("IV Curves: 1–4 Subcells (Single-Diode Model)")

# --- Description (can also be loaded from README.md if you prefer) ---
st.markdown("""
This app simulates solar cell IV characteristics using the **single-diode model** with series and shunt resistances.  
For **multijunction** solar cells, the subcell voltages are **added** to form the overall IV curve (common current density).
""")

# Sidebar: number of subcells
num_cells = st.sidebar.selectbox("Number of subcells", [1, 2, 3, 4], index=1)

# Define colors
pastel_colors = ["#AFCBFF", "#FFCBAF", "#CBAFFF", "#AFFFCB"]  # Pastel colors for subcells
stack_color = "black"

# Subcell inputs
cells = []
for i in range(num_cells):
    with st.sidebar.container():
        # Background rectangle behind the input group (visual grouping)
        st.markdown(
            f"""
            <div style="background-color:{pastel_colors[i]};padding:10px;border-radius:8px;margin-bottom:10px">
                <strong>Subcell {i+1}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
        # Input parameters (standard Streamlit fields)
        Jph = to_float(st.text_input(f"Subcell {i+1}: Jph [mA/cm²]", value="30.0" if i == 0 else "20.0", key=f"Jph{i}"))
        J0 = to_float(st.text_input(f"Subcell {i+1}: J0 [mA/cm²]", value="1e-10" if i == 0 else "1e-12", key=f"J0{i}"))
        n = to_float(st.text_input(f"Subcell {i+1}: Ideality factor n", value="1.0", key=f"n{i}"))
        Rs = to_float(st.text_input(f"Subcell {i+1}: Rs [Ω·cm²]", value="0.2", key=f"Rs{i}"))
        Rsh = to_float(st.text_input(f"Subcell {i+1}: Rsh [Ω·cm²]", value="1000.0", key=f"Rsh{i}"))
        T = to_float(st.text_input(f"Subcell {i+1}: Temperature T [K]", value="298.0", key=f"T{i}"))
    cells.append({"Jph": Jph, "J0": J0, "n": n, "Rs": Rs, "Rsh": Rsh, "T": T})

# Common current axis (mA/cm²)
J_common = np.linspace(0.0, max([c["Jph"] for c in cells]), 800)

# Compute per subcell
V_all, P_all, rows = [], [], []
for i, c in enumerate(cells):
    V, P, Voc, Vmpp, Jmpp, Pmpp, Jsc = calculate_iv(c["Jph"], c["J0"], c["n"], c["Rs"], c["Rsh"], c["T"], J_common)
    V_all.append(V)
    P_all.append(P)
    FF = calc_FF(Jsc, Voc, Jmpp, Vmpp)
    rows.append({
        "Jsc": Jsc, "Voc": Voc, "FF": FF,
        "PCE": Pmpp, "Jmpp": Jmpp, "Vmpp": Vmpp,
        "color": pastel_colors[i],
        "Label": f"Subcell {i+1}"
    })

# Multijunction (sum of voltages) only if more than one subcell
if num_cells > 1:
    V_stack = np.sum(np.vstack(V_all), axis=0)
    P_stack = V_stack * J_common
    idx_mpp_stack = int(np.nanargmax(P_stack))
    Voc_stack = float(V_stack[0])
    V_mpp_stack = float(V_stack[idx_mpp_stack])
    J_mpp_stack = float(J_common[idx_mpp_stack])
    P_mpp_stack = float(P_stack[idx_mpp_stack])
    Jsc_stack = interpolate_Jsc_two_points_linreg(V_stack, J_common)
    FF_stack = calc_FF(Jsc_stack, Voc_stack, J_mpp_stack, V_mpp_stack)

    rows.append({
        "Jsc": Jsc_stack, "Voc": Voc_stack,
        "FF": FF_stack, "PCE": P_mpp_stack,
        "Jmpp": J_mpp_stack, "Vmpp": V_mpp_stack,
        "color": "transparent",     # Transparent background in the table
        "Label": "Multijunction"
    })

# -----------------------------
# Results table
# -----------------------------
df = pd.DataFrame({
    "Cell": [r["Label"] for r in rows],
    "Jsc [mA/cm²]": [fmt(r["Jsc"], 2) for r in rows],
    "Voc [V]": [fmt(r["Voc"], 3) for r in rows],
    "FF [%]": [fmt(r["FF"]*100.0, 2) if (r["FF"] is not None and not np.isnan(r["FF"])) else "NaN" for r in rows],
    "PCE [%]": [fmt(r["PCE"], 2) for r in rows],
    "Jmpp [mA/cm²]": [fmt(r["Jmpp"], 2) for r in rows],
    "Vmpp [V]": [fmt(r["Vmpp"], 3) for r in rows],
    "color": [r["color"] for r in rows]
})

df_display = df.drop(columns=['color'])
row_colors = df['color'].tolist()

def highlight_rows(row):
    color = row_colors[row.name]
    return ['background-color: {}'.format(color)] * len(row)

st.write("### Results")
st.dataframe(
    df_display.style.apply(highlight_rows, axis=1).hide(axis="index")
)

# -----------------------------
# Plot
# -----------------------------
fig = go.Figure()

# Subcells (pastel, thin)
for i, V in enumerate(V_all):
    fig.add_trace(go.Scatter(
        x=V, y=J_common, mode="lines",
        name=f"Subcell {i+1}",
        line=dict(color=pastel_colors[i], width=2)
    ))

# Multijunction (black, thick)
if num_cells > 1:
    fig.add_trace(go.Scatter(
        x=V_stack, y=J_common, mode="lines",
        name="Multijunction",
        line=dict(color=stack_color, width=4)
    ))
    fig.add_trace(go.Scatter(
        x=[V_mpp_stack], y=[J_mpp_stack], mode="markers",
        name="Multijunction MPP",
        marker=dict(color=stack_color, size=10, symbol="x")
    ))

# Axes helpers
fig.add_vline(x=0, line=dict(color="gray", dash="dash"))
fig.add_hline(y=0, line=dict(color="gray", dash="dash"))

fig.update_layout(
    title="IV Curves",
    xaxis_title="Voltage [V]",
    yaxis_title="Current density [mA/cm²]",
    hovermode="x unified"
)

# X-range: single-junction uses its own Voc, multijunction uses combined Voc
x_max = (rows[0]["Voc"] + 0.1) if num_cells == 1 else (rows[-1]["Voc"] + 0.1)  # rows[-1] is Multijunction when present
fig.update_xaxes(range=[-0.2, x_max])

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# About / Footer
# -----------------------------
st.markdown("---")
st.markdown(
    """
    **Developed by:** Eike Köhnen (Helmholtz-Zentrum Berlin)  
    **Contact (bugs, improvements, feedback):** [eike.koehnen@helmholtz-berlin.de](mailto:eike.koehnen@helmholtz-berlin.de)
    """
)
