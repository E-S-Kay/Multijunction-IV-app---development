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
    return np.exp(np.clip(x, -700, 700))

def diode_equation_V(V, J, cell):
    q = 1.602176634e-19
    k = 1.380649e-23
    arg = q * (V + J * cell["Rs"]) / (cell["n"] * k * cell["T"])
    exp_term = safe_exp(arg)
    return J - (cell["Jph"] - cell["J0"] * (exp_term - 1.0) - (V + J * cell["Rs"]) / cell["Rsh"])

def estimate_Voc(cell):
    try:
        sol = root_scalar(lambda V: diode_equation_V(V, 0.0, cell),
                          bracket=[-0.5, 2.0], method="bisect")
        if sol.converged:
            return sol.root
    except Exception:
        pass
    return 0.6

def calculate_iv(Jph_mA, J0_mA, n, Rs, Rsh, T, J_common):
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

    # Short-circuit current density at V=0
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
    try:
        if np.isnan(Jsc) or Jsc == 0 or Voc == 0:
            return np.nan
        return (Jmpp * Vmpp) / (Jsc * Voc)
    except Exception:
        return np.nan

def to_float(text, default=0.0):
    try:
        return float(text.strip().replace(",", "."))
    except Exception:
        return float(default)

def fmt(x, dec=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NaN"
    return f"{x:.{dec}f}"

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Multijunction IV Simulator", layout="centered")
st.title("Multijunction IV Curves & Parameter Sweep")

# Number of subcells
num_cells = st.sidebar.selectbox("Number of subcells", [1, 2, 3, 4], index=1)

# Colors
pastel_colors = ["#AFCBFF", "#FFCBAF", "#CBAFFF", "#AFFFCB"]
stack_color = "black"

# Subcell inputs
cells = []
for i in range(num_cells):
    Jph = to_float(st.text_input(f"Subcell {i+1}: Jph [mA/cm²]", value="30.0" if i==0 else "20.0", key=f"Jph{i}"))
    J0  = to_float(st.text_input(f"Subcell {i+1}: J0 [mA/cm²]", value="1e-10" if i==0 else "1e-12", key=f"J0{i}"))
    n   = to_float(st.text_input(f"Subcell {i+1}: Ideality factor n", value="1.0", key=f"n{i}"))
    Rs  = to_float(st.text_input(f"Subcell {i+1}: Rs [Ω·cm²]", value="0.2", key=f"Rs{i}"))
    Rsh = to_float(st.text_input(f"Subcell {i+1}: Rsh [Ω·cm²]", value="1000.0", key=f"Rsh{i}"))
    T   = to_float(st.text_input(f"Subcell {i+1}: Temperature T [K]", value="298.0", key=f"T{i}"))
    cells.append({"Jph": Jph, "J0": J0, "n": n, "Rs": Rs, "Rsh": Rsh, "T": T})

# Common current
J_common = np.linspace(0.0, max([c["Jph"] for c in cells]), 800)

# Calculate IV curves
V_all, P_all, rows = [], [], []
for i, c in enumerate(cells):
    V, P, Voc, Vmpp, Jmpp, Pmpp, Jsc = calculate_iv(c["Jph"], c["J0"], c["n"], c["Rs"], c["Rsh"], c["T"], J_common)
    V_all.append(V)
    P_all.append(P)
    FF = calc_FF(Jsc, Voc, Jmpp, Vmpp)
    rows.append({"Label": f"Subcell {i+1}", "Jsc": Jsc, "Voc": Voc, "FF": FF,
                 "PCE": Pmpp, "Jmpp": Jmpp, "Vmpp": Vmpp, "color": pastel_colors[i]})

# Stack if >1
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
    rows.append({"Label": "Multijunction", "Jsc": Jsc_stack, "Voc": Voc_stack, "FF": FF_stack,
                 "PCE": P_mpp_stack, "Jmpp": J_mpp_stack, "Vmpp": V_mpp_stack, "color": "transparent"})

# -----------------------------
# Results Table
# -----------------------------
df = pd.DataFrame({
    "Cell": [r["Label"] for r in rows],
    "Jsc [mA/cm²]": [fmt(r["Jsc"],2) for r in rows],
    "Voc [V]": [fmt(r["Voc"],3) for r in rows],
    "FF [%]": [fmt(r["FF"]*100,2) if r["FF"] is not None else "NaN" for r in rows],
    "PCE [%]": [fmt(r["PCE"],2) for r in rows],
    "Jmpp [mA/cm²]": [fmt(r["Jmpp"],2) for r in rows],
    "Vmpp [V]": [fmt(r["Vmpp"],3) for r in rows],
    "color": [r["color"] for r in rows]
})
df_display = df.drop(columns=['color'])
row_colors = df['color'].tolist()
st.markdown("### Results Table")
def highlight_rows(row):
    color = row_colors[row.name]
    return ['background-color: {}'.format(color)] * len(row)
st.dataframe(df_display.style.apply(highlight_rows, axis=1).hide(axis="index"))

# -----------------------------
# IV Plot
# -----------------------------
fig = go.Figure()
for i, V in enumerate(V_all):
    fig.add_trace(go.Scatter(x=V, y=J_common, mode="lines",
                             name=f"Subcell {i+1}", line=dict(color=pastel_colors[i], width=2)))
if num_cells > 1:
    fig.add_trace(go.Scatter(x=V_stack, y=J_common, mode="lines", name="Multijunction",
                             line=dict(color=stack_color, width=4)))
    fig.add_trace(go.Scatter(x=[V_mpp_stack], y=[J_mpp_stack], mode="markers", name="Multijunction MPP",
                             marker=dict(color=stack_color, size=10, symbol="x")))

fig.update_layout(title="IV Curves", xaxis_title="Voltage [V]", yaxis_title="Current density [mA/cm²]",
                  hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Parameter Sweep
# -----------------------------
st.sidebar.markdown("### Parameter Sweep")
sweep_enable = st.sidebar.checkbox("Enable Sweep", value=False)
if sweep_enable:
    sweep_cell = st.sidebar.selectbox("Select Subcell for sweep", [f"Subcell {i+1}" for i in range(num_cells)])
    sweep_param = st.sidebar.selectbox("Select parameter to sweep", ["Jph","J0","n","Rs","Rsh","T"])
    sweep_min = to_float(st.sidebar.text_input("Sweep min value", "0.1"))
    sweep_max = to_float(st.sidebar.text_input("Sweep max value", "1.0"))
    sweep_steps = int(to_float(st.sidebar.text_input("Number of steps", "5")))
    sweep_values = np.linspace(sweep_min, sweep_max, sweep_steps)
    cell_idx = int(sweep_cell.split(" ")[1]) - 1
    st.markdown(f"### Sweep Results for {sweep_param} of {sweep_cell}")
    
    sweep_results = []
    for val in sweep_values:
        cells_sweep = [c.copy() for c in cells]
        cells_sweep[cell_idx][sweep_param] = val
        J_common = np.linspace(0.0, max([c["Jph"] for c in cells_sweep]), 800)
        V_all_s, P_all_s, rows_s = [], [], []
        for i, c in enumerate(cells_sweep):
            V, P, Voc, Vmpp, Jmpp, Pmpp, Jsc = calculate_iv(c["Jph"], c["J0"], c["n"], c["Rs"], c["Rsh"], c["T"], J_common)
            V_all_s.append(V)
            P_all_s.append(P)
            FF = calc_FF(Jsc, Voc, Jmpp, Vmpp)
            rows_s.append({"Label": f"Subcell {i+1}", "Jsc": Jsc, "Voc": Voc, "FF": FF,
                           "PCE": Pmpp, "Jmpp": Jmpp, "Vmpp": Vmpp})
        if num_cells > 1:
            V_stack = np.sum(np.vstack(V_all_s), axis=0)
            P_stack = V_stack * J_common
            idx_mpp_stack = int(np.nanargmax(P_stack))
            Voc_stack = float(V_stack[0])
            V_mpp_stack = float(V_stack[idx_mpp_stack])
            J_mpp_stack = float(J_common[idx_mpp_stack])
            P_mpp_stack = float(P_stack[idx_mpp_stack])
            Jsc_stack = interpolate_Jsc_two_points_linreg(V_stack, J_common)
            FF_stack = calc_FF(Jsc_stack, Voc_stack, J_mpp_stack, V_mpp_stack)
            rows_s.append({"Label": "Multijunction", "Jsc": Jsc_stack, "Voc": Voc_stack, "FF": FF_stack,
                           "PCE": P_mpp_stack, "Jmpp": J_mpp_stack, "Vmpp": V_mpp_stack})
        sweep_results.append({"sweep_value": val, "results": rows_s})
    
    # Anzeige Sweep-Tabelle
    for res in sweep_results:
        st.markdown(f"#### {sweep_param} = {res['sweep_value']}")
        df_sweep = pd.DataFrame({
            "Cell": [r["Label"] for r in res["results"]],
            "Jsc [mA/cm²]": [fmt(r["Jsc"],2) for r in res["results"]],
            "Voc [V]": [fmt(r["Voc"],3) for r in res["results"]],
            "FF [%]": [fmt(r["FF"]*100,2) if r["FF"] is not None else "NaN" for r in res["results"]],
            "PCE [%]": [fmt(r["PCE"],2) for r in res["results"]],
            "Jmpp [mA/cm²]": [fmt(r["Jmpp"],2) for r in res["results"]],
            "Vmpp [V]": [fmt(r["Vmpp"],3) for r in res["results"]]
        })
        st.dataframe(df_sweep)

# -----------------------------
# Download Options (.txt)
# -----------------------------
st.markdown("### Download Options")
base_filename = st.text_input("Base filename for export:", value="solar_simulation")

# Results Table
txt_results = df_display.to_csv(index=False, sep='\t').encode('utf-8')
st.download_button("Download Results Table (.txt)", data=txt_results,
                   file_name=f"{base_filename}_Results_Table.txt", mime="text/plain")

# IV Curves
iv_dict = {}
for i, V in enumerate(V_all):
    iv_dict[f"V{i+1} [V]"] = V
    iv_dict[f"J{i+1} [mA/cm²]"] = J_common
if num_cells > 1:
    iv_dict["Vstack [V]"] = V_stack
    iv_dict["Jstack [mA/cm²]"] = J_common
df_iv = pd.DataFrame(iv_dict)
txt_iv = df_iv.to_csv(index=False, sep='\t').encode('utf-8')
st.download_button("Download IV Curves (.txt)", data=txt_iv,
                   file_name=f"{base_filename}_IV_Curves.txt", mime="text/plain")

# Input parameters
input_list = []
for i, c in enumerate(cells):
    input_list.append({"Subcell": f"{i+1}", "Jph [mA/cm²]": c["Jph"], "J0 [mA/cm²]": c["J0"],
                       "n": c["n"], "Rs [Ω·cm²]": c["Rs"], "Rsh [Ω·cm²]": c["Rsh"], "T [K]": c["T"]})
df_input = pd.DataFrame(input_list)
txt_input = df_input.to_csv(index=False, sep='\t').encode('utf-8')
st.download_button("Download Input Parameters (.txt)", data=txt_input,
                   file_name=f"{base_filename}_Input_Parameters.txt", mime="text/plain")
