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
    J0 = float(J0_mA) / 1000.0
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

def generate_shades(hex_color, num_shades):
    """Generiert Nuancen von hell nach dunkel für eine Hex-Farbe."""
    if num_shades <= 1:
        return [hex_color]
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    shades = []
    for i in range(num_shades):
        factor = 0.35 + 0.65 * (i / (num_shades - 1))
        r_new = int(r * factor + 255 * (1 - factor))
        g_new = int(g * factor + 255 * (1 - factor))
        b_new = int(b * factor + 255 * (1 - factor))
        shades.append(f"rgb({r_new},{g_new},{b_new})")
    return shades

# -----------------------------
# Dynamic Color Palette (6 Colors for Subcells)
# -----------------------------
plotly_colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Multijunction IV Simulator", layout="centered")
st.title("Multijunction Solar Cell IV Simulator with Sweep")

num_cells = st.sidebar.selectbox("Number of subcells", [1, 2, 3, 4, 5, 6], index=1)

# Standardwerte für bis zu 6 Subzellen
default_jph = ["30.0", "20.0", "15.0", "12.0", "10.0", "8.0"]
default_j0 = ["1e-10", "1e-12", "1e-14", "1e-16", "1e-18", "1e-20"]

cells = []
for i in range(num_cells):
    color = plotly_colors[i % len(plotly_colors)]
    
    # Überschrift im exakten Plotly-Farbton mit farbigem Unterstrich
    st.sidebar.markdown(
        f"<h3 style='color: {color}; border-bottom: 2px solid {color}; padding-bottom: 4px; margin-top: 15px;'>"
        f"Subcell {i+1}</h3>", 
        unsafe_allow_html=True
    )

    jph_def = default_jph[i] if i < len(default_jph) else "10.0"
    j0_def = default_j0[i] if i < len(default_j0) else "1e-12"

    Jph = to_float(st.sidebar.text_input(f"Subcell {i+1}: Jph [mA/cm²]", jph_def, key=f"Jph{i}"))
    J0 = to_float(st.sidebar.text_input(f"Subcell {i+1}: J0 [mA/cm²]", j0_def, key=f"J0{i}"))
    n = to_float(st.sidebar.text_input(f"Subcell {i+1}: Ideality factor n", "1.0", key=f"n{i}"))
    Rs = to_float(st.sidebar.text_input(f"Subcell {i+1}: Rs [Ω·cm²]", "0.2", key=f"Rs{i}"))
    Rsh = to_float(st.sidebar.text_input(f"Subcell {i+1}: Rsh [Ω·cm²]", "1000.0", key=f"Rsh{i}"))
    T = to_float(st.sidebar.text_input(f"Subcell {i+1}: Temperature T [K]", "298.0", key=f"T{i}"))
    cells.append({"Jph": Jph, "J0": J0, "n": n, "Rs": Rs, "Rsh": Rsh, "T": T})

# -----------------------------
# Sweep Options
# -----------------------------
st.sidebar.markdown("## Parameter Sweep")
sweep_enable = st.sidebar.checkbox("Enable Sweep", value=False)

if sweep_enable:
    sweep_cell = st.sidebar.selectbox("Select Subcell to sweep", list(range(1, num_cells+1)))
    sweep_param = st.sidebar.selectbox("Parameter to sweep", ["Jph", "J0", "n", "Rs", "Rsh", "T"])
    sweep_min = st.sidebar.number_input("Min value", value=float(cells[sweep_cell-1][sweep_param]))
    sweep_max = st.sidebar.number_input("Max value", value=float(cells[sweep_cell-1][sweep_param]))
    sweep_steps = st.sidebar.number_input("Number of steps", value=5, min_value=2)
    sweep_values = np.linspace(sweep_min, sweep_max, int(sweep_steps))
else:
    sweep_values = [None]

# -----------------------------
# Simulation
# -----------------------------
J_common = np.linspace(0.0, max([c["Jph"] for c in cells]), 800)

results = []
all_V_steps = []
all_V_stack_steps = []

for val in sweep_values:
    cells_current = [c.copy() for c in cells]
    if val is not None:
        cells_current[sweep_cell-1][sweep_param] = val
    
    V_all, P_all, rows = [], [], []
    for i, c in enumerate(cells_current):
        V, P, Voc, Vmpp, Jmpp, Pmpp, Jsc = calculate_iv(c["Jph"], c["J0"], c["n"], c["Rs"], c["Rsh"], c["T"], J_common)
        V_all.append(V)
        P_all.append(P)
        FF = calc_FF(Jsc, Voc, Jmpp, Vmpp)
        rows.append({
            "Label": f"Subcell {i+1}",
            "Jsc": Jsc, "Voc": Voc, "FF": FF,
            "PCE": Pmpp, "Jmpp": Jmpp, "Vmpp": Vmpp
        })
    
    all_V_steps.append(V_all)
    
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
            "Label": "Multijunction",
            "Jsc": Jsc_stack, "Voc": Voc_stack, "FF": FF_stack,
            "PCE": P_mpp_stack, "Jmpp": J_mpp_stack, "Vmpp": V_mpp_stack
        })
        all_V_stack_steps.append(V_stack)

    for r in rows:
        r_copy = r.copy()
        r_copy["SweepValue"] = val if val is not None else np.nan
        results.append(r_copy)

df_results = pd.DataFrame(results)

# -----------------------------
# Display Table (mit dynamischen Farbnuancen im Sweep-Fall)
# -----------------------------
st.write("### Results")

def style_table(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    
    if sweep_enable and len(sweep_values) > 1:
        num_steps = len(sweep_values)
        swept_idx = sweep_cell - 1
        swept_base_color = plotly_colors[swept_idx % len(plotly_colors)]
        swept_shades = generate_shades(swept_base_color, num_steps)
        stack_shades = generate_shades("#000000", num_steps)
        rows_per_step = num_cells + (1 if num_cells > 1 else 0)

        for idx, row in df.iterrows():
            label = str(row["Label"])
            step_i = min(idx // rows_per_step, num_steps - 1)
            
            for i in range(num_cells):
                if label == f"Subcell {i+1}":
                    if i == swept_idx:
                        c = swept_shades[step_i]
                    else:
                        c = plotly_colors[i % len(plotly_colors)]
                    styles.loc[idx, "Label"] = f"color: {c}; font-weight: bold;"
                    break
            
            if label == "Multijunction":
                c = stack_shades[step_i]
                styles.loc[idx, "Label"] = f"color: {c}; font-weight: bold;"
    else:
        for idx, row in df.iterrows():
            label = str(row["Label"])
            for i in range(len(plotly_colors)):
                if label == f"Subcell {i+1}":
                    styles.loc[idx, "Label"] = f"color: {plotly_colors[i]}; font-weight: bold;"
                    break
            if label == "Multijunction":
                styles.loc[idx, "Label"] = "color: #000000; font-weight: bold;"

    return styles

st.dataframe(df_results.style.apply(style_table, axis=None))

# -----------------------------
# Plot
# -----------------------------
fig = go.Figure()

if sweep_enable and len(sweep_values) > 1:
    num_steps = len(sweep_values)
    swept_idx = sweep_cell - 1
    swept_base_color = plotly_colors[swept_idx % len(plotly_colors)]
    swept_shades = generate_shades(swept_base_color, num_steps)
    stack_shades = generate_shades("#000000", num_steps)

    for i in range(num_cells):
        if i == swept_idx:
            for step_i, val in enumerate(sweep_values):
                fig.add_trace(go.Scatter(
                    x=all_V_steps[step_i][i],
                    y=J_common,
                    mode="lines",
                    name=f"Subcell {i+1} ({sweep_param}={val:.2g})",
                    line=dict(color=swept_shades[step_i])
                ))
        else:
            fig.add_trace(go.Scatter(
                x=all_V_steps[0][i],
                y=J_common,
                mode="lines",
                name=f"Subcell {i+1}",
                line=dict(color=plotly_colors[i % len(plotly_colors)])
            ))

    if num_cells > 1:
        for step_i, val in enumerate(sweep_values):
            fig.add_trace(go.Scatter(
                x=all_V_stack_steps[step_i],
                y=J_common,
                mode="lines",
                name=f"Multijunction ({sweep_param}={val:.2g})",
                line=dict(color=stack_shades[step_i], width=2)
            ))

else:
    for i in range(num_cells):
        fig.add_trace(go.Scatter(
            x=all_V_steps[0][i],
            y=J_common,
            mode="lines",
            name=f"Subcell {i+1}",
            line=dict(color=plotly_colors[i % len(plotly_colors)])
        ))

    if num_cells > 1:
        fig.add_trace(go.Scatter(
            x=all_V_stack_steps[0],
            y=J_common,
            mode="lines",
            name="Multijunction",
            line=dict(color="black", width=3)
        ))

# Vertikale Linie bei X=0 und horizontale Linie bei Y=0
fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")

# Maximalen x-Wert dynamisch aus allen Daten berechnen
max_v = 0.0
for V_step in all_V_steps:
    for v_arr in V_step:
        max_v = max(max_v, np.nanmax(v_arr))
if num_cells > 1:
    for v_st in all_V_stack_steps:
        max_v = max(max_v, np.nanmax(v_st))

fig.update_xaxes(range=[-0.1, max_v * 1.05])

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Download Options (.txt mit Header-Parametern)
# -----------------------------
st.markdown("### Download Options")
base_filename = st.text_input("Base filename for export:", value="solar_simulation")

# Parameter-Header für die Exportdateien generieren
param_header = "# ==========================================\n"
param_header += "# Simulation Input Parameters\n"
param_header += "# ==========================================\n"
for i, c in enumerate(cells):
    param_header += f"# Subcell {i+1}: Jph={c['Jph']} mA/cm², J0={c['J0']} mA/cm², n={c['n']}, Rs={c['Rs']} Ω·cm², Rsh={c['Rsh']} Ω·cm², T={c['T']} K\n"
if sweep_enable and len(sweep_values) > 1:
    param_header += f"# Sweep Configuration: Subcell {sweep_cell}, Parameter={sweep_param}, Min={sweep_min}, Max={sweep_max}, Steps={int(sweep_steps)}\n"
param_header += "# ==========================================\n\n"

# 1. Results Table Export
txt_results_content = param_header + df_results.to_csv(index=False, sep='\t')
txt_results = txt_results_content.encode('utf-8')
st.download_button("Download Results Table (.txt)", data=txt_results, file_name=f"{base_filename}_Results_Table.txt", mime="text/plain")

# 2. IV Curves Export
iv_dict = {}
if sweep_enable and len(sweep_values) > 1:
    for step_i, val in enumerate(sweep_values):
        val_str = f"{sweep_param}={val:.2g}"
        for i in range(num_cells):
            iv_dict[f"V{i+1} ({val_str}) [V]"] = all_V_steps[step_i][i]
            iv_dict[f"J{i+1} ({val_str}) [mA/cm²]"] = J_common
        if num_cells > 1:
            iv_dict[f"Vmultijunction ({val_str}) [V]"] = all_V_stack_steps[step_i]
            iv_dict[f"Jmultijunction ({val_str}) [mA/cm²]"] = J_common
else:
    for i in range(num_cells):
        iv_dict[f"V{i+1} [V]"] = all_V_steps[0][i]
        iv_dict[f"J{i+1} [mA/cm²]"] = J_common
    if num_cells > 1:
        iv_dict["Vmultijunction [V]"] = all_V_stack_steps[0]
        iv_dict["Jmultijunction [mA/cm²]"] = J_common

df_iv = pd.DataFrame(iv_dict)
txt_iv_content = param_header + df_iv.to_csv(index=False, sep='\t')
txt_iv = txt_iv_content.encode('utf-8')
st.download_button("Download IV Curves (.txt)", data=txt_iv, file_name=f"{base_filename}_IV_Curves.txt", mime="text/plain")
