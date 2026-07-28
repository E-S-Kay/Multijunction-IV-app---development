import numpy as np
import streamlit as st
from scipy.special import lambertw
from scipy import stats
import plotly.graph_objects as go
import pandas as pd

# -----------------------------
# Fast Vectorized Lambert-W Solver
# -----------------------------
def vector_lambertw_exp(L):
    """Computes W(exp(L)) robustly for numpy arrays to avoid exponential overflow."""
    L = np.asarray(L, dtype=float)
    res = np.zeros_like(L)
    
    # Standard computation for safe range
    safe_mask = L <= 700
    if np.any(safe_mask):
        res[safe_mask] = np.real(lambertw(np.exp(L[safe_mask])))
        
    # Asymptotic approximation + Halley refinement for large numbers
    large_mask = ~safe_mask
    if np.any(large_mask):
        L_large = L[large_mask]
        w = L_large - np.log(L_large)
        for _ in range(2):
            f = w + np.log(w) - L_large
            f1 = (w + 1.0) / w
            f2 = -1.0 / (w**2)
            w -= f / (f1 - (f * f2) / (2.0 * f1))
        res[large_mask] = w
    return res

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
    """Generates light-to-dark shades for a given hex color."""
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

def calculate_iv_fast(Jph_mA, J0_mA, n, Rs, Rsh, T, J_common_mA):
    """Fully vectorized IV curve calculation using analytical Lambert W function."""
    q = 1.602176634e-19
    k = 1.380649e-23
    
    Jph = float(Jph_mA) / 1000.0  # A/cm²
    J0 = float(J0_mA) / 1000.0    # A/cm²
    Rs = float(Rs)                # Ω·cm²
    Rsh = float(Rsh)              # Ω·cm²
    n = float(n)
    T = float(T)
    
    Vt = n * k * T / q
    J = J_common_mA / 1000.0      # A/cm²
    
    term_j = Rsh * (Jph + J0 - J)
    log_arg = np.log(Rsh * J0 / Vt) + term_j / Vt
    W = vector_lambertw_exp(log_arg)
    
    V_vals = term_j - J * Rs - Vt * W
    P_plot = V_vals * J_common_mA
    
    idx_mpp = int(np.nanargmax(P_plot))
    Voc = float(V_vals[0])
    Vmpp = float(V_vals[idx_mpp])
    Jmpp = float(J_common_mA[idx_mpp])
    Pmpp = float(P_plot[idx_mpp])
    
    Jsc_val = interpolate_Jsc_two_points_linreg(V_vals, J_common_mA)
    
    return V_vals, P_plot, Voc, Vmpp, Jmpp, Pmpp, Jsc_val

# -----------------------------
# Cached Simulation Engine
# -----------------------------
@st.cache_data
def run_simulation(cells, sweep_enable, sweep_cell, sweep_param_key, sweep_values):
    max_jph = max([c["Jph"] for c in cells])
    J_common = np.linspace(0.0, max_jph, 800)

    results = []
    all_V_steps = []
    all_V_stack_steps = []

    for val in sweep_values:
        cells_current = [c.copy() for c in cells]
        if val is not None:
            cells_current[sweep_cell-1][sweep_param_key] = val
        
        V_all, P_all, rows = [], [], []
        for i, c in enumerate(cells_current):
            V, P, Voc, Vmpp, Jmpp, Pmpp, Jsc = calculate_iv_fast(
                c["Jph"], c["J0"], c["n"], c["Rs"], c["Rsh"], c["T"], J_common
            )
            V_all.append(V)
            P_all.append(P)
            FF = calc_FF(Jsc, Voc, Jmpp, Vmpp)
            rows.append({
                "Label": f"Subcell {i+1}",
                "Jsc [mA/cm²]": Jsc, 
                "Voc [V]": Voc, 
                "FF [%]": FF * 100.0 if not np.isnan(FF) else np.nan,
                "PCE [mW/cm²]": Pmpp, 
                "Jmpp [mA/cm²]": Jmpp, 
                "Vmpp [V]": Vmpp
            })
        
        all_V_steps.append(V_all)
        
        if len(cells) > 1:
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
                "Jsc [mA/cm²]": Jsc_stack, 
                "Voc [V]": Voc_stack, 
                "FF [%]": FF_stack * 100.0 if not np.isnan(FF_stack) else np.nan,
                "PCE [mW/cm²]": P_mpp_stack, 
                "Jmpp [mA/cm²]": J_mpp_stack, 
                "Vmpp [V]": V_mpp_stack
            })
            all_V_stack_steps.append(V_stack)

        for r in rows:
            r_copy = r.copy()
            if sweep_enable:
                r_copy["SweepValue"] = val
            results.append(r_copy)

    df_results = pd.DataFrame(results)
    return df_results, all_V_steps, all_V_stack_steps, J_common

# -----------------------------
# Dynamic Color Palette
# -----------------------------
plotly_colors = ["#87CEEB", "#FF7F50", "#98FF98", "#FFDAB9", "#FFD700", "E6E6FA"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Multijunction IV Simulator", layout="wide")
st.title("Multijunction Solar Cell IV Simulator with Sweep")

# Info & Credits Expander
with st.expander("ℹ️ About & Contact", expanded=False):
    st.markdown("""
    **What does this simulator do?** This simulator calculates and visualizes current-voltage (IV) curves and key performance parameters ($J_{sc}$, $V_{oc}$, $FF$, $PCE$) of multijunction solar cells and their individual subcells based on the extended single-diode model.

    **Key Features:**
    * Flexible configuration for 1 to 6 subcells.
    * Parameter sweeps for targeted analysis of individual cell parameters.
    * Interactive plot including Maximum Power Point (MPP) calculation.
    * Full export of simulation results and IV curve data including metadata headers.

    ---
    **Developed by:** Eike Köhnen (Helmholtz-Zentrum Berlin)  
    **Contact (bugs, improvements, feedback):** [eike.koehnen@helmholtz-berlin.de](mailto:eike.koehnen@helmholtz-berlin.de)
    """)

num_cells = st.sidebar.selectbox("Number of subcells", [1, 2, 3, 4, 5, 6], index=1)

# Default values for up to 6 subcells
default_jph = ["30.0", "20.0", "15.0", "12.0", "10.0", "8.0"]
default_j0 = ["1e-10", "1e-12", "1e-14", "1e-16", "1e-18", "1e-20"]

cells = []
for i in range(num_cells):
    color = plotly_colors[i % len(plotly_colors)]
    
    st.sidebar.markdown(
        f"""
        <div style="background-color: {color}; padding: 8px 12px; border-radius: 8px; margin-top: 15px; margin-bottom: 10px; color: #000000; font-weight: bold; font-size: 16px;">
            Subcell {i+1}
        </div>
        """, 
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

param_options = {
    "Jph": {"label": "Jph [mA/cm²]", "key": "Jph"},
    "J0": {"label": "J0 [mA/cm²]", "key": "J0"},
    "n": {"label": "Ideality factor n [-]", "key": "n"},
    "Rs": {"label": "Rs [Ω·cm²]", "key": "Rs"},
    "Rsh": {"label": "Rsh [Ω·cm²]", "key": "Rsh"},
    "T": {"label": "Temperature T [K]", "key": "T"}
}

if sweep_enable:
    sweep_cell = st.sidebar.selectbox("Select Subcell to sweep", list(range(1, num_cells+1)))
    
    selected_param_label = st.sidebar.selectbox("Parameter to sweep", list(param_options.keys()), format_func=lambda x: param_options[x]["label"])
    sweep_param_key = param_options[selected_param_label]["key"]
    sweep_param_display = param_options[selected_param_label]["label"]
    
    sweep_min = st.sidebar.number_input("Min value", value=float(cells[sweep_cell-1][sweep_param_key]))
    sweep_max = st.sidebar.number_input("Max value", value=float(cells[sweep_cell-1][sweep_param_key]))
    sweep_steps = st.sidebar.number_input("Number of steps", value=5, min_value=2)
    sweep_values = list(np.linspace(sweep_min, sweep_max, int(sweep_steps)))
else:
    sweep_cell = 1
    sweep_param_key = "Jph"
    sweep_param_display = "Jph [mA/cm²]"
    sweep_values = [None]

# -----------------------------
# Run Simulation
# -----------------------------
df_results, all_V_steps, all_V_stack_steps, J_common = run_simulation(
    cells, sweep_enable, sweep_cell, sweep_param_key, sweep_values
)

if sweep_enable and "SweepValue" in df_results.columns:
    df_results = df_results.rename(columns={"SweepValue": f"SweepValue ({sweep_param_display})"})

# -----------------------------
# Display Table
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

# Mapping der Dezimalstellen für die exakte Formatierung in der UI
rounding_dict = {
    "Jsc [mA/cm²]": 2,
    "Voc [V]": 3,
    "FF [%]": 2,
    "PCE [mW/cm²]": 2,
    "Jmpp [mA/cm²]": 2,
    "Vmpp [V]": 2
}

df_results_display = df_results.copy()
for col, decimals in rounding_dict.items():
    if col in df_results_display.columns:
        df_results_display[col] = df_results_display[col].round(decimals)

if sweep_enable and f"SweepValue ({sweep_param_display})" in df_results_display.columns:
    df_results_display[f"SweepValue ({sweep_param_display})"] = df_results_display[f"SweepValue ({sweep_param_display})"].round(3)

st.dataframe(df_results_display.style.apply(style_table, axis=None), use_container_width=True)

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
                    name=f"Subcell {i+1} ({sweep_param_display}={val:.2g})",
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
                name=f"Multijunction ({sweep_param_display}={val:.2g})",
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

fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")

max_v = 0.0
for V_step in all_V_steps:
    for v_arr in V_step:
        max_v = max(max_v, np.nanmax(v_arr))
if num_cells > 1:
    for v_st in all_V_stack_steps:
        max_v = max(max_v, np.nanmax(v_st))

fig.update_layout(
    xaxis_title="Voltage V [V]",
    yaxis_title="Current Density J [mA/cm²]",
    template="plotly_white",
    margin=dict(l=20, r=20, t=30, b=20),
    height=500,
)
fig.update_xaxes(range=[-0.1, max_v * 1.05])

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Download Options
# -----------------------------
st.markdown("### Download Options")
base_filename = st.text_input("Base filename for export:", value="solar_simulation")

param_header = "# ==========================================\n"
param_header += "# Simulation Input Parameters\n"
param_header += "# Developed by: Eike Köhnen (Helmholtz-Zentrum Berlin)\n"
param_header += "# Contact: eike.koehnen@helmholtz-berlin.de\n"
param_header += "# ==========================================\n"
for i, c in enumerate(cells):
    param_header += f"# Subcell {i+1}: Jph={c['Jph']} mA/cm², J0={c['J0']} mA/cm², n={c['n']}, Rs={c['Rs']} Ω·cm², Rsh={c['Rsh']} Ω·cm², T={c['T']} K\n"
if sweep_enable and len(sweep_values) > 1:
    param_header += f"# Sweep Configuration: Subcell {sweep_cell}, Parameter={sweep_param_display}, Min={sweep_min}, Max={sweep_max}, Steps={int(sweep_steps)}\n"
param_header += "# ==========================================\n\n"

txt_results_content = param_header + df_results.to_csv(index=False, sep='\t')
txt_results = txt_results_content.encode('utf-8')
st.download_button("Download Results Table (.txt)", data=txt_results, file_name=f"{base_filename}_Results_Table.txt", mime="text/plain")

iv_dict = {}
if sweep_enable and len(sweep_values) > 1:
    for step_i, val in enumerate(sweep_values):
        val_str = f"{sweep_param_display}={val:.2g}"
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
