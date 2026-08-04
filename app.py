import numpy as np
import streamlit as st
from scipy.special import lambertw
from scipy import stats
import plotly.graph_objects as go
import pandas as pd

# -----------------------------
# Fast Vectorized Lambert-W Solver
# -----------------------------
@st.cache_data
def vector_lambertw_exp(L):
    """Computes W(exp(L)) robustly for numpy arrays to avoid exponential overflow."""
    L = np.asarray(L, dtype=float)
    res = np.zeros_like(L)
    
    safe_mask = L <= 700
    if np.any(safe_mask):
        res[safe_mask] = np.real(lambertw(np.exp(L[safe_mask])))
        
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

@st.cache_data
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
# Optimized Cached Simulation Engine
# -----------------------------
@st.cache_data
def run_simulation(cells_tuple, sweep_enable, sweep_cell, sweep_param_key, sweep_values_tuple):
    # Convert tuples back to lists/arrays for internal usage (cache compatibility)
    cells = [dict(c) for c in cells_tuple]
    sweep_values = list(sweep_values_tuple)
    
    max_jph = max([c["Jph"] for c in cells])
    J_common = np.linspace(0.0, max_jph, 100)

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
                "Jsc [mA/cm²]": round(Jsc, 2) if not np.isnan(Jsc) else np.nan, 
                "Voc [V]": round(Voc, 3) if not np.isnan(Voc) else np.nan, 
                "FF [%]": round(FF * 100.0, 2) if not np.isnan(FF) else np.nan,
                "PCE [mW/cm²]": round(Pmpp, 2) if not np.isnan(Pmpp) else np.nan, 
                "Jmpp [mA/cm²]": round(Jmpp, 2) if not np.isnan(Jmpp) else np.nan, 
                "Vmpp [V]": round(Vmpp, 3) if not np.isnan(Vmpp) else np.nan
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
                "Jsc [mA/cm²]": round(Jsc_stack, 2) if not np.isnan(Jsc_stack) else np.nan, 
                "Voc [V]": round(Voc_stack, 3) if not np.isnan(Voc_stack) else np.nan, 
                "FF [%]": round(FF_stack * 100.0, 2) if not np.isnan(FF_stack) else np.nan,
                "PCE [mW/cm²]": round(P_mpp_stack, 2) if not np.isnan(P_mpp_stack) else np.nan, 
                "Jmpp [mA/cm²]": round(J_mpp_stack, 2) if not np.isnan(J_mpp_stack) else np.nan, 
                "Vmpp [V]": round(V_mpp_stack, 3) if not np.isnan(V_mpp_stack) else np.nan
            })
            all_V_stack_steps.append(V_stack)

        for r in rows:
            r_copy = r.copy()
            if sweep_enable:
                r_copy["SweepValue"] = round(val, 3) if val is not None else None
            results.append(r_copy)

    df_results = pd.DataFrame(results)
    return df_results, all_V_steps, all_V_stack_steps, J_common

# -----------------------------
# Dynamic Color Palette & UI setup
# -----------------------------
plotly_colors = ["#87CEEB", "#FF7F50", "#98FF98", "#FFDAB9", "#FFD700", "#E6E6FA"]

st.set_page_config(page_title="Multijunction IV Simulator", layout="wide")
st.title("Multijunction Solar Cell IV Simulator with Sweep")

# (Keep your sidebar inputs and sweep configuration code here...)
