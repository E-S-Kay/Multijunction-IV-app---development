# Multijunction Solar Cell IV Simulator

This Streamlit app simulates the current-voltage (IV) characteristics of solar cells using the **single-diode model** with series and shunt resistances.  
For multijunction solar cells, the subcell voltages are **added** to obtain the overall IV curve.

---

## ✨ Features
- Simulate **1–4 subcells** individually and combined into a multijunction.
- Adjustable parameters for each subcell:
  - Photocurrent density (Jph)
  - Saturation current density (J0)
  - Ideality factor (n)
  - Series resistance (Rs)
  - Shunt resistance (Rsh)
  - Temperature (T)
- Interactive IV curve plotting with **Plotly**.
- Automatic extraction of key metrics:
  - Short-circuit current density (Jsc)
  - Open-circuit voltage (Voc)
  - Fill factor (FF)
  - Maximum power point (Jmpp, Vmpp, Pmpp)
  - Efficiency estimate (PCE)
- Clear result table with color-coded rows.

---

## 🚀 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/multijunction-iv-app.git
cd multijunction-iv-app
