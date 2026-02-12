#%% 
#Setup
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from a_my_utilities import set_chini_dataset, calc_biogas_production_rate, load_ch4_emissions_with_ad_only, calculate_production_normalized_ch4
import matplotlib.ticker as mtick
from scipy.stats import linregress

chini_data = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))  # or csv

set_chini_dataset(
    chini_data,
    x_col="flow_m3_per_day",
    y_col="methane_gen_kgh",
    drop_negative=True
)
################# AD FACILIITIES WITH REPORTED FLOW RATE (M3/DAY) ##########################


measurement_data_ad = calculate_production_normalized_ch4(
    load_data_func=load_ch4_emissions_with_ad_only,
    calc_biogas_func=calc_biogas_production_rate
)

# Filter measurement data to ensure values are > 0 and not NaN
measurement_data_ad = measurement_data_ad[
    (measurement_data_ad['flow_m3_per_day'] > 0) &
    (measurement_data_ad['production_normalized_CH4_percent']> 0)]

# Plotting

# Custom group labels
# Uncomment for color based on source data (also uncomment hue='source_group' below)
# measurement_data_ad['source_group'] = (
#     measurement_data_ad['source']
#     .fillna('')  # avoid errors if source is NaN
#     .apply(lambda x: (
#         'Moore et al., 2023' if 'Moore' in x
#         else 'Fredenslund et al., 2023' if 'Fredenslund et al., 2023' in x
#         else 'Song et al., 2023 (compilation)'
#     ))
# )

# Custom palette
# palette = {
#     'Moore et al., 2023': '#E24A33',
#     'Fredenslund et al., 2023': '#226f90',
#     'Song et al., 2023 (compilation)': '#999999',
# }

# Fit data mask once (positive + finite)
x = measurement_data_ad['flow_m3_per_day'].to_numpy()
y = measurement_data_ad['production_normalized_CH4_percent'].to_numpy()
mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
x, y = x[mask], y[mask]

# Power-law fit in log–log space
logx, logy = np.log(x), np.log(y)
res = linregress(logx, logy)
b = float(res.slope)
a = float(np.exp(res.intercept))
r2 = float(res.rvalue**2)

# Smooth x for plotting the fitted curve
xfit = np.geomspace(x.min(), x.max(), 200)
yfit = a * xfit**b

# Helper for nice scientific notation in legend
def _sci_tex(num, precision=2):
    s = f"{num:.{precision}e}"
    base, exp = s.split("e")
    return rf"{base} \times 10^{{{int(exp)}}}"

# Build legend label in percent units (y-axis is percent)
a_tex_pct = _sci_tex(100 * a, 2)  # label only; model unchanged
eqn = rf"$y(\%) = {a_tex_pct}\,x^{{{b:.2f}}}$ (R$^2$={r2:.2f})"

# ---------- Plot ----------
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=measurement_data_ad,
    x='flow_m3_per_day',
    y='production_normalized_CH4_percent',
    edgecolor='k',
    s=80
)

plt.xscale('log')
plt.yscale('log')

# Plot the fitted trendline
plt.plot(xfit, yfit, lw=3.5, color="black", label=f"Trend: {eqn}")

# Percent formatter for y-axis
ax = plt.gca()
def percent_formatter(val, _):
    pct = val * 100
    if pct >= 1:
        return f"{pct:.0f}%"
    elif pct >= 0.1:
        return f"{pct:.1f}%"
    else:
        return f"{pct:.2f}%"
ax.yaxis.set_major_formatter(percent_formatter)

plt.xlabel("Flow (m³/day)")
plt.ylabel("Production Normalized CH₄ Emissions (%)")
plt.title("Production Normalized Methane Emissions vs. Flow Rate (AD Only)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

# Save & show (single figure)
save_path = pathlib.Path("03_figures", "Figure_2b.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()