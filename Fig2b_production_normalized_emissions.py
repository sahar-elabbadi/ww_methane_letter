#%% 
#Setup
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from a_my_utilities import calc_biogas_production_rate, load_ch4_emissions_with_ad_only, calculate_production_normalized_ch4
import matplotlib.ticker as mtick


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
measurement_data_ad['source_group'] = (
    measurement_data_ad['source']
    .fillna('')  # avoid errors if source is NaN
    .apply(lambda x: (
        'Moore et al., 2023' if 'Moore' in x
        else 'Fredenslund et al., 2023' if 'Fredenslund et al., 2023' in x
        else 'Song et al., 2023 (compilation)'
    ))
)

# Custom palette
palette = {
    'Moore et al., 2023': '#E24A33',
    'Fredenslund et al., 2023': '#226f90',
    'Song et al., 2023 (compilation)': '#999999',
}

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=measurement_data_ad,
    x='flow_m3_per_day',
    y='production_normalized_CH4_percent',
    hue='source_group',
    palette=palette,
    edgecolor='k',
    s=80
)

# Log scales
plt.xscale('log')
plt.yscale('log')

# Format y-axis as percent
ax = plt.gca()

def percent_formatter(y, _):
    pct = y * 100
    if pct >= 1:
        return f"{pct:.0f}%"
    elif pct >= 0.1:
        return f"{pct:.1f}%"
    else:
        return f"{pct:.2f}%"

ax.yaxis.set_major_formatter(percent_formatter)

# Labels and layout
plt.xlabel("Flow (m³/day)")
plt.ylabel("Production Normalized CH₄ Emissions (%)")
plt.title("Production Normalized Methane Emissions vs. Flow Rate (AD Only)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(title='Source')
plt.tight_layout()

# Save with AD-specific filename
save_path = pathlib.Path("03_figures", "production_normalized_emissions_vs_flow_AD_only.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

# %%
