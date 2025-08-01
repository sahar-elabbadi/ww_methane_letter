#%% 
#Setup
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from a_my_utilities import load_ch4_emissions_data, calc_biogas_production_rate
import matplotlib.ticker as mtick


measurement_data = load_ch4_emissions_data()

# %%

# Calculate production normalized flow data 

# Filter and copy data for safety
measurement_data = measurement_data[
    (measurement_data['flow_m3_per_day'] > 0) &
    (measurement_data['ch4_kg_per_hr'] > 0)
].copy()


measurement_data['biogas_production_kgCH4_per_hr_mid'] = measurement_data['flow_m3_per_day'].apply(calc_biogas_production_rate)
measurement_data['production_normalized_CH4_percent'] = measurement_data['ch4_kg_per_hr']/measurement_data['biogas_production_kgCH4_per_hr_mid']

#TODO deal with uncertainty 

#%%
# Plotting

# Custom group labels
measurement_data['source_group'] = measurement_data['source'].apply(
    lambda x: 'Moore et al., 2023' if 'Moore' in x else 'Song et al., 2023 (compilation)'
)

# Custom palette with updated label
palette = {
    'Moore et al., 2023': '#E24A33',
    'Song et al., 2023 (compilation)': '#999999'
}

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=measurement_data,
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

# Labels as percent
ax = plt.gca()  # get current axis

# Format y-axis as percent with log-scale values
ax.set_yscale('log')

# Custom formatter: 1 sig fig, no scientific notation
def percent_formatter(y, _):
    pct = y * 100
    if pct >= 1:
        return f"{pct:.0f}%"
    elif pct >= 0.1:
        return f"{pct:.1f}%"
    else:
        return f"{pct:.2f}%"

ax.yaxis.set_major_formatter(percent_formatter)

# Labels
plt.xlabel("Flow (m³/day)")
plt.ylabel("Production Normalized CH₄ Emissions (%)")
plt.title("Production Normalized Methane Emissions vs. Flow Rate by Source (Log-Log)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Legend with clean title
plt.legend(title='Source')
plt.tight_layout()
save_path = pathlib.Path("03_figures", "production_normalized_emissions_vs_flow.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()



# %%
# Linear scale plot 

import seaborn as sns
import matplotlib.pyplot as plt

# Filter and copy data (optional if already clean)
filtered = measurement_data[
    (measurement_data['flow_m3_per_day'] > 0) &
    (measurement_data['ch4_kg_per_hr'] > 0)
].copy()

# Create source group labels
filtered['source_group'] = filtered['source'].apply(
    lambda x: 'Moore et al., 2023' if 'Moore' in x else 'Song et al., 2023 (compilation)'
)

# Define color palette
palette = {
    'Moore et al., 2023': '#E24A33',
    'Song et al., 2023 (compilation)': '#999999'
}

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=filtered,
    x='flow_m3_per_day',
    y='ch4_kg_per_hr',
    hue='source_group',
    palette=palette,
    edgecolor='k',
    s=80
)

# No log scales here!
plt.xlabel("Flow (m³/day)")
plt.ylabel("Production Normalized CH₄ Emissions (%)")
plt.title("Methane Emissions vs. Flow Rate by Source")
plt.grid(True, linestyle='--', linewidth=0.5)

plt.legend(title='Source')
plt.tight_layout()
plt.show()

# %%
