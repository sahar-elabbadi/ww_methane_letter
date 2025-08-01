#%% 
# Setup 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from a_my_utilities import load_ch4_emissions_data

measurement_data = load_ch4_emissions_data()
# %%

# Filter and copy data for safety
filtered = measurement_data[
    (measurement_data['flow_m3_per_day'] > 0) &
    (measurement_data['ch4_kg_per_hr'] > 0)
].copy()

# Custom group labels
filtered['source_group'] = filtered['source'].apply(
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
    data=filtered,
    x='flow_m3_per_day',
    y='ch4_kg_per_hr',
    hue='source_group',
    palette=palette,
    edgecolor='k',
    s=80
)

# Log scales
plt.xscale('log')
plt.yscale('log')

# Labels
plt.xlabel("Flow (m³/day)")
plt.ylabel("CH₄ Emissions (kg/hr)")
plt.title("Methane Emissions vs. Flow Rate by Source (Log-Log)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Legend with clean title
plt.legend(title='Source')
plt.tight_layout()
save_path = pathlib.Path("03_figures", "emissions_vs_flow.png")
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
plt.ylabel("CH₄ Emissions (kg/hr)")
plt.title("Methane Emissions vs. Flow Rate by Source")
plt.grid(True, linestyle='--', linewidth=0.5)

plt.legend(title='Source')
plt.tight_layout()
plt.show()

# %%
