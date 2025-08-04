#%% 
#Setup
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from a_my_utilities import load_ch4_emissions_data, calc_biogas_production_rate, load_ch4_emissions_with_ad_only
import matplotlib.ticker as mtick



# %%

################# ALL FACILIITIES ##########################
# Calculate production normalized flow data - all facilities
measurement_data = load_ch4_emissions_data()

# Filter and copy data for safety
measurement_data = measurement_data[
    (measurement_data['flow_m3_per_day'] > 0) &
    (measurement_data['ch4_kg_per_hr'] > 0)
].copy()


measurement_data['biogas_production_kgCH4_per_hr_mid'] = measurement_data['flow_m3_per_day'].apply(calc_biogas_production_rate)
measurement_data['production_normalized_CH4_percent'] = measurement_data['ch4_kg_per_hr']/measurement_data['biogas_production_kgCH4_per_hr_mid']

#TODO deal with uncertainty 

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


#%% 

# Label by has AD - all facilities


####### ALL FACILITIES - AD vs Non-AD ##########################################

# Load all facilities
measurement_data = load_ch4_emissions_data()

# Filter and copy data for safety
measurement_data = measurement_data[
    (measurement_data['flow_m3_per_day'] > 0) &
    (measurement_data['ch4_kg_per_hr'] > 0)
].copy()

# Calculate production-normalized emissions
measurement_data['biogas_production_kgCH4_per_hr_mid'] = measurement_data['flow_m3_per_day'].apply(calc_biogas_production_rate)
measurement_data['production_normalized_CH4_percent'] = (
    measurement_data['ch4_kg_per_hr'] / measurement_data['biogas_production_kgCH4_per_hr_mid']
)

# Map has_ad to readable group labels
measurement_data['ad_group'] = measurement_data['has_ad'].str.strip().str.lower().map({
    'yes': 'Anaerobic Digestion',
    'no': 'No Anaerobic Digesters'
})

# Custom palette for AD groups
palette = {
    'Anaerobic Digestion': '#1f77b4',
    'No Anaerobic Digesters': '#ff7f0e'
}

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=measurement_data,
    x='flow_m3_per_day',
    y='production_normalized_CH4_percent',
    hue='ad_group',
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
plt.title("Production Normalized CH₄ Emissions vs. Flow Rate by AD Status (Log-Log)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(title='Anaerobic Digestion')
plt.tight_layout()

# Save with updated filename
save_path = pathlib.Path("03_figures", "production_normalized_emissions_vs_flow_by_ad.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()


#%% 
# Calculate production normalized flow data - AD facilities only 

################# AD FACILIITIES ##########################

# Load only facilities with AD
measurement_data_ad = load_ch4_emissions_with_ad_only()


# Calculate production normalized flow data 

# Filter and copy data for safety
measurement_data_ad = measurement_data_ad[
    (measurement_data_ad['flow_m3_per_day'] > 0) &
    (measurement_data_ad['ch4_kg_per_hr'] > 0)
].copy()

# Compute derived values
measurement_data_ad['biogas_production_kgCH4_per_hr_mid'] = measurement_data_ad['flow_m3_per_day'].apply(calc_biogas_production_rate)
measurement_data_ad['production_normalized_CH4_percent'] = (
    measurement_data_ad['ch4_kg_per_hr'] /
    measurement_data_ad['biogas_production_kgCH4_per_hr_mid']
)

# TODO: deal with uncertainty 

# %%
# Plotting

# Custom group labels
measurement_data_ad['source_group'] = measurement_data_ad['source'].apply(
    lambda x: 'Moore et al., 2023' if 'Moore' in x else 'Song et al., 2023 (compilation)'
)

# Custom palette
palette = {
    'Moore et al., 2023': '#E24A33',
    'Song et al., 2023 (compilation)': '#999999'
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















#######################################################
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
