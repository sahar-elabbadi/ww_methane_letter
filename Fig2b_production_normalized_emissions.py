#%% 
#Setup
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from a_my_utilities import load_ch4_emissions_data, calc_biogas_production_rate, load_ch4_emissions_with_ad_only
import matplotlib.ticker as mtick



# # %%

# ################# ALL FACILIITIES ##########################
# # Calculate production normalized flow data - all facilities
# measurement_data = load_ch4_emissions_data()

# # Filter and copy data for safety
# measurement_data = measurement_data[
#     (measurement_data['flow_m3_per_day'] > 0) &
#     (measurement_data['ch4_kg_per_hr'] > 0)
# ].copy()


# measurement_data['biogas_production_kgCH4_per_hr_mid'] = measurement_data['flow_m3_per_day'].apply(calc_biogas_production_rate)
# measurement_data['production_normalized_CH4_percent'] = measurement_data['ch4_kg_per_hr']/measurement_data['biogas_production_kgCH4_per_hr_mid']

# #TODO deal with uncertainty 

# # Plotting

# # Custom group labels
# measurement_data['source_group'] = measurement_data['source'].apply(
#     lambda x: 'Moore et al., 2023' if 'Moore' in x else 'Song et al., 2023 (compilation)'
# )

# # Custom palette with updated label
# palette = {
#     'Moore et al., 2023': '#E24A33',
#     'Song et al., 2023 (compilation)': '#999999'
# }

# # Plot
# plt.figure(figsize=(8, 6))
# sns.scatterplot(
#     data=measurement_data,
#     x='flow_m3_per_day',
#     y='production_normalized_CH4_percent',
#     hue='source_group',
#     palette=palette,
#     edgecolor='k',
#     s=80
# )

# # Log scales
# plt.xscale('log')
# plt.yscale('log')

# # Labels as percent
# ax = plt.gca()  # get current axis

# # Format y-axis as percent with log-scale values
# ax.set_yscale('log')

# # Custom formatter: 1 sig fig, no scientific notation
# def percent_formatter(y, _):
#     pct = y * 100
#     if pct >= 1:
#         return f"{pct:.0f}%"
#     elif pct >= 0.1:
#         return f"{pct:.1f}%"
#     else:
#         return f"{pct:.2f}%"

# ax.yaxis.set_major_formatter(percent_formatter)

# # Labels
# plt.xlabel("Flow (m³/day)")
# plt.ylabel("Production Normalized CH₄ Emissions (%)")
# plt.title("Production Normalized Methane Emissions vs. Flow Rate by Source (Log-Log)")
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# # Legend with clean title
# plt.legend(title='Source')
# plt.tight_layout()
# save_path = pathlib.Path("03_figures", "production_normalized_emissions_vs_flow.png")
# plt.savefig(save_path, dpi=300, bbox_inches='tight')
# plt.show()


#%% 

# Label by has AD - all facilities


# ####### ALL FACILITIES - AD vs Non-AD ##########################################

# # Load all facilities
# measurement_data = load_ch4_emissions_data()

# # Filter and copy data for safety
# measurement_data = measurement_data[
#     (measurement_data['flow_m3_per_day'] > 0) &
#     (measurement_data['ch4_kg_per_hr'] > 0)
# ].copy()

# # Calculate production-normalized emissions
# measurement_data['biogas_production_kgCH4_per_hr_mid'] = measurement_data['flow_m3_per_day'].apply(calc_biogas_production_rate)
# measurement_data['production_normalized_CH4_percent'] = (
#     measurement_data['ch4_kg_per_hr'] / measurement_data['biogas_production_kgCH4_per_hr_mid']
# )

# # Map has_ad to readable group labels
# measurement_data['ad_group'] = measurement_data['has_ad'].str.strip().str.lower().map({
#     'yes': 'Anaerobic Digestion',
#     'no': 'No Anaerobic Digesters'
# })

# # Custom palette for AD groups
# palette = {
#     'Anaerobic Digestion': '#1f77b4',
#     'No Anaerobic Digesters': '#ff7f0e'
# }

# # Plot
# plt.figure(figsize=(8, 6))
# sns.scatterplot(
#     data=measurement_data,
#     x='flow_m3_per_day',
#     y='production_normalized_CH4_percent',
#     hue='ad_group',
#     palette=palette,
#     edgecolor='k',
#     s=80
# )

# # Log scales
# plt.xscale('log')
# plt.yscale('log')

# # Format y-axis as percent
# ax = plt.gca()

# def percent_formatter(y, _):
#     pct = y * 100
#     if pct >= 1:
#         return f"{pct:.0f}%"
#     elif pct >= 0.1:
#         return f"{pct:.1f}%"
#     else:
#         return f"{pct:.2f}%"

# ax.yaxis.set_major_formatter(percent_formatter)

# # Labels and layout
# plt.xlabel("Flow (m³/day)")
# plt.ylabel("Production Normalized CH₄ Emissions (%)")
# plt.title("Production Normalized CH₄ Emissions vs. Flow Rate by AD Status (Log-Log)")
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.legend(title='Anaerobic Digestion')
# plt.tight_layout()

# # Save with updated filename
# save_path = pathlib.Path("03_figures", "production_normalized_emissions_vs_flow_by_ad.png")
# plt.savefig(save_path, dpi=300, bbox_inches='tight')
# plt.show()


#%% 
# Calculate production normalized flow data - AD facilities only 

#Setup
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from a_my_utilities import load_ch4_emissions_data, calc_biogas_production_rate, load_ch4_emissions_with_ad_only
import matplotlib.ticker as mtick


################# AD FACILIITIES ##########################

# # Load only facilities with AD
# measurement_data_ad = load_ch4_emissions_with_ad_only()
# print(measurement_data_ad.head())

# # Calculate production normalized flow data 

# # Filter and copy data for safety
# # measurement_data_ad = measurement_data_ad[
# #     (measurement_data_ad['flow_m3_per_day'] > 0) &
# #     (measurement_data_ad['ch4_kg_per_hr'] > 0)
# # ].copy()

# # Calculate biogas production and production-normalized emissions


# # 1) Ensure the measured production is numeric
# measurement_data_ad['biogas_measured_num'] = pd.to_numeric(
#     measurement_data_ad['biogas_production_kgCH4_per_hr'], errors='coerce'
# )

# # 2) Flag if measured data for biogas production is available 
# use_measured = (
#     measurement_data_ad['reported_biogas_production'].astype(str).str.lower().eq('yes')
#     & measurement_data_ad['biogas_measured_num'].notna()
#     & measurement_data_ad['biogas_measured_num'].gt(0)
# )

# # 3) Only calculate if flow data exists
# has_flow = measurement_data_ad['flow_m3_per_day'].notna()

# measurement_data_ad['calculated_biogas_production_kgCH4_per_hr'] = np.nan
# measurement_data_ad.loc[has_flow, 'calculated_biogas_production_kgCH4_per_hr'] = (
#     measurement_data_ad.loc[has_flow, 'flow_m3_per_day'].apply(calc_biogas_production_rate)
# )

# # 4) Choose measured if valid, else calculated
# measurement_data_ad['biogas_production_used_kgCH4_per_hr'] = (
#     measurement_data_ad['biogas_measured_num'].where(use_measured)
#     .combine_first(measurement_data_ad['calculated_biogas_production_kgCH4_per_hr'])
# )

# # 5) Calculate production-normalized CH4 (%), avoid divide-by-zero
# denom = measurement_data_ad['biogas_production_used_kgCH4_per_hr']
# denom = denom.where(denom > 0)  # non-positive → NaN
# measurement_data_ad['production_normalized_CH4_percent'] = (
#     measurement_data_ad['ch4_kg_per_hr'] / denom
# )

import numpy as np
import pandas as pd

def calculate_production_normalized_ch4(
    data: pd.DataFrame = None,
    load_data_func=None,
    calc_biogas_func=None
):
    """
    Calculate production-normalized CH4 emissions (% of biogas production).

    Parameters
    ----------
    data : pd.DataFrame, optional
        Input dataframe containing AD facility data. If None, will call `load_data_func()`.
    load_data_func : callable, optional
        Function to load the dataset if `data` is None.
    calc_biogas_func : callable
        Function to calculate biogas production rate from flow (required).

    Returns
    -------
    pd.DataFrame
        Dataframe with additional columns:
        - biogas_measured_num
        - calculated_biogas_production_kgCH4_per_hr
        - biogas_production_used_kgCH4_per_hr
        - production_normalized_CH4_percent
    """
    if data is None:
        if load_data_func is None:
            raise ValueError("Either `data` must be provided or `load_data_func` must be specified.")
        data = load_data_func()

    if calc_biogas_func is None:
        raise ValueError("`calc_biogas_func` must be provided.")

    df = data.copy()

    # 1) Ensure numeric measured biogas production
    df['biogas_measured_num'] = pd.to_numeric(
        df['biogas_production_kgCH4_per_hr'], errors='coerce'
    )

    # 2) Flag valid measured values
    use_measured = (
        df['reported_biogas_production'].astype(str).str.lower().eq('yes')
        & df['biogas_measured_num'].notna()
        & df['biogas_measured_num'].gt(0)
    )

    # 3) Check for flow data
    has_flow = df['flow_m3_per_day'].notna()

    # 4) Calculate from flow where available
    df['calculated_biogas_production_kgCH4_per_hr'] = np.nan
    df.loc[has_flow, 'calculated_biogas_production_kgCH4_per_hr'] = (
        df.loc[has_flow, 'flow_m3_per_day'].apply(calc_biogas_func)
    )

    # 5) Choose measured if valid, else calculated
    df['biogas_production_used_kgCH4_per_hr'] = (
        df['biogas_measured_num'].where(use_measured)
        .combine_first(df['calculated_biogas_production_kgCH4_per_hr'])
    )

    # 6) Calculate production-normalized CH4 (%)
    denom = df['biogas_production_used_kgCH4_per_hr'].where(lambda x: x > 0)
    df['production_normalized_CH4_percent'] = df['ch4_kg_per_hr'] / denom

    return df


measurement_data_ad = calculate_production_normalized_ch4(
    load_data_func=load_ch4_emissions_with_ad_only,
    calc_biogas_func=calc_biogas_production_rate
)

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















#######################################################
# %%
# # Linear scale plot 

# import seaborn as sns
# import matplotlib.pyplot as plt

# # Filter and copy data (optional if already clean)
# filtered = measurement_data[
#     (measurement_data['flow_m3_per_day'] > 0) &
#     (measurement_data['ch4_kg_per_hr'] > 0)
# ].copy()

# # Create source group labels
# filtered['source_group'] = filtered['source'].apply(
#     lambda x: 'Moore et al., 2023' if 'Moore' in x else 'Song et al., 2023 (compilation)'
# )

# # Define color palette
# palette = {
#     'Moore et al., 2023': '#E24A33',
#     'Song et al., 2023 (compilation)': '#999999'
# }

# # Plot
# plt.figure(figsize=(8, 6))
# sns.scatterplot(
#     data=filtered,
#     x='flow_m3_per_day',
#     y='ch4_kg_per_hr',
#     hue='source_group',
#     palette=palette,
#     edgecolor='k',
#     s=80
# )

# # No log scales here!
# plt.xlabel("Flow (m³/day)")
# plt.ylabel("Production Normalized CH₄ Emissions (%)")
# plt.title("Methane Emissions vs. Flow Rate by Source")
# plt.grid(True, linestyle='--', linewidth=0.5)

# plt.legend(title='Source')
# plt.tight_layout()
# plt.show()

# # %%
