#%% 
# Setup 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from a_my_utilities import load_ch4_emissions_data

measurement_data = load_ch4_emissions_data()
# %%

############### Make log-log plot ###############

import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

def plot_emissions_vs_flow(
    data,
    group_col,
    group_label_map=None,
    group_label_func=None,
    palette=None,
    save_dir=pathlib.Path("03_figures"),
    title="Methane Emissions vs. Flow Rate by Group (Log-Log)"
):
    """
    Plots CH4 emissions vs flow rate, grouped by a specified column with custom labeling.

    Parameters:
        data (pd.DataFrame): Data with columns 'flow_m3_per_day', 'ch4_kg_per_hr', and the grouping column.
        group_col (str): Column to group data by (e.g. 'source', 'has_ad').
        group_label_map (dict, optional): Dict mapping raw values to display labels.
        group_label_func (function, optional): Function that maps raw values to display labels.
        palette (dict): Dict mapping display labels to colors.
        save_dir (Path): Directory to save the plot image.
        title (str): Title of the plot.
    """
    # Filter valid data and entries where flow or emissions is NaN 
    filtered = data[
        (data['flow_m3_per_day'] > 0) &
        (data['ch4_kg_per_hr'] > 0)
    ].copy()

    # Apply group label function or mapping
    if group_label_func is not None:
        filtered['group'] = filtered[group_col].apply(group_label_func)
    elif group_label_map is not None:
        filtered['group'] = filtered[group_col].map(group_label_map)
    else:
        filtered['group'] = filtered[group_col]

    # Drop any unmatched
    filtered = filtered.dropna(subset=['group'])

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=filtered,
        x='flow_m3_per_day',
        y='ch4_kg_per_hr',
        hue='group',
        palette=palette,
        edgecolor='k',
        s=80
    )

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Flow (m³/day)")
    plt.ylabel("CH₄ Emissions (kg/hr)")
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title=group_col.replace('_', ' ').capitalize())
    plt.tight_layout()

    # Save
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"emissions_vs_flow_by_{group_col}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Make plot with labels based on whether AD is present
plot_emissions_vs_flow(
    data=measurement_data,
    group_col='has_ad',
    group_label_map={
        'yes': 'Has AD',
        'no': 'No AD'
    },
    palette={
        'Has AD': '#1f77b4',
        'No AD': '#ff7f0e'
    },
    title="Methane Emissions by Anaerobic Digestion (Log-Log)"
)

# Make plot with labels based on data source 
plot_emissions_vs_flow(
    data=measurement_data,
    group_col='source',
    group_label_func=lambda x: (
        'Moore et al., 2023' if 'Moore' in x else 'Song et al., 2023 (compilation)'
    ),
    palette={
        'Moore et al., 2023': '#E24A33',
        'Song et al., 2023 (compilation)': '#999999'
    },
    title="Methane Emissions by Source (Log-Log)"
)


# %%
################ Linear scale plot ################
# Commenting out as log-log plot is better for visual display of data 

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
# plt.ylabel("CH₄ Emissions (kg/hr)")
# plt.title("Methane Emissions vs. Flow Rate by Source")
# plt.grid(True, linestyle='--', linewidth=0.5)

# plt.legend(title='Source')
# plt.tight_layout()
# plt.show()
