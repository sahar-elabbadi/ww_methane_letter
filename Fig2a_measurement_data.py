#%% 
# Setup 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from scipy.stats import linregress
import json
from a_my_utilities import load_ch4_emissions_data

measurement_data = load_ch4_emissions_data()


############### Make log-log plot ###############



################# FUNCTIONS FOR BEST FIT CURVES ###################

def _powerlaw_fit(x, y):
    """
    Fit y = a * x^b by linear regression on (log x, log y).
    Returns dict with a, b, r2, stderr, intercept_stderr.
    """
    logx = np.log(x)
    logy = np.log(y)
    res = linregress(logx, logy)  # slope=b, intercept=log(a)
    b = res.slope
    a = float(np.exp(res.intercept))
    r2 = float(res.rvalue**2)
    return {
        "model": "power",
        "a": a,
        "b": float(b),
        "r2_loglog": r2,
        "slope_stderr": float(res.stderr),
        "intercept_stderr": float(res.intercept_stderr)
    }

def _format_sci_tex(num, precision=2):
    """
    LaTeX-friendly scientific notation string: '1.23 × 10^{-4}'
    (for use inside math mode: $...$)
    """
    num = float(num)
    s = f"{num:.{precision}e}"   # e.g., '1.23e-04'
    base, exp = s.split("e")
    exp = int(exp)
    return rf"{base} \times 10^{{{exp}}}"
    

def save_coeffs_as_py(module_path, coeffs_dict, var_name="EMISSIONS_FLOW_COEFFS"):
    """
    Writes a small Python module with a top-level dict you can import.
    Example import: 
        from coefficients_emissions import EMISSIONS_FLOW_COEFFS
    """
    text = (
        "# Auto-generated coefficients module\n"
        f"{var_name} = {json.dumps(coeffs_dict, indent=2)}\n"
    )
    with open(module_path, "w", encoding="utf-8") as f:
        f.write(text)


################# FUNCTIONS FOR BEST FIT CURVES ###################


# --- plotting function (power-law only) ---
def plot_emissions_vs_flow(
    data,
    group_col,
    group_label_map=None,
    group_label_func=None,
    palette=None,
    linewidth=2,
    save_dir=pathlib.Path("03_figures"),
    title="Methane Emissions vs. Flow Rate by Group (Log-Log)",
    export_coeffs_py=pathlib.Path("coefficients_emissions.py"),
    export_coeffs_json=pathlib.Path("coefficients_emissions.json"),
    legend_precision=2
):
    """
    Plots CH4 vs Flow (log–log) with 3 power-law trendlines:
      - Has AD
      - No AD
      - All data
    Exports coefficients to JSON and an importable .py module.
    """
    # Filter valid rows (strictly positive for log space)
    filtered = data[
        (data['flow_m3_per_day'] > 0) &
        (data['ch4_kg_per_hr'] > 0)
    ].copy()

    # Apply group labels
    if group_label_func is not None:
        filtered['group'] = filtered[group_col].apply(group_label_func)
    elif group_label_map is not None:
        filtered['group'] = filtered[group_col].map(group_label_map)
    else:
        filtered['group'] = filtered[group_col]
    filtered = filtered.dropna(subset=['group'])

    # Scatter
    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(
        data=filtered,
        x='flow_m3_per_day',
        y='ch4_kg_per_hr',
        hue='group',
        palette=palette,
        edgecolor='k',
        s=80, 
        alpha=0.8
    )

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Flow (m³/day)")
    plt.ylabel("CH₄ Emissions (kg/hr)")
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    coeffs_out = {}

    def _color_for(label, fallback="black"):
        if palette and label in palette:
            return palette[label]
        return fallback

    def _fit_and_plot(x, y, label, color):
        fit = _powerlaw_fit(x, y)
        a, b, r2 = fit["a"], fit["b"], fit["r2_loglog"]

        # Smooth curve across the data span
        xv = np.asarray(x, dtype=float)
        xfit = np.geomspace(np.nanmin(xv[xv > 0]), np.nanmax(xv), 200)
        yfit = a * xfit**b

        # Legend with LaTeX superscripts and sci notation
        a_tex = _format_sci_tex(a, legend_precision)
        eqn = rf"$y = {a_tex}\,x^{{{b:.2f}}}$"
        r2_text = rf"$R^{2}={r2:.2f}$"
        ax.plot(xfit, yfit, linewidth=linewidth, color=color, label=f"Trend: {label} {eqn} ({r2_text})")

        return fit

    # 1) Has AD
    mask_ad = filtered['group'] == 'Has AD'
    if mask_ad.any():
        fit_ad = _fit_and_plot(
            filtered.loc[mask_ad, 'flow_m3_per_day'],
            filtered.loc[mask_ad, 'ch4_kg_per_hr'],
            "Has AD",
            _color_for('Has AD', '#1f77b4')
        )
        coeffs_out["Has AD"] = fit_ad

    # 2) No AD
    mask_no = filtered['group'] == 'No AD'
    if mask_no.any():
        fit_no = _fit_and_plot(
            filtered.loc[mask_no, 'flow_m3_per_day'],
            filtered.loc[mask_no, 'ch4_kg_per_hr'],
            "No AD",
            _color_for('No AD', '#ff7f0e')
        )
        coeffs_out["No AD"] = fit_no

    # 3) All
    fit_all = _fit_and_plot(
        filtered['flow_m3_per_day'],
        filtered['ch4_kg_per_hr'],
        "All data",
        "black"
    )
    coeffs_out["All"] = fit_all

    # Clean legend (avoid duplicate hue entries)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    new_h, new_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            new_h.append(h); new_l.append(l); seen.add(l)
    ax.legend(new_h, new_l, title=group_col.replace('_', ' ').capitalize())

    plt.tight_layout()

    # Save figure
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "Figure_2a.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Export coefficients
    export_blob = {
        "model": "power",
        "coefficients": coeffs_out
    }
    with open(export_coeffs_json, "w", encoding="utf-8") as f:
        json.dump(export_blob, f, indent=2)
    save_coeffs_as_py(export_coeffs_py, export_blob, var_name="EMISSIONS_FLOW_COEFFS")

    return export_blob

## TRY OUT ADDING BOX PLOTS

import matplotlib.gridspec as gridspec

def plot_emissions_vs_flow_with_boxplot(
    data,
    group_col,
    group_label_map=None,
    group_label_func=None,
    palette=None,
    linewidth=2,
    save_dir=pathlib.Path("03_figures"),
    title="Methane Emissions vs. Flow Rate by Group (Log-Log)",
    export_coeffs_py=pathlib.Path("coefficients_emissions.py"),
    export_coeffs_json=pathlib.Path("coefficients_emissions.json"),
    legend_precision=2
):
    """
    Plots CH4 vs Flow (log–log) with 3 power-law trendlines and adds 
    a side boxplot of emissions by group for distribution comparison.
    """

    # Filter valid rows (strictly positive for log space)
    filtered = data[
        (data['flow_m3_per_day'] > 0) &
        (data['ch4_kg_per_hr'] > 0)
    ].copy()

    # Apply group labels
    if group_label_func is not None:
        filtered['group'] = filtered[group_col].apply(group_label_func)
    elif group_label_map is not None:
        filtered['group'] = filtered[group_col].map(group_label_map)
    else:
        filtered['group'] = filtered[group_col]
    filtered = filtered.dropna(subset=['group'])

    # Setup figure with two columns (scatter + boxplot)
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)

    # --- Scatter + fits (left panel) ---
    ax_scatter = plt.subplot(gs[0])
    sns.scatterplot(
        data=filtered,
        x='flow_m3_per_day',
        y='ch4_kg_per_hr',
        hue='group',
        palette=palette,
        edgecolor='k',
        s=80,
        alpha=0.8,
        ax=ax_scatter
    )

    ax_scatter.set_xscale('log')
    ax_scatter.set_yscale('log')
    ax_scatter.set_xlabel("Flow (m³/day)")
    ax_scatter.set_ylabel("CH₄ Emissions (kg/hr)")
    ax_scatter.set_title(title)
    ax_scatter.grid(True, which='both', linestyle='--', linewidth=0.5)

    coeffs_out = {}

    def _color_for(label, fallback="black"):
        if palette and label in palette:
            return palette[label]
        return fallback

    def _fit_and_plot(x, y, label, color):
        fit = _powerlaw_fit(x, y)
        a, b, r2 = fit["a"], fit["b"], fit["r2_loglog"]

        # Smooth curve across the data span
        xv = np.asarray(x, dtype=float)
        xfit = np.geomspace(np.nanmin(xv[xv > 0]), np.nanmax(xv), 200)
        yfit = a * xfit**b

        # Legend with LaTeX superscripts and sci notation
        a_tex = _format_sci_tex(a, legend_precision)
        eqn = rf"$y = {a_tex}\,x^{{{b:.2f}}}$"
        r2_text = rf"$R^{2}={r2:.2f}$"
        ax_scatter.plot(xfit, yfit, linewidth=linewidth, color=color, label=f"Trend: {label} {eqn} ({r2_text})")

        return fit

    # 1) Has AD
    mask_ad = filtered['group'] == 'Has AD'
    if mask_ad.any():
        coeffs_out["Has AD"] = _fit_and_plot(
            filtered.loc[mask_ad, 'flow_m3_per_day'],
            filtered.loc[mask_ad, 'ch4_kg_per_hr'],
            "Has AD",
            _color_for('Has AD', '#1f77b4')
        )

    # 2) No AD
    mask_no = filtered['group'] == 'No AD'
    if mask_no.any():
        coeffs_out["No AD"] = _fit_and_plot(
            filtered.loc[mask_no, 'flow_m3_per_day'],
            filtered.loc[mask_no, 'ch4_kg_per_hr'],
            "No AD",
            _color_for('No AD', '#ff7f0e')
        )

    # 3) All
    coeffs_out["All"] = _fit_and_plot(
        filtered['flow_m3_per_day'],
        filtered['ch4_kg_per_hr'],
        "All data",
        "black"
    )

    # Clean legend
    handles, labels = ax_scatter.get_legend_handles_labels()
    seen = set()
    new_h, new_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            new_h.append(h); new_l.append(l); seen.add(l)
    ax_scatter.legend(new_h, new_l, title=group_col.replace('_', ' ').capitalize())

    # --- Boxplot (right panel, aligned y-axis) ---
    ax_box = plt.subplot(gs[1], sharey=ax_scatter)
    sns.boxplot(
        data=filtered,
        y='ch4_kg_per_hr',
        x='group',
        palette=palette,
        ax=ax_box
    )
    ax_box.set_yscale('log')
    ax_box.set_xlabel("")
    ax_box.set_ylabel("")
    ax_box.tick_params(axis='x', rotation=0)
    ax_box.tick_params(axis='y', which='both', left=False, labelleft=False)


    plt.tight_layout()

    # Save figure
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "Figure_2a_with_boxplot.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Export coefficients
    export_blob = {"model": "power", "coefficients": coeffs_out}
    with open(export_coeffs_json, "w", encoding="utf-8") as f:
        json.dump(export_blob, f, indent=2)
    save_coeffs_as_py(export_coeffs_py, export_blob, var_name="EMISSIONS_FLOW_COEFFS")

    return export_blob



coeffs = plot_emissions_vs_flow(
    data=measurement_data,
    group_col='has_ad',
    group_label_map={'yes': 'Has AD', 'no': 'No AD'},
    palette={'Has AD': '#1f77b4', 'No AD': '#ff7f0e'},
    title="Methane Emissions by Anaerobic Digestion (Log-Log)",
    linewidth=4,
    export_coeffs_py=pathlib.Path("coefficients_emissions.py"),
    export_coeffs_json=pathlib.Path("coefficients_emissions.json"),
    legend_precision=2
)



coeffs = plot_emissions_vs_flow_with_boxplot(
    data=measurement_data,
    group_col='has_ad',
    group_label_map={'yes': 'Has AD', 'no': 'No AD'},
    palette={'Has AD': '#1f77b4', 'No AD': '#ff7f0e'},
    title="Methane Emissions by Anaerobic Digestion (Log-Log)",
    linewidth=4,
    export_coeffs_py=pathlib.Path("coefficients_emissions.py"),
    export_coeffs_json=pathlib.Path("coefficients_emissions.json"),
    legend_precision=2
)




#%% 
## EXAMINE RESIDUALS 

# import numpy as np
# import matplotlib.pyplot as plt

# def plot_residuals(data, group_col='has_ad', group_label_map={'yes':'Has AD','no':'No AD'}):
#     # Filter valid
#     df = data[(data['flow_m3_per_day'] > 0) & (data['ch4_kg_per_hr'] > 0)].copy()
#     df['group'] = df[group_col].map(group_label_map).fillna(df[group_col])

#     groups = ['Has AD', 'No AD', 'All']

#     plt.figure(figsize=(10, 6))

#     for i, group in enumerate(groups, start=1):
#         if group == 'All':
#             subset = df
#         else:
#             subset = df[df['group'] == group]
#         if subset.empty:
#             continue

#         x = subset['flow_m3_per_day'].values
#         y = subset['ch4_kg_per_hr'].values

#         # Fit power law in log-log
#         logx, logy = np.log(x), np.log(y)
#         slope, intercept = np.polyfit(logx, logy, 1)
#         y_pred = intercept + slope*logx

#         residuals = logy - y_pred

#         plt.subplot(3, 1, i)
#         plt.scatter(logx, residuals, alpha=0.6)
#         plt.axhline(0, color='k', linestyle='--')
#         plt.xlabel("log(Flow)")
#         plt.ylabel("Residual") #(log(y) - log(y_pred))
#         plt.title(f"Residuals for {group}")

#     plt.tight_layout()
#     plt.show()

# # Example usage:
# plot_residuals(measurement_data)



#### OLD CODE WITHOUT REGRESSION #####
# def plot_emissions_vs_flow(
#     data,
#     group_col,
#     group_label_map=None,
#     group_label_func=None,
#     palette=None,
#     save_dir=pathlib.Path("03_figures"),
#     title="Methane Emissions vs. Flow Rate by Group (Log-Log)"
# ):
#     """
#     Plots CH4 emissions vs flow rate, grouped by a specified column with custom labeling.

#     Parameters:
#         data (pd.DataFrame): Data with columns 'flow_m3_per_day', 'ch4_kg_per_hr', and the grouping column.
#         group_col (str): Column to group data by (e.g. 'source', 'has_ad').
#         group_label_map (dict, optional): Dict mapping raw values to display labels.
#         group_label_func (function, optional): Function that maps raw values to display labels.
#         palette (dict): Dict mapping display labels to colors.
#         save_dir (Path): Directory to save the plot image.
#         title (str): Title of the plot.
#     """
#     # Filter valid data and entries where flow or emissions is NaN 
#     filtered = data[
#         (data['flow_m3_per_day'] > 0) &
#         (data['ch4_kg_per_hr'] > 0)
#     ].copy()

#     # Apply group label function or mapping
#     if group_label_func is not None:
#         filtered['group'] = filtered[group_col].apply(group_label_func)
#     elif group_label_map is not None:
#         filtered['group'] = filtered[group_col].map(group_label_map)
#     else:
#         filtered['group'] = filtered[group_col]

#     # Drop any unmatched
#     filtered = filtered.dropna(subset=['group'])

#     # Plot
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(
#         data=filtered,
#         x='flow_m3_per_day',
#         y='ch4_kg_per_hr',
#         hue='group',
#         palette=palette,
#         edgecolor='k',
#         s=80
#     )

#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlabel("Flow (m³/day)")
#     plt.ylabel("CH₄ Emissions (kg/hr)")
#     plt.title(title)
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.legend(title=group_col.replace('_', ' ').capitalize())
#     plt.tight_layout()

#     # Save
#     save_dir.mkdir(parents=True, exist_ok=True)
#     save_path = save_dir / f"Figure_2a.png"
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()

# # Make plot with labels based on whether AD is present
# plot_emissions_vs_flow(
#     data=measurement_data,
#     group_col='has_ad',
#     group_label_map={
#         'yes': 'Has AD',
#         'no': 'No AD'
#     },
#     palette={
#         'Has AD': '#1f77b4',
#         'No AD': '#ff7f0e'
#     },
#     title="Methane Emissions by Anaerobic Digestion (Log-Log)"
# )

# # # Make plot with labels based on data source 
# # plot_emissions_vs_flow(
# #     data=measurement_data,
# #     group_col='source',
# #     group_label_func=lambda x: (
# #         'Moore et al., 2023' if 'Moore' in x else 'Song et al., 2023 (compilation)'
# #     ),
# #     palette={
# #         'Moore et al., 2023': '#E24A33',
# #         'Song et al., 2023 (compilation)': '#999999'
# #     },
# #     title="Methane Emissions by Source (Log-Log)"
# # )


# # %%
# ################ Linear scale plot ################
# # Commenting out as log-log plot is better for visual display of data 

# # import seaborn as sns
# # import matplotlib.pyplot as plt

# # # Filter and copy data (optional if already clean)
# # filtered = measurement_data[
# #     (measurement_data['flow_m3_per_day'] > 0) &
# #     (measurement_data['ch4_kg_per_hr'] > 0)
# # ].copy()

# # # Create source group labels
# # filtered['source_group'] = filtered['source'].apply(
# #     lambda x: 'Moore et al., 2023' if 'Moore' in x else 'Song et al., 2023 (compilation)'
# # )

# # # Define color palette
# # palette = {
# #     'Moore et al., 2023': '#E24A33',
# #     'Song et al., 2023 (compilation)': '#999999'
# # }

# # # Plot
# # plt.figure(figsize=(8, 6))
# # sns.scatterplot(
# #     data=filtered,
# #     x='flow_m3_per_day',
# #     y='ch4_kg_per_hr',
# #     hue='source_group',
# #     palette=palette,
# #     edgecolor='k',
# #     s=80
# # )

# # # No log scales here!
# # plt.xlabel("Flow (m³/day)")
# # plt.ylabel("CH₄ Emissions (kg/hr)")
# # plt.title("Methane Emissions vs. Flow Rate by Source")
# # plt.grid(True, linestyle='--', linewidth=0.5)

# # plt.legend(title='Source')
# # plt.tight_layout()
# # plt.show()

# %%
