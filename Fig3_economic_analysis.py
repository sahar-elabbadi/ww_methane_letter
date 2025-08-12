
#%% 

############### Vary capturable fraction instead of leak rate ###############
#Setup
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors



from a_my_utilities import calc_annual_savings


def plot_methane_savings_contour(
    biogas_production_rate,
    leak_fraction_capturable,
    electricity_price_per_kWh,
    ogi_cost=100000,
    plant_sizes_m3_per_day_range=(0, 1_200_000),
    leak_rates=np.linspace(0, 0.5, 200),
    resolution=200,
    fig=None,
    ax=None,
    save_path=None,
    levels_fill=None,
    levels_line=None, 
    cmap=None,
    norm=None
):
    """
    Returns a methane savings contour plot (Jianan-style) using user's methane functions.

    Parameters:
    - biogas_production_rate: MJ biogas / m³
    - leak_fraction_capturable: Fraction of leakage that is recoverable
    - electricity_price_per_kWh: $ per kWh
    - ogi_cost: Fixed cost of OGI survey
    - plant_sizes_m3_per_day_range: (min, max) plant size in m³/day
    - leak_rates: Array of leak rates (0 to 1)
    - resolution: Number of grid points along x-axis (plant size)
    - fig, ax: Optional Matplotlib figure and axis
    - save_path: Optional filepath to save plot

    Returns:
    - ax: The axis containing the contour plot
    """

    # Grid
    plant_sizes = np.linspace(*plant_sizes_m3_per_day_range, resolution)
    X, Y = np.meshgrid(plant_sizes, leak_rates)

    # Compute annual savings
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = calc_annual_savings(
                plant_size=X[i, j],
                biogas_production_rate=biogas_production_rate,
                leak_rate=Y[i, j],
                leak_fraction_capturable=leak_fraction_capturable,
                electricity_price_per_kWh=electricity_price_per_kWh,
                ogi_cost=ogi_cost
            )

    # Flatten for tricontourf
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    # Jianan-style color map
    b, g, y, o, r = "#60C1CF", "#79BF82", "#F3C354", "#F98F60", "#ED586F"
    color_map = mcolors.LinearSegmentedColormap.from_list('custom_map', [r, o, y, g, b])

    # Create fig/ax if needed
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7.3, 6))

    # Filled contour
    levels_fill = np.linspace(0, np.max(Z_flat), 100)
    
    # Filled contour — now uses shared norm so colors == same Z bins everywhere
    fills = ax.tricontourf(
        X_flat, Y_flat, Z_flat,
        levels=levels_fill,
        cmap=cmap,
        norm=norm
    )

    # Contour lines

    def label_formatter(x):
        if abs(x) >= 1e6:
            return f"${x/1e6:.0f}M"
        elif abs(x) >= 1e3:
            return f"${x/1e3:.0f}k"
        else:
            return f"${x:.0f}"
        
    contour_lines = ax.tricontour(
        X_flat, Y_flat, Z_flat, 
        levels=levels_line, 
        colors='black', linewidths=2.0)
    ax.clabel(contour_lines, 
    inline=True, fontsize=10, fmt=label_formatter, rightside_up=True)

    # Breakeven line
    breakeven = ax.tricontour(
        X_flat, Y_flat, Z_flat,
        levels=[0], colors='black', linewidths=3.0, linestyles='solid', zorder=3
    )
    # ax.clabel(breakeven, fmt={0: '$0'}, fontsize=12, colors='red')

    # Axes formatting
    ax.tick_params(direction='inout', length=15, width=1.5, pad=6)
    ax.set_xlabel("Plant Size (m³/day)", fontsize=16)
    ax.set_ylabel("Leak Rate (%)", fontsize=16)
    ax.set_xlim(plant_sizes_m3_per_day_range)
    xticks = np.arange(0, plant_sizes_m3_per_day_range[1] + 1, 200_000)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{val / 1e6:.1f}M" for val in xticks])
    ax.set_ylim(leak_rates.min(), leak_rates.max())
    ax.set_yticks(np.linspace(leak_rates.min(), leak_rates.max(), 6))
    ax.set_yticklabels([f"{val:.1f}" for val in np.linspace(leak_rates.min(), leak_rates.max(), 6)])
    ax.tick_params(labelsize=12)
    
    ax.set_title(f"Fraction of capturable gas: {params['leak_fraction_capturable']}, Electricity price: ${params['electricity_price_per_kWh']}/kWh")


    # Save plot if needed
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return ax


# ======================
# SHARED SCALE SETTINGS
# ======================
vmin = -100_000
vmax = 8_000_000
levels_fill = np.linspace(vmin, vmax, 100)   # same as before
levels_line = np.linspace(vmin, vmax, 10)    # same as before

# Shared cmap + BoundaryNorm to lock colors to identical Z bins
b, g, y, o, r = "#60C1CF", "#79BF82", "#F3C354", "#F98F60", "#ED586F"
shared_cmap = mcolors.LinearSegmentedColormap.from_list('custom_map', [r, o, y, g, b])
shared_norm = mcolors.BoundaryNorm(levels_fill, ncolors=shared_cmap.N, clip=True)

# Many smooth filled levels for the color gradient (same everywhere)
levels_fill = np.linspace(vmin, vmax, 100)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# "Main contour lines" also fixed/consistent (same everywhere).
# Keeping 10 lines like your current behavior:
levels_line = np.linspace(vmin, vmax, 10)

scenarios = [
    {"leak_fraction_capturable": 0.5, "electricity_price_per_kWh": 0.05},
    {"leak_fraction_capturable": 0.8, "electricity_price_per_kWh": 0.05},
    {"leak_fraction_capturable": 0.5, "electricity_price_per_kWh": 0.10},
    {"leak_fraction_capturable": 0.8, "electricity_price_per_kWh": 0.10},
]

for ax, params in zip(axes.flat, scenarios):
    plot_methane_savings_contour(
        biogas_production_rate=6609 / 3785.41,
        leak_fraction_capturable=params["leak_fraction_capturable"],
        electricity_price_per_kWh=params["electricity_price_per_kWh"],
        ogi_cost=100000,
        fig=fig,
        ax=ax,
        levels_fill=levels_fill,
        levels_line=levels_line,
        cmap=shared_cmap,
        norm=shared_norm
    )
plt.tight_layout()
plt.show()
# %%


def plot_methane_savings_contour_by_capturable(
    biogas_production_rate,
    leak_rate,  # <-- fixed leak rate
    electricity_price_per_kWh,
    ogi_cost=100000,
    plant_sizes_m3_per_day_range=(0, 1_200_000),
    capturable_fraction_range=(0, 1.0),   # <-- now the y-axis variable
    resolution=200,
    fig=None,
    ax=None,
    save_path=None,
    levels_fill=None,
    levels_line=None,
    cmap=None,
    norm=None
):
    """
    Returns a methane savings contour plot (Jianan-style) where leak rate is fixed and
    fraction of gas capturable varies along the Y-axis.

    Parameters:
    - biogas_production_rate: MJ biogas / m³
    - leak_rate: Fixed leak rate (fraction)
    - electricity_price_per_kWh: $ per kWh
    - ogi_cost: Fixed cost of OGI survey
    - plant_sizes_m3_per_day_range: (min, max) plant size in m³/day
    - capturable_fraction_range: (min, max) capturable fraction range
    - resolution: Number of grid points along x-axis (plant size)
    - fig, ax: Optional Matplotlib figure and axis
    - save_path: Optional filepath to save plot
    - levels_fill: shared fill contour levels for consistent color scaling
    - levels_line: shared line contour levels for consistent labeling
    - cmap, norm: shared colormap & normalization for fixed color mapping
    """

    # Grid
    plant_sizes = np.linspace(*plant_sizes_m3_per_day_range, resolution)
    capturable_fractions = np.linspace(*capturable_fraction_range, resolution)
    X, Y = np.meshgrid(plant_sizes, capturable_fractions)

    # Compute annual savings
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = calc_annual_savings(
                plant_size=X[i, j],
                biogas_production_rate=biogas_production_rate,
                leak_rate=leak_rate,   # <-- fixed
                leak_fraction_capturable=Y[i, j],  # <-- variable
                electricity_price_per_kWh=electricity_price_per_kWh,
                ogi_cost=ogi_cost
            )

    # Flatten for tricontourf
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    # Create fig/ax if needed
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7.3, 6))

    # Mask negatives so they plot as background
    Z_masked = np.ma.masked_less(Z_flat, 0)

    # Draw a white background rectangle so masked areas are white
    ax.set_facecolor('white')

    # Filled contour
    fills = ax.tricontourf(
        X_flat, Y_flat, Z_flat,
        levels=levels_fill,
        cmap=cmap,
        norm=norm
    )

    # Contour lines
    def label_formatter(x):
        if abs(x) >= 1e6:
            return f"${x/1e6:.0f}M"
        elif abs(x) >= 1e3:
            return f"${x/1e3:.0f}k"
        else:
            return f"${x:.0f}"

    contour_lines = ax.tricontour(
        X_flat, Y_flat, Z_flat,
        levels=levels_line,
        colors='black', linewidths=2.0
    )
    ax.clabel(contour_lines, inline=True, fontsize=10, fmt=label_formatter, rightside_up=True)

    # Breakeven line
    ax.tricontour(
        X_flat, Y_flat, Z_flat,
        levels=[0], colors='black', linewidths=3.0, linestyles='solid', zorder=3
    )

    # Axes formatting
    ax.tick_params(direction='inout', length=15, width=1.5, pad=6)
    ax.set_xlabel("Plant Size (m³/day)", fontsize=16)
    ax.set_ylabel("Fraction of Gas Capturable", fontsize=16)
    ax.set_xlim(plant_sizes_m3_per_day_range)
    xticks = np.arange(0, plant_sizes_m3_per_day_range[1] + 1, 200_000)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{val / 1e6:.1f}M" for val in xticks])
    ax.set_ylim(capturable_fraction_range)
    ax.set_yticks(np.linspace(capturable_fraction_range[0], capturable_fraction_range[1], 6))
    ax.set_yticklabels([f"{val:.1f}" for val in np.linspace(capturable_fraction_range[0], capturable_fraction_range[1], 6)])
    ax.tick_params(labelsize=12)

    # Save plot if needed
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return ax


# ======================
# SHARED SCALE SETTINGS
# ======================
# Lock the Z-scale across all panels.
pos_min = 0
pos_max = 3_000_000  # or your desired max
levels_fill = np.linspace(pos_min, pos_max, 101)  # 100 bins for positive values

# Main contour lines (identical everywhere)
levels_line = np.linspace(vmin, vmax, 10)

# Shared cmap/norm for positives
b, g, y, o, r = "#60C1CF", "#79BF82", "#F3C354", "#F98F60", "#ED586F"
shared_cmap = mcolors.LinearSegmentedColormap.from_list('custom_map', [r, o, y, g, b])
shared_norm = mcolors.BoundaryNorm(levels_fill, ncolors=shared_cmap.N, clip=True)

# ======================
# READY-TO-RUN EXAMPLE
# ======================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Four scenarios: fixed leak_rate per panel, plus electricity price variation
scenarios = [
    {"leak_rate": 0.05, "electricity_price_per_kWh": 0.05},
    {"leak_rate": 0.05, "electricity_price_per_kWh": 0.10},
    {"leak_rate": 0.10, "electricity_price_per_kWh": 0.05},
    {"leak_rate": 0.10, "electricity_price_per_kWh": 0.10},
]

for ax, params in zip(axes.flat, scenarios):
    plot_methane_savings_contour_by_capturable(
        biogas_production_rate=6609 / 3785.41,
        leak_rate=params["leak_rate"],  # fixed per subplot
        electricity_price_per_kWh=params["electricity_price_per_kWh"],
        ogi_cost=100000,
        plant_sizes_m3_per_day_range=(0, 1_200_000),
        capturable_fraction_range=(0, 1.0),
        resolution=200,
        fig=fig,
        ax=ax,
        levels_fill=levels_fill,
        levels_line=levels_line,
        cmap=shared_cmap,
        norm=shared_norm
    )
    ax.set_title(
        f"Fixed leak rate: {params['leak_rate']:.0%}, "
        f"Electricity price: ${params['electricity_price_per_kWh']}/kWh"
    )

plt.tight_layout()
plt.show()

# Save figure
# save_path = pathlib.Path("03_figures", "economic_analysis_by_capturable.png")
# save_path.parent.mkdir(parents=True, exist_ok=True)
# fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
# print(f"Plot saved to: {save_path.resolve()}")



#%%
elec_price = 0.08  # $/kWh fixed for all panels
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# -------------------------
# Top row — vary leak rate
# -------------------------
# Panel (0,0): capturable = 0.5
params = {"leak_fraction_capturable": 0.5, "electricity_price_per_kWh": elec_price}
plot_methane_savings_contour(
    biogas_production_rate=6609 / 3785.41,
    leak_fraction_capturable=0.5,
    electricity_price_per_kWh=elec_price,
    ogi_cost=100000,
    fig=fig,
    ax=axes[0, 0],
    levels_line=levels_line,
    cmap=shared_cmap,
    norm=None  # keep as your working version
)
# (Optional) override/clarify title outside the function
axes[0, 0].set_title("Capturable fraction: 0.5 — vary leak rate")

# Panel (0,1): capturable = 0.8
params = {"leak_fraction_capturable": 0.8, "electricity_price_per_kWh": elec_price}
plot_methane_savings_contour(
    biogas_production_rate=6609 / 3785.41,
    leak_fraction_capturable=0.8,
    electricity_price_per_kWh=elec_price,
    ogi_cost=100000,
    fig=fig,
    ax=axes[0, 1],
    levels_line=levels_line,
    cmap=shared_cmap,
    norm=None
)
axes[0, 1].set_title("Capturable fraction: 0.8 — vary leak rate")

# ----------------------------------------
# Bottom row — vary capturable fraction
# ----------------------------------------
# Panel (1,0): leak rate = 5%
plot_methane_savings_contour_by_capturable(
    biogas_production_rate=6609 / 3785.41,
    leak_rate=0.05,
    electricity_price_per_kWh=elec_price,
    ogi_cost=100000,
    fig=fig,
    ax=axes[1, 0],
    levels_fill=pos_levels_fill,
    levels_line=levels_line,
    cmap=shared_cmap,
    norm=shared_norm
)
axes[1, 0].set_title("Leak rate: 5% — vary capturable fraction")

# Panel (1,1): leak rate = 10%
plot_methane_savings_contour_by_capturable(
    biogas_production_rate=6609 / 3785.41,
    leak_rate=0.10,
    electricity_price_per_kWh=elec_price,
    ogi_cost=100000,
    fig=fig,
    ax=axes[1, 1],
    levels_fill=pos_levels_fill,
    levels_line=levels_line,
    cmap=shared_cmap,
    norm=shared_norm
)
axes[1, 1].set_title("Leak rate: 10% — vary capturable fraction")

plt.tight_layout()
plt.show()
#%%
#%%
#%%


#### OLD CODE ####
# # #%% 

# #Setup
# import pandas as pd 
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pathlib
# import numpy as np
# import pandas as pd
# import matplotlib.ticker as mticker
# import matplotlib.colors as mcolors



# from a_my_utilities import calc_annual_savings


# def plot_methane_savings_contour(
#     biogas_production_rate,
#     leak_fraction_capturable,
#     electricity_price_per_kWh,
#     ogi_cost=100000,
#     plant_sizes_m3_per_day_range=(0, 1_200_000),
#     leak_rates=np.linspace(0, 0.5, 200),
#     resolution=200,
#     fig=None,
#     ax=None,
#     save_path=None
# ):
#     """
#     Returns a methane savings contour plot (Jianan-style) using user's methane functions.

#     Parameters:
#     - biogas_production_rate: MJ biogas / m³
#     - leak_fraction_capturable: Fraction of leakage that is recoverable
#     - electricity_price_per_kWh: $ per kWh
#     - ogi_cost: Fixed cost of OGI survey
#     - plant_sizes_m3_per_day_range: (min, max) plant size in m³/day
#     - leak_rates: Array of leak rates (0 to 1)
#     - resolution: Number of grid points along x-axis (plant size)
#     - fig, ax: Optional Matplotlib figure and axis
#     - save_path: Optional filepath to save plot

#     Returns:
#     - ax: The axis containing the contour plot
#     """

#     # Grid
#     plant_sizes = np.linspace(*plant_sizes_m3_per_day_range, resolution)
#     X, Y = np.meshgrid(plant_sizes, leak_rates)

#     # Compute annual savings
#     Z = np.zeros_like(X)
#     for i in range(X.shape[0]):
#         for j in range(X.shape[1]):
#             Z[i, j] = calc_annual_savings(
#                 plant_size=X[i, j],
#                 biogas_production_rate=biogas_production_rate,
#                 leak_rate=Y[i, j],
#                 leak_fraction_capturable=leak_fraction_capturable,
#                 electricity_price_per_kWh=electricity_price_per_kWh,
#                 ogi_cost=ogi_cost
#             )

#     # Flatten for tricontourf
#     X_flat = X.flatten()
#     Y_flat = Y.flatten()
#     Z_flat = Z.flatten()

#     # Jianan-style color map
#     b, g, y, o, r = "#60C1CF", "#79BF82", "#F3C354", "#F98F60", "#ED586F"
#     color_map = mcolors.LinearSegmentedColormap.from_list('custom_map', [r, o, y, g, b])

#     # Create fig/ax if needed
#     if fig is None or ax is None:
#         fig, ax = plt.subplots(figsize=(7.3, 6))

#     # Filled contour
#     levels_fill = np.linspace(0, np.max(Z_flat), 100)
#     fills = ax.tricontourf(X_flat, Y_flat, Z_flat, levels=levels_fill, cmap=color_map)

#     # Contour lines

#     def label_formatter(x):
#         if abs(x) >= 1e6:
#             return f"${x/1e6:.0f}M"
#         elif abs(x) >= 1e3:
#             return f"${x/1e3:.0f}k"
#         else:
#             return f"${x:.0f}"
        
#     contour_lines = ax.tricontour(X_flat, Y_flat, Z_flat, levels=10, colors='black', linewidths=2.0)
#     ax.clabel(contour_lines, inline=True, fontsize=10, fmt=label_formatter, rightside_up=True)

#     # Breakeven line
#     breakeven = ax.tricontour(
#         X_flat, Y_flat, Z_flat,
#         levels=[0], colors='black', linewidths=3.0, linestyles='solid', zorder=3
#     )
#     # ax.clabel(breakeven, fmt={0: '$0'}, fontsize=12, colors='red')

#     # Axes formatting
#     ax.tick_params(direction='inout', length=15, width=1.5, pad=6)
#     ax.set_xlabel("Plant Size (m³/day)", fontsize=16)
#     ax.set_ylabel("Leak Rate (%)", fontsize=16)
#     ax.set_xlim(plant_sizes_m3_per_day_range)
#     xticks = np.arange(0, plant_sizes_m3_per_day_range[1] + 1, 200_000)
#     ax.set_xticks(xticks)
#     ax.set_xticklabels([f"{val / 1e6:.1f}M" for val in xticks])
#     ax.set_ylim(leak_rates.min(), leak_rates.max())
#     ax.set_yticks(np.linspace(leak_rates.min(), leak_rates.max(), 6))
#     ax.set_yticklabels([f"{val:.1f}" for val in np.linspace(leak_rates.min(), leak_rates.max(), 6)])
#     ax.tick_params(labelsize=12)

#     # Save plot if needed
#     if save_path:
#         fig.savefig(save_path, dpi=300, bbox_inches='tight')

#     return ax

# fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# scenarios = [
#     {"leak_fraction_capturable": 0.5, "electricity_price_per_kWh": 0.05},
#     {"leak_fraction_capturable": 0.8, "electricity_price_per_kWh": 0.05},
#     {"leak_fraction_capturable": 0.5, "electricity_price_per_kWh": 0.10},
#     {"leak_fraction_capturable": 0.8, "electricity_price_per_kWh": 0.10},
# ]

# for ax, params in zip(axes.flat, scenarios):
#     plot_methane_savings_contour(
#         biogas_production_rate=6609 / 3785.41,
#         leak_fraction_capturable=params["leak_fraction_capturable"],
#         electricity_price_per_kWh=params["electricity_price_per_kWh"],
#         ogi_cost=100000,
#         fig=fig,
#         ax=ax
#     )
#     ax.set_title(f"Fraction of capturable gas: {params['leak_fraction_capturable']}, Electricity price: ${params['electricity_price_per_kWh']}/kWh")

# plt.tight_layout()
# plt.show()

# # Ensure output directory exists
# save_path=pathlib.Path("03_figures", "economic_analysis.png")

# fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)

# print(f"Plot saved to: {save_path.resolve()}")
# # %%
