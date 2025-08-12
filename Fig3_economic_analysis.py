#%% 

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
    save_path=None
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

    # Compute annual savings using user's function
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
    fills = ax.tricontourf(X_flat, Y_flat, Z_flat, levels=levels_fill, cmap=color_map)

    # Contour lines

    def label_formatter(x):
        if abs(x) >= 1e6:
            return f"${x/1e6:.0f}M"
        elif abs(x) >= 1e3:
            return f"${x/1e3:.0f}k"
        else:
            return f"${x:.0f}"
        
    contour_lines = ax.tricontour(X_flat, Y_flat, Z_flat, levels=10, colors='black', linewidths=2.0)
    ax.clabel(contour_lines, inline=True, fontsize=10, fmt=label_formatter, rightside_up=True)

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

    # Save plot if needed
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return ax

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

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
        ax=ax
    )
    ax.set_title(f"Fraction of capturable gas: {params['leak_fraction_capturable']}, Electricity price: ${params['electricity_price_per_kWh']}/kWh")

plt.tight_layout()
plt.show()

# Ensure output directory exists
save_path=pathlib.Path("03_figures", "economic_analysis.png")

fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)

print(f"Plot saved to: {save_path.resolve()}")
# %%
