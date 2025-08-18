import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from a_my_utilities import calc_annual_savings, calc_annual_revenue, set_chini_dataset

chini_data = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))  

set_chini_dataset(
    chini_data,
    x_col="flow_m3_per_day",
    y_col="methane_gen_kgh",
    drop_negative=True
)


def plot_methane_savings_vary_leak_rate(
    leak_fraction_capturable,
    electricity_price_per_kWh,
    ogi_cost=100000,
    plant_sizes_m3_per_day_range=(0, 1_200_000),
    leak_rates=np.linspace(0, 0.25, 200),
    engine_efficiency=0.45,
    resolution=200,
    fig=None,
    ax=None,
    save_path=None,
    levels_fill=None,   # <- use caller's if provided
    levels_line=None,   # <- use caller's if provided
    cmap=None,
    norm=None,
    title=False          # <- optional custom title
):
    """
    Returns a methane savings contour plot (Jianan-style) using user's methane functions.
    """
    # Grid
    plant_sizes = np.linspace(*plant_sizes_m3_per_day_range, resolution)
    X, Y = np.meshgrid(plant_sizes, leak_rates)

    # Compute annual savings
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = calc_annual_revenue(
                plant_size=X[i, j],
                leak_rate=Y[i, j],
                leak_fraction_capturable=leak_fraction_capturable,
                engine_efficiency=engine_efficiency, 
                electricity_price_per_kWh=electricity_price_per_kWh,
                ogi_cost=ogi_cost
            )

    # Flatten for tricontourf
    Xf, Yf, Zf = X.ravel(), Y.ravel(), Z.ravel()

    # Create fig/ax if needed
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7.3, 6))

    # Respect caller's shared levels; otherwise compute a local default
    if levels_fill is None:
        # if using a shared BoundaryNorm, you *should* pass levels_fill from caller
        zmax = max(0.0, float(np.nanmax(Zf)))
        levels_fill = np.linspace(0, zmax if zmax > 0 else 1, 100)

    # Filled contour
    ax.tricontourf(Xf, Yf, Zf, levels=levels_fill, cmap=cmap, norm=norm)

    # Contour lines (respect caller; else simple defaults)
    if levels_line is None:
        # choose a few nice round numbers up to the max
        zmax = float(np.nanmax(Zf))
        step = max(1, int(zmax // 5))  # crude default
        levels_line = np.linspace(step, zmax, 5)

    def label_formatter(x):
        if abs(x) >= 1e6:
            return f"${x/1e6:.1f}M"
        elif abs(x) >= 1e3:
            return f"${x/1e3:.1f}k"
        else:
            return f"${x:.0f}"

    cs = ax.tricontour(Xf, Yf, Zf, levels=levels_line, colors='black', linewidths=2.0)
    ax.clabel(cs, inline=True, fontsize=10, fmt=label_formatter, rightside_up=True)

    # # Breakeven line
    # ax.tricontour(Xf, Yf, Zf, levels=[0], colors='black', linewidths=3.0, linestyles='solid', zorder=3)

    # Cost of OGI survey
    cost1 = ax.tricontour(Xf, Yf, Zf, levels=[100_000], colors='black', linewidths=3.0, linestyles='dashed', zorder=3)
        # Cost of OGI survey
        #     
    ax.clabel(cost1, inline=True, fmt={100_000: "OGI survey"}, fontsize=10)


    # Axes formatting
    ax.tick_params(direction='in', length=8, width=1.5, pad=6, labelsize=12)
    ax.set_xlabel("Plant Size (m³/day)", fontsize=16)
    ax.set_ylabel("Leak Rate (%)", fontsize=16)
    ax.set_xlim(plant_sizes_m3_per_day_range)

    xticks = np.arange(0, plant_sizes_m3_per_day_range[1] + 1, 200_000)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{val / 1e6:.1f}M" for val in xticks])

    ax.set_ylim(leak_rates.min(), leak_rates.max())
    yticks = np.linspace(leak_rates.min(), leak_rates.max(), 6)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{val*100:.0f}%" for val in yticks])  # show percents

    # Title: use provided or compose from args 
    if title is True:
        title = (
            f"Fraction capturable: {leak_fraction_capturable}, "
            f"Electricity price: ${electricity_price_per_kWh}/kWh"
        )
        ax.set_title(title)


    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return ax


def plot_methane_savings_vary_capturable(
    leak_rate,  # <-- fixed leak rate
    electricity_price_per_kWh,
    ogi_cost=100000,
    plant_sizes_m3_per_day_range=(0, 1_200_000),
    capturable_fraction_range=(0, 1.0),   # <-- now the y-axis variable
    engine_efficiency=0.45,
    resolution=200,
    fig=None,
    ax=None,
    save_path=None,
    levels_fill=None,
    levels_line=None,
    cmap=None,
    norm=None,
    title=False          # <- optional custom title
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
            Z[i, j] = calc_annual_revenue(
                plant_size=X[i, j],
                leak_rate=leak_rate,   # <-- fixed
                leak_fraction_capturable=Y[i, j],  # <-- variable
                engine_efficiency=engine_efficiency,
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
            return f"${x/1e6:.1f}M"
        elif abs(x) >= 1e3:
            return f"${x/1e3:.1f}k"
        else:
            return f"${x:.1f}"

    contour_lines = ax.tricontour(
        X_flat, Y_flat, Z_flat,
        levels=levels_line,
        colors='black', linewidths=2.0
    )
    ax.clabel(contour_lines, inline=True, fontsize=10, fmt=label_formatter, rightside_up=True)

    # Breakeven line
    # ax.tricontour(
    #     X_flat, Y_flat, Z_flat,
    #     levels=[0], colors='black', linewidths=3.0, linestyles='solid', zorder=3
    # )

    # Cost of OGI survey
    cost1 = ax.tricontour(X_flat, Y_flat, Z_flat, levels=[100_000], colors='black', 
                  linewidths=3.0, linestyles='dashed', zorder=3)
    
    ax.clabel(cost1, inline=True, fmt={100_000: "OGI survey"}, fontsize=10)


    # Axes formatting
    ax.tick_params(direction='in', length=8, width=1.5, pad=6, labelsize=12)
    ax.set_xlabel("Plant Size (m³/day)", fontsize=16)
    ax.set_ylabel("Fraction of Gas Capturable", fontsize=16)
    ax.set_xlim(plant_sizes_m3_per_day_range)
    xticks = np.arange(0, plant_sizes_m3_per_day_range[1] + 1, 200_000)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{val / 1e6:.1f}M" for val in xticks])
    ax.set_ylim(capturable_fraction_range)
    ax.set_yticks(np.linspace(capturable_fraction_range[0], capturable_fraction_range[1], 6))
    ax.set_yticklabels([f"{val:.2f}" for val in np.linspace(capturable_fraction_range[0], capturable_fraction_range[1], 6)])

    # Title: use provided or compose from args (NO 'params' here)
    if title is True:
        title = (
            f"Leak rate: {leak_rate*100:.1f}%, "
            f"Electricity price: ${electricity_price_per_kWh}/kWh"
        )
        ax.set_title(title)

    # Save plot if needed
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return ax




