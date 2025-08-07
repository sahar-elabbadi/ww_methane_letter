# #%% 

#Setup
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors



from a_my_utilities import mj_per_kg_CH4, mj_per_kWh, mgd_to_m3_per_day, m3_per_mg

def calc_leak_value(plant_size, biogas_production_rate, leak_rate, leak_fraction_capturable, electricity_price_per_kWh):
    """
    plant_size: size of the plant in m3/day
    biogas_production_rate: biogas production rate as MJ biogas per m3 treated flow 
    leak_rate: leak rate as a fraction of the biogas production rate
    leak_fraction_capturable: fraction of the leak that can be captured

    """

    biogas_production_MJ_per_day = plant_size * biogas_production_rate
    # print(f'Biogas production rate: {biogas_production_MJ_per_day} MJ/day')

    methane_leakage_MJ_per_hr = biogas_production_MJ_per_day * leak_rate * (1/24) # Convert to per hour
    # print(f'Methane leakage: {methane_leakage_MJ_per_hr} MJ/hr')
   
    methane_leakage_kg_per_hr = methane_leakage_MJ_per_hr * (1/mj_per_kg_CH4()) # Convert to kg CH4 per hour
    # print(f'Methane leakage: {methane_leakage_kg_per_hr} kg CH4/hr')
   
    electricity_generation_potential_kWh_per_hour = methane_leakage_MJ_per_hr *\
          leak_fraction_capturable * (1/mj_per_kWh()) # Convert to kWh per hour
    
    leak_value_usd_per_hour = electricity_generation_potential_kWh_per_hour * electricity_price_per_kWh # Convert to USD per hour
    
    return leak_value_usd_per_hour


def calc_payback_period(plant_size, biogas_production_rate, leak_rate, leak_fraction_capturable, electricity_price_per_kWh, ogi_cost=100000): 
    """
    Calculate the payback period (days) for a methane leak OGI survey based on the leak value.
    
    plant_size: size of the plant in m3/day
    biogas_production_rate: biogas production rate as MJ biogas per m3 treated flow 
    leak_rate: leak rate as a fraction of the biogas production rate
    leak_fraction_capturable: fraction of the leak that can be captured
    electricity_price_per_kWh: price of electricity in USD per kWh
    """

    leak_value = calc_leak_value(plant_size, biogas_production_rate, leak_rate, leak_fraction_capturable, electricity_price_per_kWh)
    
    payback_period = ogi_cost / leak_value * (1/24) # Payback period in days
    
    return payback_period


def calc_annual_savings(plant_size, biogas_production_rate, leak_rate, leak_fraction_capturable, electricity_price_per_kWh, ogi_cost=100000):
    """
    Calculate the annual savings from capturing methane leaks.
    
    plant_size: size of the plant in m3/day
    biogas_production_rate: biogas production rate as MJ biogas per m3 treated flow 
    leak_rate: leak rate as a fraction of the biogas production rate
    leak_fraction_capturable: fraction of the leak that can be captured
    electricity_price_per_kWh: price of electricity in USD per kWh
    ogi_cost: cost of OGI survey in USD
    """
    
    leak_value = calc_leak_value(plant_size, biogas_production_rate, leak_rate, leak_fraction_capturable, electricity_price_per_kWh)
    

    annual_savings = leak_value * 24 * 365 - ogi_cost  # Annual savings in USD
    
    return annual_savings

# def plot_methane_savings_contour(
#     biogas_production_rate,
#     leak_fraction_capturable,
#     electricity_price_per_kWh,
#     ogi_cost=100000,
#     plant_sizes_m3_per_day_range=(0, 1_200_000),
#     leak_rates=np.linspace(0, 1, 200),
#     resolution=200,
#     fig=None,
#     ax=None
# ):
#     """
#     Returns an axis with a methane savings contour plot (Jianan-style).
#     If fig/ax are not provided, creates and returns them.
#     """

#     import matplotlib.pyplot as plt
#     import matplotlib.colors as mcolors
#     import numpy as np

#     # Helper functions
#     def mj_per_kWh(): return 3.6
#     def calc_leak_value(size, rate): 
#         return (size * biogas_production_rate * rate / 24) * leak_fraction_capturable / mj_per_kWh() * electricity_price_per_kWh
    
#     def calc_annual_savings(size, rate): 
#         val = calc_leak_value(size, rate)
#         return val * 24 * 365 - ogi_cost

#     # Grid
#     plant_sizes = np.linspace(*plant_sizes_m3_per_day_range, resolution)
#     X, Y = np.meshgrid(plant_sizes, leak_rates)
#     Z = np.vectorize(calc_annual_savings)(X, Y)
#     X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()

#     # Custom color map
#     b, g, y, o, r = "#60C1CF", "#79BF82", "#F3C354", "#F98F60", "#ED586F"
#     color_map = mcolors.LinearSegmentedColormap.from_list('custom_map', [b, g, y, o, r])

#     # Create fig/ax if not provided
#     if ax is None or fig is None:
#         fig, ax = plt.subplots(figsize=(7.3, 6))

#     # Plot filled contours
#     levels_fill = np.linspace(0, np.max(Z_flat), 100)
#     fills = ax.tricontourf(X_flat, Y_flat, Z_flat, levels=levels_fill, cmap=color_map)

#     # --- Bold breakeven line (where savings = $0)
#     breakeven = ax.tricontour(
#         X_flat,
#         Y_flat,
#         Z_flat,
#         levels=[0],
#         colors='red',
#         linewidths=3.0,
#         linestyles='solid',
#         zorder=3)

#     # Optional: Label it directly
#     # ax.clabel(
#     #     breakeven,
#     #     fmt={0: '$0'},
#     #     fontsize=12,
#     #     colors='red'
#     # )

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

#     # Axis styling
#     ax.tick_params(direction='inout', length=15, width=1.5, pad=6)
#     ax.set_xlabel("Plant Size (m³/day)", fontsize=16)
#     ax.set_ylabel("Leak Rate (%)", fontsize=16)
#     ax.set_xlim(plant_sizes_m3_per_day_range)
#     xticks = np.arange(0, plant_sizes_m3_per_day_range[1] + 1, 200_000)
#     ax.set_xticks(xticks)
#     ax.set_xticklabels([f"{val / 1e6:.1f}M" for val in xticks])
#     # ax.set_ylim(0, 1.0)
#     ax.set_ylim(leak_rates.min(), leak_rates.max())
#     ax.set_yticks(np.linspace(leak_rates.min(), leak_rates.max(), 6))
#     ax.tick_params(labelsize=12)

#     return ax

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
    color_map = mcolors.LinearSegmentedColormap.from_list('custom_map', [b, g, y, o, r])

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
        levels=[0], colors='red', linewidths=3.0, linestyles='solid', zorder=3
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
save_path=pathlib.Path("03_figures", "Fig3_economic_analysis.png")

fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)

print(f"Plot saved to: {save_path.resolve()}")
# %%
