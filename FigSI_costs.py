#%% 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pathlib
from a_my_utilities import Engine, ENGINES, calc_annual_revenue

####### Data and constants #######
# Load chp dataset 
chp_data = pd.read_csv(pathlib.Path("02_clean_data", "chp_data.csv"))

max_plant_size_m3_per_day=1_600_000

elec_price = 0.09  # $/kWh fixed for all panels - value based on calculations in Paper_Calculations.py, median electricity price for facilities with CHP
gas_price = 0.008 # $/MJ fixed for all panels - value based on calculations in Paper_Calculations.py, median natural gas price for facilities with CHP
engine = ENGINES['reciprocating_lean_burn'] # SET ENGINE TYPE HERE 


def plot_methane_savings_vary_leak_rate_add_cost_labels(
    leak_fraction_capturable,
    electricity_price_per_kWh,
    nat_gas_price_per_MJ,
    engine: Engine, # <- pass an Engine object
    ogi_cost=100000,
    plant_sizes_m3_per_day_range=(0, 1_200_000),
    leak_rates=np.linspace(0, 0.25, 200),
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
    Returns a methane savings contour plot (Jianan-style) where leak rate varies along the Y-axis.
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
                engine=engine, 
                electricity_price_per_kWh=electricity_price_per_kWh,
                nat_gas_price_per_MJ=nat_gas_price_per_MJ,
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
    ax.clabel(cs, inline=True, fontsize=16, fmt=label_formatter, rightside_up=True)

    #### Cost of repairs
     
    ax.tricontour(Xf, Yf, Zf, levels=[0], colors='black', linewidths=3.0, linestyles='solid', zorder=3)

    # Cost of OGI survey
    # OGI - Low 
    ogi_cost_low = 30_000
    cost1 = ax.tricontour(Xf, Yf, Zf, levels=[ogi_cost_low], colors='black', linewidths=3.0, linestyles='dashed', zorder=3)
    ax.clabel(cost1, inline=True, fmt={ogi_cost_low: "OGI (low)"}, fontsize=16)

    # OGI - High
    ogi_cost_high = 60_000
    cost2 = ax.tricontour(Xf, Yf, Zf, levels=[ogi_cost_high], colors='black', linewidths=3.0, linestyles='dashed', zorder=3)
    ax.clabel(cost2, inline=True, fmt={ogi_cost_high: "OGI (high)"}, fontsize=16)

    # Replace digestor cover - low 
    digester_cover_cost_low = 2_000_000
    cost3 = ax.tricontour(Xf, Yf, Zf, levels=[digester_cover_cost_low], colors='black', linewidths=3.0, linestyles='dashed', zorder=3)
    ax.clabel(cost3, inline=True, fmt={digester_cover_cost_low: "Digester Cover (low)"}, fontsize=16)


    # Axes formatting
    ax.tick_params(direction='in', length=8, width=1.5, pad=6, labelsize=14)
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

###### Generate Figure #######

# Range for heat map contour lines 
vmin = 0 # min dollar value
vmax = 5_500_000 # max dollar value 
levels_fill = np.linspace(vmin, vmax, 100)  
# levels_line = np.linspace(vmin, vmax, 10)    # default evenly spaced lines
levels_line = [100_000, 250_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000, 4_000_000, 5_000_000]

####### Set up figure #######
fig, ax = plt.subplots(figsize=(20, 20))


####### Color maps #######
# Shared cmap/norm for positives
b, g, y, o, r = "#60C1CF", "#79BF82", "#F3C354", "#F98F60", "#ED586F"
shared_cmap = mcolors.LinearSegmentedColormap.from_list('custom_map', [o, y, g, b])
shared_norm = mcolors.BoundaryNorm(levels_fill, ncolors=shared_cmap.N, clip=True)


# Range for heat map contour lines 
vmin = 0 # min dollar value
vmax = 5_500_000 # max dollar value 
levels_fill = np.linspace(vmin, vmax, 100)  
# levels_line = np.linspace(vmin, vmax, 10)    # default evenly spaced lines
levels_line = [250_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_500_000, 3_000_000, 4_000_000, 5_000_000]


params = {"leak_fraction_capturable": 0.8, "electricity_price_per_kWh": elec_price}
plot_methane_savings_vary_leak_rate_add_cost_labels(
    leak_fraction_capturable=0.8,
    electricity_price_per_kWh=elec_price,
    nat_gas_price_per_MJ=gas_price, 
    leak_rates=np.linspace(0, 0.50, 200),
    engine=engine,
    plant_sizes_m3_per_day_range=(0, max_plant_size_m3_per_day),
    ogi_cost=100000,
    fig=fig,
    ax=ax,
    levels_line=levels_line,
    cmap=shared_cmap,
    title=False,
    levels_fill=levels_fill,
    norm=shared_norm,
)

ax.set_xlabel("Plant size (m³/day)", fontsize=18)
ax.set_ylabel("Leak rate (%)", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)

