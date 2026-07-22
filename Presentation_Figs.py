#%% 

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pathlib
from c_plotting_functions import plot_methane_savings_vary_leak_rate, plot_methane_savings_vary_capturable
from a_my_utilities import Engine, ENGINES


# Copy from Fig4_economic_analysis.py

####### Data and constants #######
# Load chp dataset 
chp_data = pd.read_csv(pathlib.Path("02_clean_data", "chp_data.csv"))

max_plant_size_m3_per_day=1_600_000

elec_price = 0.09  # $/kWh fixed for all panels - value based on calculations in Paper_Calculations.py, median electricity price for facilities with CHP
gas_price = 0.008 # $/MJ fixed for all panels - value based on calculations in Paper_Calculations.py, median natural gas price for facilities with CHP
engine = ENGINES['reciprocating_lean_burn'] # SET ENGINE TYPE HERE 

vmin = 0
vmax = 5_500_000
levels_fill = np.linspace(vmin, vmax, 100)  
# levels_line = np.linspace(vmin, vmax, 10)    # default evenly spaced lines
levels_line = [100_000, 250_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000, 4_000_000, 5_000_000]
levels_line_bottom = [50_000, 100_000, 250_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000, 4_000_000, 5_000_000]
levels_line_bottom_left = [25_000, 50_000, 100_000, 250_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000, 4_000_000, 5_000_000]

####### Color maps #######
# Shared cmap/norm for positives
b, g, y, o, r = "#60C1CF", "#79BF82", "#F3C354", "#F98F60", "#ED586F"
shared_cmap = mcolors.LinearSegmentedColormap.from_list('custom_map', [o, y, g, b])
shared_norm = mcolors.BoundaryNorm(levels_fill, ncolors=shared_cmap.N, clip=True)

# initialize ax, 1 x 1 grid 
fig, ax = plt.subplots(figsize=(8, 8))
# Panel (0,0): capturable = 0.5
params = {"leak_fraction_capturable": 0.5, "electricity_price_per_kWh": elec_price}
plot_methane_savings_vary_leak_rate(
    leak_fraction_capturable=0.8,
    electricity_price_per_kWh=elec_price,
    nat_gas_price_per_MJ=gas_price, 
    engine=engine,
    ogi_cost=100000,
    plant_sizes_m3_per_day_range=(0, max_plant_size_m3_per_day),
    leak_rates=np.linspace(0, 0.50, 200),
    fig=fig,
    ax=ax,
    levels_line=levels_line,
    levels_fill=levels_fill,
    norm=shared_norm,
    cmap=shared_cmap,
    title=False  
)


# Increase font size for all axes 
fontsize_axis_labels = 20
fontsize_tick_labels = 18

ax.set_xlabel("Plant Size (m³/day)", fontsize=fontsize_axis_labels)
ax.set_ylabel("Leak Rate (%)", fontsize=fontsize_axis_labels)
ax.tick_params(labelsize=fontsize_tick_labels)

# %%
