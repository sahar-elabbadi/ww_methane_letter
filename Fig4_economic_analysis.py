
#%%
# ==========================================================================
# ############## COMBINED FIGURE WITH BOTH VARIATIONS ###############
# ==========================================================================

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import pathlib
from c_plotting_functions import plot_methane_savings_vary_leak_rate, plot_methane_savings_vary_capturable

elec_price = 0.09  # $/kWh fixed for all panels - value based on calculations in Paper_Calculations.py, median electricity price for facilities with CHP
max_plant_size_m3_per_day=1_600_000

vmin = 0
vmax = 4_700_000
levels_fill = np.linspace(vmin, vmax, 100)  
# levels_line = np.linspace(vmin, vmax, 10)    # default evenly spaced lines
levels_line = [250_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000, 4_000_000, 5_000_000]


# Shared cmap/norm for positives
b, g, y, o, r = "#60C1CF", "#79BF82", "#F3C354", "#F98F60", "#ED586F"
shared_cmap = mcolors.LinearSegmentedColormap.from_list('custom_map', [o, y, g, b])
shared_norm = mcolors.BoundaryNorm(levels_fill, ncolors=shared_cmap.N, clip=True)


fig, axes = plt.subplots(2, 2, figsize=(14, 12))


# -------------------------
# Top row — vary leak rate
# -------------------------
# Panel (0,0): capturable = 0.5
params = {"leak_fraction_capturable": 0.5, "electricity_price_per_kWh": elec_price}
plot_methane_savings_vary_leak_rate(
    leak_fraction_capturable=0.5,
    electricity_price_per_kWh=elec_price,
    ogi_cost=100000,
    plant_sizes_m3_per_day_range=(0, max_plant_size_m3_per_day),
    leak_rates=np.linspace(0, 0.50, 200),
    engine_efficiency=0.45,
    fig=fig,
    ax=axes[0, 0],
    levels_line=levels_line,
    levels_fill=levels_fill,
    norm=shared_norm,
    cmap=shared_cmap,
    title=False  
)
# (Optional) override/clarify title outside the function
# axes[0, 0].set_title("Capturable fraction: 0.5 — vary leak rate")
# axes[0,0].set_xticks([])  # Remove x-ticks for cleaner look
axes[0, 0].text(0.02, 0.02, "Capturable: 0.5", transform=axes[0, 0].transAxes, fontsize=18)


# Panel (0,1): capturable = 0.8
params = {"leak_fraction_capturable": 0.8, "electricity_price_per_kWh": elec_price}
plot_methane_savings_vary_leak_rate(
    leak_fraction_capturable=0.8,
    electricity_price_per_kWh=elec_price,
    leak_rates=np.linspace(0, 0.50, 200),
    engine_efficiency=0.45,
    plant_sizes_m3_per_day_range=(0, max_plant_size_m3_per_day),
    ogi_cost=100000,
    fig=fig,
    ax=axes[0, 1],
    levels_line=levels_line,
    cmap=shared_cmap,
    title=False,
    levels_fill=levels_fill,
    norm=shared_norm,
)

# axes[0, 1].set_title("Capturable fraction: 0.8 — vary leak rate")
axes[0, 1].text(0.02, 0.02, "Capturable: 0.8", transform=axes[0, 1].transAxes, fontsize=18)


# ----------------------------------------
# Bottom row — vary capturable fraction
# ----------------------------------------
# Panel (1,0): leak rate = 5%
plot_methane_savings_vary_capturable(
    leak_rate=0.05,
    electricity_price_per_kWh=elec_price,
    ogi_cost=100000,
    engine_efficiency=0.45,
    plant_sizes_m3_per_day_range=(0, max_plant_size_m3_per_day),
    fig=fig,
    ax=axes[1, 0],
    levels_fill=levels_fill,
    levels_line=levels_line,
    cmap=shared_cmap,
    norm=shared_norm
)
# axes[1, 0].set_title("Leak rate: 5% — vary capturable fraction")
axes[1,0].text(0.02, 0.04, "Leak rate: 5%", transform=axes[1, 0].transAxes, fontsize=18)



# Panel (1,1): leak rate = 15%
plot_methane_savings_vary_capturable(
    leak_rate=0.15,
    electricity_price_per_kWh=elec_price,
    ogi_cost=100000,
    engine_efficiency=0.45,
    plant_sizes_m3_per_day_range=(0, max_plant_size_m3_per_day),
    fig=fig,
    ax=axes[1, 1],
    levels_fill=levels_fill,
    levels_line=levels_line,
    cmap=shared_cmap,
    norm=shared_norm
)
# axes[1, 1].set_title("Leak rate: 10% — vary capturable fraction")
axes[1, 1].text(0.02, 0.04, "Leak rate: 15%", transform=axes[1, 1].transAxes, fontsize=18)


plt.tight_layout()

# Increase tick label size in all subplots
for ax in axes.flat:
    
    ax.tick_params(labelsize=14, direction='in', length=5, width=1.5, pad=6)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # thickness in points


# Axes and tick labels for combined figure: 

# TOP LEFT
axes[0, 0].set_xlabel(None) # Remove xlabel to avoid clutter
axes[0,0].set_xticklabels([])  # Remove x-ticks for cleaner look

# TOP RIGHT
axes[0, 1].set_xlabel(None) # Remove xlabel to avoid clutter
axes[0,1].set_xticklabels([])  # Remove x-ticks for cleaner look
axes[0,1].set_yticklabels([])  # Remove y-ticks for cleaner look
axes[0,1].set_ylabel(None)  # Remove ylabel to avoid clutter

# BOTTOM RIGHT
axes[1,1].set_yticklabels([])  # Remove x-ticks for cleaner look
axes[1,1].set_ylabel(None)  # Remove x-ticks for cleaner look

plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
plt.show()

for ax in axes.flat:
    ax.set_xlabel(ax.get_xlabel(), fontsize=16)  # bump x-axis label font size
    ax.set_ylabel(ax.get_ylabel(), fontsize=16)  # bump y-axis label font size

for ax in axes.flat:
    ax.tick_params(axis="both", which="major", labelsize=14, length=6, width=1.5, direction="in", pad=6)


# ===========================
# SAVE FIGURE
# ===========================
save_path = pathlib.Path("03_figures", "Figure_4.png")
save_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
print(f"Plot saved to: {save_path.resolve()}")

#%%
### CHANGE ELECTRICITY PRICE 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import pathlib
from c_plotting_functions import plot_methane_savings_vary_leak_rate, plot_methane_savings_vary_capturable

max_plant_size_m3_per_day=1_600_000

vmin = 0
vmax = 9_200_000
levels_fill = np.linspace(vmin, vmax, 100)  
# levels_line = np.linspace(vmin, vmax, 10)    # default evenly spaced lines
levels_line = [250_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 
               3_000_000, 4_000_000, 5_000_000, 6_000_000, 7_000_000, 8_000_000, 9_000_000]


# Shared cmap/norm for positives
b, g, y, o, r = "#60C1CF", "#79BF82", "#F3C354", "#F98F60", "#ED586F"
shared_cmap = mcolors.LinearSegmentedColormap.from_list('custom_map', [o, y, g, b])
shared_norm = mcolors.BoundaryNorm(levels_fill, ncolors=shared_cmap.N, clip=True)


fig, axes = plt.subplots(2, 2, figsize=(14, 12))


# -------------------------
# Top row — $0.09 / kWh
# -------------------------

elec_price = 0.08  # $/kWh fixed for all panels

# Panel (0,0): capturable = 0.5
params = {"leak_fraction_capturable": 0.5, "electricity_price_per_kWh": elec_price}
plot_methane_savings_vary_leak_rate(
    leak_fraction_capturable=0.5,
    electricity_price_per_kWh=elec_price,
    ogi_cost=100000,
    plant_sizes_m3_per_day_range=(0, max_plant_size_m3_per_day),
    leak_rates=np.linspace(0, 0.50, 200),
    engine_efficiency=0.45,
    fig=fig,
    ax=axes[0, 0],
    levels_line=levels_line,
    levels_fill=levels_fill,
    norm=shared_norm,
    cmap=shared_cmap,
    title=False  
)
# (Optional) override/clarify title outside the function
# axes[0, 0].set_title("Capturable fraction: 0.5 — vary leak rate")
# axes[0,0].set_xticks([])  # Remove x-ticks for cleaner look
axes[0, 0].text(0.02, 0.02, "Capturable fraction: 0.5", transform=axes[0, 0].transAxes, fontsize=18)
axes[0, 0].text(0.02, 0.1, "$0.08/kWh", transform=axes[0, 0].transAxes, fontsize=18)


# Panel (0,1): capturable = 0.8
params = {"leak_fraction_capturable": 0.8, "electricity_price_per_kWh": elec_price}
plot_methane_savings_vary_leak_rate(
    leak_fraction_capturable=0.8,
    electricity_price_per_kWh=elec_price,
    leak_rates=np.linspace(0, 0.50, 200),
    engine_efficiency=0.45,
    plant_sizes_m3_per_day_range=(0, max_plant_size_m3_per_day),
    ogi_cost=100000,
    fig=fig,
    ax=axes[0, 1],
    levels_line=levels_line,
    cmap=shared_cmap,
    title=False,
    levels_fill=levels_fill,
    norm=shared_norm,
)

# axes[0, 1].set_title("Capturable fraction: 0.8 — vary leak rate")
axes[0, 1].text(0.02, 0.02, "Capturable frac.: 0.8", transform=axes[0, 1].transAxes, fontsize=18)
axes[0, 1].text(0.02, 0.1, "$0.08/kWh", transform=axes[0, 1].transAxes, fontsize=18)

# -------------------------
# Bottom row — $0.18/kWh 
# -------------------------

elec_price = 0.18  # $/kWh fixed for all panels

# Panel (0,0): capturable = 0.5
params = {"leak_fraction_capturable": 0.5, "electricity_price_per_kWh": elec_price}
plot_methane_savings_vary_leak_rate(
    leak_fraction_capturable=0.5,
    electricity_price_per_kWh=elec_price,
    ogi_cost=100000,
    plant_sizes_m3_per_day_range=(0, max_plant_size_m3_per_day),
    leak_rates=np.linspace(0, 0.50, 200),
    engine_efficiency=0.45,
    fig=fig,
    ax=axes[1, 0],
    levels_line=levels_line,
    levels_fill=levels_fill,
    norm=shared_norm,
    cmap=shared_cmap,
    title=False  
)
# (Optional) override/clarify title outside the function
# axes[0, 0].set_title("Capturable fraction: 0.5 — vary leak rate")
# axes[0,0].set_xticks([])  # Remove x-ticks for cleaner look
axes[1, 0].text(0.02, 0.02, "Capturable fraction: 0.5", transform=axes[1, 0].transAxes, fontsize=18)
axes[1, 0].text(0.02, 0.1, "$0.18/kWh", transform=axes[1, 0].transAxes, fontsize=18)


# Panel (0,1): capturable = 0.8
params = {"leak_fraction_capturable": 0.8, "electricity_price_per_kWh": elec_price}
plot_methane_savings_vary_leak_rate(
    leak_fraction_capturable=0.8,
    electricity_price_per_kWh=elec_price,
    leak_rates=np.linspace(0, 0.50, 200),
    engine_efficiency=0.45,
    plant_sizes_m3_per_day_range=(0, max_plant_size_m3_per_day),
    ogi_cost=100000,
    fig=fig,
    ax=axes[1, 1],
    levels_line=levels_line,
    cmap=shared_cmap,
    title=False,
    levels_fill=levels_fill,
    norm=shared_norm,
)

# axes[0, 1].set_title("Capturable fraction: 0.8 — vary leak rate")
axes[1, 1].text(0.02, 0.02, "Capturable frac.: 0.8", transform=axes[1, 1].transAxes, fontsize=18)
axes[1, 1].text(0.02, 0.1, "$0.18/kWh", transform=axes[1, 1].transAxes, fontsize=18)


#### FORMATTING ####


plt.tight_layout()

# Increase tick label size in all subplots
for ax in axes.flat:
    
    ax.tick_params(labelsize=14, direction='in', length=5, width=1.5, pad=6)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # thickness in points


# Axes and tick labels for combined figure: 

# TOP LEFT
axes[0, 0].set_xlabel(None) # Remove xlabel to avoid clutter
axes[0,0].set_xticklabels([])  # Remove x-ticks for cleaner look

# TOP RIGHT
axes[0, 1].set_xlabel(None) # Remove xlabel to avoid clutter
axes[0,1].set_xticklabels([])  # Remove x-ticks for cleaner look
axes[0,1].set_yticklabels([])  # Remove y-ticks for cleaner look
axes[0,1].set_ylabel(None)  # Remove ylabel to avoid clutter

# BOTTOM RIGHT
axes[1,1].set_yticklabels([])  # Remove x-ticks for cleaner look
axes[1,1].set_ylabel(None)  # Remove x-ticks for cleaner look

plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
plt.show()

for ax in axes.flat:
    ax.set_xlabel(ax.get_xlabel(), fontsize=16)  # bump x-axis label font size
    ax.set_ylabel(ax.get_ylabel(), fontsize=16)  # bump y-axis label font size

for ax in axes.flat:
    ax.tick_params(axis="both", which="major", labelsize=14, length=6, width=1.5, direction="in", pad=6)


# ===========================
# SAVE FIGURE
# ===========================
save_path = pathlib.Path("03_figures", "Figure_4_electricity.png")
save_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
print(f"Plot saved to: {save_path.resolve()}")

# %%

# #%% 
# # ==========================================================================
# ############### LEAK RATE ON Y-AXIS, FIX FRACTION GAS CAPTURABLE ###########
# # ==========================================================================

# #Setup
# import pandas as pd 
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import matplotlib.colors as mcolors
# from c_plotting_functions import plot_methane_savings_vary_leak_rate

# # ===========================
# # SCENARIO SETTINGS 
# # ===========================

# scenarios = [
#     {"leak_fraction_capturable": 0.5, "electricity_price_per_kWh": 0.05},
#     {"leak_fraction_capturable": 0.8, "electricity_price_per_kWh": 0.05},
#     {"leak_fraction_capturable": 0.5, "electricity_price_per_kWh": 0.10},
#     {"leak_fraction_capturable": 0.8, "electricity_price_per_kWh": 0.10},
# ]


# # ===========================
# # SHARED PLOTTING SETTINGS
# # ===========================
# vmin = 0
# vmax = 4_300_000
# levels_fill = np.linspace(vmin, vmax, 100)  
# # levels_line = np.linspace(vmin, vmax, 10)    # default evenly spaced lines
# levels_line = [250_000, 500_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000]


# # Shared cmap + BoundaryNorm to lock colors to identical Z bins
# b, g, y, o, r = "#60C1CF", "#79BF82", "#F3C354", "#F98F60", "#ED586F"
# shared_cmap = mcolors.LinearSegmentedColormap.from_list('custom_map', [o, y, g, b])
# shared_norm = mcolors.BoundaryNorm(levels_fill, ncolors=shared_cmap.N, clip=True)

# # Many smooth filled levels for the color gradient (same everywhere)
# levels_fill = np.linspace(vmin, vmax, 100)

# fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# # "Main contour lines" also fixed/consistent (same everywhere).
# # Keeping 10 lines like your current behavior:
# # levels_line = np.linspace(vmin, vmax, 10)


# # ===========================
# # MAKE FIGURE
# # ===========================

# for ax, params in zip(axes.flat, scenarios):
#     plot_methane_savings_vary_leak_rate(
#         leak_fraction_capturable=params["leak_fraction_capturable"],
#         electricity_price_per_kWh=params["electricity_price_per_kWh"],
#         leak_rates=np.linspace(0, 0.25, 200), ### <-- Set range for leak rates here
#         ogi_cost=100000,
#         fig=fig,
#         ax=ax,
#         levels_fill=levels_fill,
#         levels_line=levels_line,
#         cmap=shared_cmap,
#         norm=shared_norm
#     )
# plt.tight_layout()
# plt.show()

# # ===========================
# # SAVE FIGURE
# # ===========================

# # %%

# # ==========================================================================
# ############### CAPTURABLE FRACTION ON Y-AXIS, FIX LEAK RATE ###############
# # ==========================================================================

# import pandas as pd 
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import matplotlib.colors as mcolors
# from c_plotting_functions import plot_methane_savings_vary_capturable
# # ===========================
# # SCENARIO SETTINGS
# # ===========================

# # Four scenarios: fixed leak_rate per panel, plus electricity price variation
# scenarios = [
#     {"leak_rate": 0.05, "electricity_price_per_kWh": 0.05},
#     {"leak_rate": 0.05, "electricity_price_per_kWh": 0.10},
#     {"leak_rate": 0.10, "electricity_price_per_kWh": 0.05},
#     {"leak_rate": 0.10, "electricity_price_per_kWh": 0.10},
# ]


# # ===========================
# # SHARED PLOTTING SETTINGS
# # ===========================

# # Lock the Z-scale across all panels.
# vmin = 0
# vmax = 3_000_000  # or your desired max
# levels_fill = np.linspace(vmin, vmax, 101)  # 100 bins for positive values

# # Main contour lines (identical everywhere)
# levels_line = [50_000, 250_000, 500_000, 750_000, 1_000_000, 1_500_000]


# # Shared cmap/norm for positives
# b, g, y, o, r = "#60C1CF", "#79BF82", "#F3C354", "#F98F60", "#ED586F"
# shared_cmap = mcolors.LinearSegmentedColormap.from_list('custom_map', [o, y, g, b])
# shared_norm = mcolors.BoundaryNorm(levels_fill, ncolors=shared_cmap.N, clip=True)

# # ===========================
# # MAKE FIGURE
# # ===========================

# fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# for ax, params in zip(axes.flat, scenarios):
#     plot_methane_savings_vary_capturable(
#         leak_rate=params["leak_rate"],  # fixed per subplot
#         electricity_price_per_kWh=params["electricity_price_per_kWh"],
#         ogi_cost=100000,
#         plant_sizes_m3_per_day_range=(0, 1_200_000),
#         capturable_fraction_range=(0, 1.0),
#         resolution=200,
#         fig=fig,
#         ax=ax,
#         levels_fill=levels_fill,
#         levels_line=levels_line,
#         cmap=shared_cmap,
#         norm=shared_norm
#     )
#     ax.set_title(
#         f"Fixed leak rate: {params['leak_rate']:.0%}, "
#         f"Electricity price: ${params['electricity_price_per_kWh']}/kWh"
#     )

# plt.tight_layout()
# plt.show()

# # ===========================
# # SAVE FIGURE
# # ===========================
# # save_path = pathlib.Path("03_figures", "economic_analysis_by_capturable.png")
# # save_path.parent.mkdir(parents=True, exist_ok=True)
# # fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
# # print(f"Plot saved to: {save_path.resolve()}")
