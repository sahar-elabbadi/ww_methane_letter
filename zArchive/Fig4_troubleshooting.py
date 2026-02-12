
#%%
# ==========================================================================
# ############## COMBINED FIGURE WITH BOTH VARIATIONS ###############
# ==========================================================================

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import pathlib
from c_plotting_functions import plot_methane_savings_vary_leak_rate, plot_methane_savings_vary_capturable
from a_my_utilities import Engine, ENGINES

####### Data and constants #######
# Load chp dataset 
chp_data = pd.read_csv(pathlib.Path("02_clean_data", "chp_data.csv"))

max_plant_size_m3_per_day=1_600_000

elec_price = 0.09  # $/kWh fixed for all panels - value based on calculations in Paper_Calculations.py, median electricity price for facilities with CHP
gas_price = 0.008 # $/MJ fixed for all panels - value based on calculations in Paper_Calculations.py, median natural gas price for facilities with CHP
engine = ENGINES['reciprocating_lean_burn'] # SET ENGINE TYPE HERE 

vmin = 0
vmax = 4_700_000
levels_fill = np.linspace(vmin, vmax, 100)  
# levels_line = np.linspace(vmin, vmax, 10)    # default evenly spaced lines
levels_line = [100_000, 250_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000, 4_000_000, 5_000_000]

####### Color maps #######
# Shared cmap/norm for positives
b, g, y, o, r = "#60C1CF", "#79BF82", "#F3C354", "#F98F60", "#ED586F"
shared_cmap = mcolors.LinearSegmentedColormap.from_list('custom_map', [o, y, g, b])
shared_norm = mcolors.BoundaryNorm(levels_fill, ncolors=shared_cmap.N, clip=True)

####### Set up figure #######
fig = plt.figure(figsize=(15, 14))

# Choose bottom subgrid main vs strip heights
MAIN, STRIP = 12, 2           # your current values
s = MAIN / (MAIN + STRIP)     # fraction of bottom row given to the main panel

# Make top row height == bottom main panel height
outer = fig.add_gridspec(2, 2, height_ratios=[1, 1/s], hspace=0.05, wspace=0.05)

# Top row
ax00 = fig.add_subplot(outer[0, 0])
ax01 = fig.add_subplot(outer[0, 1])

# Bottom-left subgrid (main + box strip)
gbl = outer[1, 0].subgridspec(2, 1, height_ratios=[MAIN, STRIP], hspace=0.22)
ax10 = fig.add_subplot(gbl[0, 0], sharex=ax00)
ax_box_left = fig.add_subplot(gbl[1, 0], sharex=ax10)

# Bottom-right subgrid (main + box strip)
gbr = outer[1, 1].subgridspec(2, 1, height_ratios=[MAIN, STRIP], hspace=0.22)
ax11 = fig.add_subplot(gbr[0, 0], sharex=ax01)
ax_box_right = fig.add_subplot(gbr[1, 0], sharex=ax11)

axes = np.array([[ax00, ax01], [ax10, ax11]])




# -------------------------
# Top row — vary leak rate
# -------------------------
# Panel (0,0): capturable = 0.5
params = {"leak_fraction_capturable": 0.5, "electricity_price_per_kWh": elec_price}
plot_methane_savings_vary_leak_rate(
    leak_fraction_capturable=0.5,
    electricity_price_per_kWh=elec_price,
    nat_gas_price_per_MJ=gas_price, 
    engine=engine,
    ogi_cost=100000,
    plant_sizes_m3_per_day_range=(0, max_plant_size_m3_per_day),
    leak_rates=np.linspace(0, 0.50, 200),
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
    nat_gas_price_per_MJ=gas_price, 
    leak_rates=np.linspace(0, 0.50, 200),
    engine=engine,
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
    nat_gas_price_per_MJ=gas_price, 
    engine=engine,
    ogi_cost=100000,
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
    nat_gas_price_per_MJ=gas_price, 
    engine=engine,
    ogi_cost=100000,
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

# ----------------------------------------
# Box & whisker plot — bottom row
# ----------------------------------------
for ax in [axes[1, 0], axes[1, 1]]:
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelbottom=False)

# ==== BOX & WHISKER STRIPS (CHP) ====
flows_chp = chp_data['flow_m3_per_day'].to_numpy()
# If you want to clip to plotted range so whiskers don't extend beyond:
flows_for_plot = np.clip(flows_chp, 0, max_plant_size_m3_per_day)

for axb in [ax_box_left, ax_box_right]:
    axb.boxplot(
        [flows_for_plot],
        vert=False, patch_artist=True, widths=0.8,
        boxprops=dict(facecolor='white', alpha=0.9, linewidth=1.5),
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='o', markersize=3, markerfacecolor='black',
                        markeredgecolor='none', alpha=0.6)
    )

    # Styling so it reads as a small strip
    axb.set_ylim(0.5, 1.5)
    axb.set_yticks([])
    axb.tick_params(axis='x', labelsize=12, length=5, width=1.5, direction='in', pad=4)
    for spine in axb.spines.values():
        spine.set_linewidth(2)

# Label x only on the strips (since we hid the labels on the main bottom panels)
ax_box_left.set_xlabel("Plant size (m³/day)", fontsize=14)
ax_box_right.set_xlabel("Plant size (m³/day)", fontsize=14)

# Keep y on left column only (optional)
ax_box_right.set_ylabel(None)

# Do your existing tick/label styling for main panels...
for ax in axes.flat:
    ax.tick_params(labelsize=14, direction='in', length=6, width=1.5, pad=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2)


for axb in [ax_box_left, ax_box_right]:
    # Hide the border (spines) around the subplot
    for spine in axb.spines.values():
        spine.set_visible(False)
        
    # Optional: remove ticks entirely for an ultra-clean look
    axb.tick_params(bottom=False, top=False)

# --- Label below the box-and-whisker strip ---
for axb in [ax_box_left, ax_box_right]:
    label = "CHP facility flows"
    axb.text(
        0.5, .1,                     # centered horizontally, below the axis
        label,
        transform=axb.transAxes,
        ha='center', va='top',
        fontsize=13, color='black'
    )

from matplotlib.ticker import FuncFormatter, MaxNLocator

# === Format numbers on x-axis (for bottom row only) ===
def _fmt_m3_day(x, pos):
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    elif x >= 1_000:
        return f"{x/1_000:.0f}k"
    else:
        return f"{int(x)}"

# Shared x-limits across all main and box axes
for ax in [ax00, ax01, ax10, ax11, ax_box_left, ax_box_right]:
    ax.set_xlim(0, max_plant_size_m3_per_day)

# ✅ Only bottom row gets x tick labels and formatter
for ax in [ax10, ax11]:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_m3_day))
    ax.tick_params(axis='x', labelbottom=True, pad=4)
    ax.set_xlabel("Plant size (m³/day)", fontsize=14, labelpad=2)

# ❌ Hide x tick labels for the top row
for ax in [ax00, ax01]:
    ax.tick_params(axis='x', labelbottom=False)

# ❌ Hide x tick labels on box strips (so numbers appear only above them)
for axb in [ax_box_left, ax_box_right]:
    axb.tick_params(axis='x', labelbottom=False)
    axb.set_xlabel(None)


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

