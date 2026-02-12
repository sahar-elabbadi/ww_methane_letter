#%% 

#### SETUP ####
import pandas as pd
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from a_my_utilities import mgd_to_m3_per_day, METHANE_KG_PER_SCF, BIOGAS_FRACTION_CH4

euguene_data = pd.read_csv(pathlib.Path("01_raw_data", "chini-biogas", "EugeneOR.csv")) 
euguene_data["date"] = pd.to_datetime(euguene_data["date"])
euguene_data['flow_m3_per_day'] = euguene_data.apply(lambda row: mgd_to_m3_per_day(row['flow_mgd']), axis=1)
euguene_data['methane_gen_kg_per_day'] = euguene_data['biogas_produced_scfd']* METHANE_KG_PER_SCF * BIOGAS_FRACTION_CH4

# Ensure datetime
euguene_data["date"] = pd.to_datetime(euguene_data["date"])

# Daily x as day-of-year
x_daily = euguene_data["date"].dt.dayofyear.to_numpy()

# Daily ratio (kg CH4 per m^3 water)
euguene_data["ratio"] = euguene_data["methane_gen_kg_per_day"] / euguene_data["flow_m3_per_day"]

# Infer year (cast to int to avoid float issues)
year = int(euguene_data["date"].dt.year.mode().iloc[0])

# Monthly average ratio (methane/flow)
monthly = (
    euguene_data
    .assign(month=euguene_data["date"].dt.month)
    .groupby("month", as_index=False)["ratio"].mean()
    .rename(columns={"ratio": "avg_ratio"})
)

# Reindex to all 12 months to preserve order; NaN where month missing
monthly = monthly.set_index("month").reindex(range(1, 13)).reset_index()

# Compute x-positions for the 15th of each month using date_range (avoids dtype issues)
month_center_dates = (
    pd.date_range(start=f"{year}-01-01", periods=12, freq="MS")  # 1st of each month
    + pd.Timedelta(days=14)                                     # -> 15th
)
month_centers = month_center_dates.dayofyear.to_numpy()

######### PLOTTING #########

# --- Figure with two stacked panels ---
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [1, 3]}
)

# --- Shared x-axis formatting ---
label_fontsize = 18
tick_fontsize = 16

# --- Top panel: monthly ratio ---
ax_top.scatter(
    month_centers,
    monthly["avg_ratio"].to_numpy(),
    color="tab:purple",
    s=60,          # marker size (scatter uses 's' instead of markersize)
    alpha=0.85,
    label="Monthly Avg Methane/Flow"
)

ax_top.set_ylabel("Flow normalized\nbiogas production\n(kg CH₄/m³)", fontsize=label_fontsize, color="black")
ax_top.tick_params(axis="y", colors="black", labelsize=tick_fontsize, direction="in")
ax_top.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.4f}"))
# ax_top.legend(loc="upper right", frameon=False, fontsize=12)

# --- Bottom panel: flow + methane ---
color1 = "tab:blue"
ax_bot.set_ylabel("Flow (m³/day)", fontsize=label_fontsize, color="black")
# Flow as points (left y-axis)
ln1 = ax_bot.scatter(
    x_daily,
    euguene_data["flow_m3_per_day"],
    color="tab:blue",
    alpha=0.6,
    s=60,
    label="Flow (m³/day)"
)
ax_bot.set_ylabel("Reported daily flow\n (m³/day)", fontsize=label_fontsize, color="black")
ax_bot.tick_params(axis="y", colors="black", labelsize=tick_fontsize, direction="in")
ax_bot.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

# Secondary y-axis for methane
ax2 = ax_bot.twinx()
color2 = "tab:green"
ax2.set_ylabel("Methane Produced (kg CH₄/day)", fontsize=label_fontsize, color="black")
# Methane as points (right y-axis)
ln2 = ax2.scatter(
    x_daily,
    euguene_data["methane_gen_kg_per_day"],
    color="tab:green",
    alpha=0.6,
    s=60,
    label="Methane (kg CH₄/day)"
)
ax2.set_ylabel("Reported biogas production\n(kg CH₄/day)", 
               fontsize=label_fontsize, 
               color="black", 
               rotation=270,
               va="bottom",
               ha="center")
ax2.tick_params(axis="y", colors="black", labelsize=tick_fontsize, direction="in")
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))



# Compute first day-of-year for each month in your dataset's year
month_starts = pd.date_range(start=f"{year}-01-01", periods=12, freq="MS")
month_start_days = month_starts.dayofyear.to_numpy()
month_labels = month_starts.strftime("%b").to_list()  # "Jan", "Feb", ...


# BOTTOM PANEL X-AXIS
ax_bot.set_xlim(1, 366)
ax_bot.set_xticks(month_start_days)
ax_bot.set_xticklabels(month_labels, fontsize=tick_fontsize, color="black")
ax_bot.set_xlabel("Month (2012)", fontsize=label_fontsize, color="black")

ax_bot.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k")
)

# Right y-axis (Methane)
ax2.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f"{x/1000:.1f}k")
)


# BOTTOM PANEL Y-AXIS
ax_bot.set_ylim(0, 800_000)  # Adjust based on your data

# TOP PANEL X_AXIS
ax_top.set_xlim(1, 366)
ax_top.set_ylim(0, 0.08)
ax_top.set_xticks(month_start_days)
ax_top.set_xticklabels(month_labels, fontsize=tick_fontsize, color="black")

ax_top.set_yticks([0.00, 0.04, 0.08])
ax_top.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

# Tick formatting 
for ax in [ax_top, ax_bot, ax2]:
    ax.tick_params(axis="both", direction="in", length=5)



# --- Legends ---
ax_bot.legend(handles=[ln1, ln2], loc="upper right", frameon=False, fontsize=label_fontsize)


for ax in [ax_top, ax_bot, ax2]:
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)   # default is ~0.8; try 1.5–2 for a bolder frame


# --- Titles ---
# ax_top.set_title("Monthly Avg Methane-to-Flow Ratio", fontsize=16, color="black", pad=10)
# ax_bot.set_title("Daily Flow and Methane Production (2012)", fontsize=16, color="black", pad=10)

fig.tight_layout(h_pad=2)
plt.show()

# %%
