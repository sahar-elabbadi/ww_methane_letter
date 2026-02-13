#%%

# Combined Fig 1 

# Setup 
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pathlib

from a_my_utilities import (
    set_chini_dataset,
    get_chini_stats,
    chini_confidence_intervals,
    get_chini_xy,
    mgd_to_m3_per_day,
    METHANE_KG_PER_SCF,
    
    BIOGAS_FRACTION_CH4,
)

# formatting tick labels 
def label_formatter(x, pos=None):
    if abs(x) >= 1e6:
        return f"{x/1e6:.1f}M"
    elif abs(x) >=1e5: 
        return f"{x/1e3:.0f}k"
    elif abs(x) >= 1e3:
        return f"{x/1e3:.1f}k"
    elif abs(x) >= 100: 
        return f"{x:.0f}"
    elif x == 0: 
        return f"{x:.0f}"
    else:
        return f"{x:.2f}"
    
# =========================
# Data prep (LEFT panel)
# =========================
chini_data = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))
set_chini_dataset(
    chini_data,
    x_col="flow_m3_per_day",
    y_col="methane_gen_kgh",
    drop_negative=True,
)

# =========================
# Data prep (RIGHT panels)
# =========================
eugene = pd.read_csv(pathlib.Path("01_raw_data", "chini-biogas", "EugeneOR.csv"))
eugene["date"] = pd.to_datetime(eugene["date"])
eugene["flow_m3_per_day"] = eugene.apply(lambda r: mgd_to_m3_per_day(r["flow_mgd"]), axis=1)
eugene["methane_gen_kg_per_day"] = (
    eugene["biogas_produced_scfd"] * METHANE_KG_PER_SCF * BIOGAS_FRACTION_CH4
)

# Daily x as day-of-year
x_daily = eugene["date"].dt.dayofyear.to_numpy()

# Daily ratio (kg CH4 per m^3 water)
eugene["ratio"] = eugene["methane_gen_kg_per_day"] / eugene["flow_m3_per_day"]

# Year (assumes single-year dataset)
year = int(eugene["date"].dt.year.mode().iloc[0])

# Monthly average ratio
monthly = (
    eugene.assign(month=eugene["date"].dt.month)
    .groupby("month", as_index=False)["ratio"].mean()
    .rename(columns={"ratio": "avg_ratio"})
)
monthly = monthly.set_index("month").reindex(range(1, 13)).reset_index()

# X positions for the 15th of each month
month_center_dates = pd.date_range(start=f"{year}-01-01", periods=12, freq="MS") + pd.Timedelta(days=14)
month_centers = month_center_dates.dayofyear.to_numpy()

# Month tick positions/labels
month_starts = pd.date_range(start=f"{year}-01-01", periods=12, freq="MS")
month_start_days = month_starts.dayofyear.to_numpy()
month_labels = month_starts.strftime("%b").to_list()

# =========================
# Figure & layout
# =========================
plt.rcParams.update({"mathtext.default": "regular"})  # consistent math text

fig = plt.figure(figsize=(17, 7), constrained_layout=True)
# 2 columns, 2 rows overall; left takes both rows, right splits into 2 stacked axes

gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[1, 1], height_ratios=[1, 1])

# Left: occupies both rows, col 0
ax_left = fig.add_subplot(gs[:, 0])

# Right: subgridspec in col 1 with two rows (top smaller, bottom larger)
gs_right = gs[:, 1].subgridspec(2, 1, height_ratios=[1, 4], hspace=0.15)
ax_top = fig.add_subplot(gs_right[0, 0])
ax_bot = fig.add_subplot(gs_right[1, 0])
ax2 = ax_bot.twinx()  # methane on secondary y-axis


# =========================
# LEFT: Chini regression with intervals
# =========================
# Stats & data
stats_cached = get_chini_stats()
slope = stats_cached["slope"]
r2_origin = stats_cached["r2"]
x, y = get_chini_xy()
n = len(x)

# Grid and intervals
n_points = 200
x_grid = np.linspace(0.0, x.max() * 1.05, n_points)
lower_floor = 1e-6
ci = chini_confidence_intervals(x_grid, alpha=0.05)
ci_lower = np.maximum(lower_floor, ci["lower_ci"])
pi_lower = np.maximum(lower_floor, ci["lower_pi"])

eq_text = f"$\\hat{{y}} = {slope:.5f}\\,x$  [R$^2$ = {r2_origin:.3f}, n={n}]"

# Bands
ax_left.fill_between(
    x_grid, ci_lower, ci["upper_ci"],
    color="#60C1CF", alpha=0.30,
    label="95% confidence interval (mean)",
)
ax_left.fill_between(
    x_grid, pi_lower, ci["upper_pi"],
    color="#79BF82", alpha=0.20,
    label="95% prediction interval",
)

# Regression line
ax_left.plot(x_grid, ci["estimate"], color="black", linewidth=3, label=eq_text)

# Scatter (fill then outline)
ax_left.scatter(x, y, s=50, facecolors="black", edgecolors="none", alpha=0.4)
ax_left.scatter(x, y, s=50, facecolors="none", edgecolors="black", linewidths=1.2)

# Labels & ticks
ax_left.set_xlabel("Flow (m³/day)", fontsize=16, labelpad=6)
ax_left.set_ylabel("Biogas production (kg CH₄/h)", fontsize=16, labelpad=6)

ax_left.tick_params(axis="both", which="both", direction="in", length=6, width=1, labelsize=14, pad=8)
ax_left.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax_left.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

# Axes limits & spines
ax_left.set_xlim(0, x_grid[-1])  # no right padding beyond grid
ax_left.set_ylim(bottom=0)
ax_left.spines["top"].set_visible(False)
ax_left.spines["right"].set_visible(False)
ax_left.spines["bottom"].set_linewidth(2)
ax_left.spines["left"].set_linewidth(2)

# Legend (no box)
ax_left.legend(fontsize=13, frameon=False, loc="upper left")

# =========================
# RIGHT TOP: Monthly avg ratio
# =========================
label_fs = 16
tick_fs = 14

ax_top.scatter(
    month_centers,
    monthly["avg_ratio"].to_numpy(),
    s=60, color="tab:purple", alpha=0.85,
    label="Monthly Avg Methane/Flow",
)
ax_top.text(
    0.05, 0.85,   # (x, y) in axes coordinates (0=left, 1=top)
    "Flow-normalized biogas production (kg CH₄/m³)",
    transform=ax_top.transAxes,
    ha="left", va="bottom",
    fontsize=label_fs, color="black"
)
ax_top.set_xlim(1, 366)
ax_top.set_ylim(0, 0.1)
ax_top.set_xticks(month_start_days)
ax_top.set_xticklabels([])
ax_top.set_yticks([0.00, 0.04, 0.08])
ax_top.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

# Ticks & spines
ax_top.tick_params(axis="both", which="both", direction="in", length=5, labelsize=tick_fs, pad=6)
for spine in ax_top.spines.values():
    spine.set_linewidth(1.5)

ax_top.spines["top"].set_visible(False)
ax_top.spines["right"].set_visible(False)
ax_top.spines["bottom"].set_linewidth(2)
ax_top.spines["left"].set_linewidth(2)

# =========================
# RIGHT BOTTOM: Flow (left y) + Methane (right y)
# =========================
ln1 = ax_bot.scatter(
    x_daily, eugene["flow_m3_per_day"],
    color="tab:blue", alpha=0.6, s=60, label="Flow (m³/day)"
)
ax_bot.set_ylabel("Reported daily flow\n(m³/day)", fontsize=label_fs, color="black", labelpad=6)
ax_bot.set_xlim(1, 366)
ax_bot.set_xticks(month_start_days)
ax_bot.set_xticklabels(month_labels, fontsize=tick_fs, color="black")
ax_bot.set_xlabel(f"Month ({year})", fontsize=label_fs, color="black", labelpad=6)
ax_bot.tick_params(axis="y", colors="black", labelsize=tick_fs, direction="in")
ax_bot.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax_bot.set_ylim(0, 800_000)

ln2 = ax2.scatter(
    x_daily, eugene["methane_gen_kg_per_day"],
    color="tab:green", alpha=0.6, s=60, label="Methane (kg CH₄/day)"
)
ax2.set_ylabel("Reported biogas production\n(kg CH₄/day)",
               fontsize=label_fs, color="black",
               rotation=270, va="bottom", ha="center", labelpad=16)
ax2.tick_params(axis="y", colors="black", labelsize=tick_fs, direction="in")
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.1f}k"))

# Ticks & spines
for ax in (ax_bot, ax2):
    ax.tick_params(axis="both", which="both", direction="in", length=5, labelsize=tick_fs, pad=6)

for ax in (ax_bot,):
    for spine in ax.spines.values():
        spine.set_linewidth(2)

for spine in ax2.spines.values():
    spine.set_linewidth(2)

ax_bot.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)



# Combined legend (no box)
ax_bot.legend(handles=[ln1, ln2], loc="upper right", frameon=False, fontsize=14)


# Axes labels formatting 

for ax in [ax_left, ax_top, ax_bot, ax2]:
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(label_formatter))
# =========================
# Layout & save/show
# =========================
fig.tight_layout()
fig.subplots_adjust(top=.9999)  # nudge down slightly to avoid cutoff
save_path = pathlib.Path("03_figures", "fig1_combined.png")
fig.savefig(save_path, dpi=400, bbox_inches="tight", pad_inches=0.2)
plt.show()
