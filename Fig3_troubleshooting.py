#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pathlib

from a_my_utilities import calc_annual_savings

# ---------- Shared helpers ----------
def _label_formatter(x):
    if abs(x) >= 1e6:   return f"${x/1e6:.0f}M"
    if abs(x) >= 1e3:   return f"${x/1e3:.0f}k"
    return f"${x:.0f}"

# Shared positive-only color mapping (negatives masked to white)
POS_MIN = 0
POS_MAX = 8_000_000  # set as you like; keep consistent with your other figs
levels_fill = np.linspace(POS_MIN, POS_MAX, 101)  # 100 bins for smooth gradient
levels_line = np.linspace(-100_000, POS_MAX, 10)  # lines can include some negatives

# Jianan-style cmap + BoundaryNorm (consistent color bins across all panels)
b, g, y, o, r = "#60C1CF", "#79BF82", "#F3C354", "#F98F60", "#ED586F"
shared_cmap = mcolors.LinearSegmentedColormap.from_list('custom_map', [r, o, y, g, b])
shared_norm = mcolors.BoundaryNorm(levels_fill, ncolors=shared_cmap.N, clip=True)


# ---------- Function A: vary leak rate; capturable fraction fixed ----------
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
    plant_sizes = np.linspace(*plant_sizes_m3_per_day_range, resolution)
    X, Y = np.meshgrid(plant_sizes, leak_rates)

    Z = np.zeros_like(X, dtype=float)
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

    # Flatten and FILTER non-finite points (NaN/Inf) before triangulation
    Xf, Yf, Zf = X.ravel(), Y.ravel(), Z.ravel()
    valid = np.isfinite(Xf) & np.isfinite(Yf) & np.isfinite(Zf)
    Xf, Yf, Zf = Xf[valid], Yf[valid], Zf[valid]

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7.3, 6))

    # White background for negatives
    ax.set_facecolor("white")
    Zpos = np.ma.masked_less(Zf, 0.0)

    # Filled contours: positives only, consistent bins/colors across panels
    ax.tricontourf(Xf, Yf, Zpos, levels=levels_fill, cmap=cmap, norm=norm)

    # Contour lines: show full range (incl. negatives)
    lines = ax.tricontour(Xf, Yf, Zf, levels=levels_line, colors='black', linewidths=2.0)
    def label_formatter(x):
        if abs(x) >= 1e6: return f"${x/1e6:.0f}M"
        if abs(x) >= 1e3: return f"${x/1e3:.0f}k"
        return f"${x:.0f}"
    ax.clabel(lines, inline=True, fontsize=10, fmt=label_formatter, rightside_up=True)

    # Breakeven line
    ax.tricontour(Xf, Yf, Zf, levels=[0], colors='black', linewidths=3.0, linestyles='solid', zorder=3)

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

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return ax



# ---------- Function B: vary capturable fraction; leak rate fixed ----------
def plot_methane_savings_contour_by_capturable(
    biogas_production_rate,
    leak_rate,
    electricity_price_per_kWh,
    ogi_cost=100000,
    plant_sizes_m3_per_day_range=(0, 1_200_000),
    capturable_fraction_range=(0, 1.0),
    resolution=200,
    fig=None,
    ax=None,
    save_path=None,
    levels_fill=None,
    levels_line=None,
    cmap=None,
    norm=None
):
    plant_sizes = np.linspace(*plant_sizes_m3_per_day_range, resolution)
    capturable_fractions = np.linspace(*capturable_fraction_range, resolution)
    X, Y = np.meshgrid(plant_sizes, capturable_fractions)

    Z = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = calc_annual_savings(
                plant_size=X[i, j],
                biogas_production_rate=biogas_production_rate,
                leak_rate=leak_rate,
                leak_fraction_capturable=Y[i, j],
                electricity_price_per_kWh=electricity_price_per_kWh,
                ogi_cost=ogi_cost
            )

    # Flatten and FILTER non-finite
    Xf, Yf, Zf = X.ravel(), Y.ravel(), Z.ravel()
    valid = np.isfinite(Xf) & np.isfinite(Yf) & np.isfinite(Zf)
    Xf, Yf, Zf = Xf[valid], Yf[valid], Zf[valid]

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7.3, 6))

    # White background for negatives
    ax.set_facecolor("white")
    Zpos = np.ma.masked_less(Zf, 0.0)

    # Filled contours
    ax.tricontourf(Xf, Yf, Zpos, levels=levels_fill, cmap=cmap, norm=norm)

    # Contour lines
    lines = ax.tricontour(Xf, Yf, Zf, levels=levels_line, colors='black', linewidths=2.0)
    def label_formatter(x):
        if abs(x) >= 1e6: return f"${x/1e6:.0f}M"
        if abs(x) >= 1e3: return f"${x/1e3:.0f}k"
        return f"${x:.0f}"
    ax.clabel(lines, inline=True, fontsize=10, fmt=label_formatter, rightside_up=True)

    # Breakeven line
    ax.tricontour(Xf, Yf, Zf, levels=[0], colors='black', linewidths=3.0, linestyles='solid', zorder=3)

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

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return ax



# ---------- Four-panel figure combining both ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
elec_price = 0.08  # $/kWh (fixed for all)

# Top row: vary leak rate, fixed capturable (0.5 and 0.8)
top_scenarios = [
    {"leak_fraction_capturable": 0.5, "title": "Capturable fraction: 0.5 (vary leak rate)"},
    {"leak_fraction_capturable": 0.8, "title": "Capturable fraction: 0.8 (vary leak rate)"},
]

# Bottom row: vary capturable fraction, fixed leak rate (5% and 10%)
bottom_scenarios = [
    {"leak_rate": 0.05, "title": "Leak rate: 5% (vary capturable fraction)"},
    {"leak_rate": 0.10, "title": "Leak rate: 10% (vary capturable fraction)"},
]

# Top row plotting
for ax, sc in zip(axes[0, :], top_scenarios):
    plot_methane_savings_contour(
        biogas_production_rate=6609 / 3785.41,
        leak_fraction_capturable=sc["leak_fraction_capturable"],
        electricity_price_per_kWh=elec_price,
        ogi_cost=100000,
        fig=fig,
        ax=ax,
        levels_fill=levels_fill,
        levels_line=levels_line,
        cmap=shared_cmap,
        norm=shared_norm
    )
    ax.set_title(sc["title"])

# Bottom row plotting
for ax, sc in zip(axes[1, :], bottom_scenarios):
    plot_methane_savings_contour_by_capturable(
        biogas_production_rate=6609 / 3785.41,
        leak_rate=sc["leak_rate"],
        electricity_price_per_kWh=elec_price,
        ogi_cost=100000,
        fig=fig,
        ax=ax,
        levels_fill=levels_fill,
        levels_line=levels_line,
        cmap=shared_cmap,
        norm=shared_norm
    )
    ax.set_title(sc["title"])

plt.tight_layout()
plt.show()

# Save if desired
# out = pathlib.Path("03_figures", "economic_analysis_mixed_4panel.png")
# out.parent.mkdir(parents=True, exist_ok=True)
# fig.savefig(out, dpi=300, bbox_inches='tight', transparent=False)
# print(f"Saved: {out.resolve()}")
#%%
