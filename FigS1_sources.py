# supplementary_measurement_figure_swapped_encodings.py
# Color = AD (left) / Biogas availability (right)
# Shape = Source
# Dual legends with manual bbox placement
# Saves: 03_figures/supplementary_measurement_data.png

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from scipy.stats import linregress
import json
from matplotlib.lines import Line2D

# ===== Legend placement (edit these manually) =====
LEGEND_POS = {
    "left_subplot": {
        "color": {"loc": "upper left", "bbox_to_anchor": None},  # e.g., (1.02, 1)
        "shape": {"loc": "lower left", "bbox_to_anchor": None},  # e.g., (1.02, 0.75)
        "color_title": "Anaerobic Digestion",
        "shape_title": "Source",
    },
    "right_subplot": {
        "color": {"loc": "upper left", "bbox_to_anchor": None},
        "shape": {"loc": "lower left", "bbox_to_anchor": None},
        "color_title": "Data Availability",
        "shape_title": "Source",
    },
}
# ===================================================

# --- project utilities (your originals) ---
from a_my_utilities import (
    load_ch4_emissions_data,
    set_chini_dataset,
    calc_biogas_production_rate,
    load_ch4_emissions_with_ad_only,
    calculate_production_normalized_ch4,
)

# =========================
# Data loading
# =========================
measurement_data = load_ch4_emissions_data()

chini_data = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))
set_chini_dataset(
    chini_data, x_col="flow_m3_per_day", y_col="methane_gen_kgh", drop_negative=True
)

measurement_data_ad = calculate_production_normalized_ch4(
    load_data_func=load_ch4_emissions_with_ad_only,
    calc_biogas_func=calc_biogas_production_rate,
)

measurement_data_ad = measurement_data_ad[
    (measurement_data_ad["biogas_production_used_kgCH4_per_hr"] > 0)
    & (measurement_data_ad["production_normalized_CH4_percent"] > 0)
].copy()

measurement_data_ad["data_availability"] = (
    measurement_data_ad["reported_biogas_production"]
    .fillna("")
    .apply(
        lambda x: "Biogas data available"
        if str(x).lower() == "yes"
        else "Biogas production interpolated from flow"
    )
)

# =========================
# Helpers
# =========================

def _powerlaw_fit(x, y):
    logx, logy = np.log(x), np.log(y)
    res = linregress(logx, logy)
    return {"a": np.exp(res.intercept), "b": res.slope, "r2_loglog": res.rvalue**2}


def _percent_formatter(y, _):
    pct = y * 100
    if pct >= 1:
        return f"{pct:.0f}%"
    elif pct >= 0.1:
        return f"{pct:.1f}%"
    else:
        return f"{pct:.2f}%"


def _make_color_palettes():
    # color for AD or Data Availability
    return (
        {"Anaerobic digestion": "#1f77b4", "No anaerobic digestion": "#ff7f0e"},
        {
            "Biogas data available": "#E24A33",
            "Biogas production interpolated from flow": "#226f90",
        },
    )


def _make_markers_for_sources(df_left, df_right):
    all_sources = (
        pd.concat([df_left["source"], df_right["source"]], axis=0)
        .fillna("Unknown")
        .astype(str)
    )
    unique_sources = all_sources.unique().tolist()
    marker_shapes = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<", ">"]
    return {src: marker_shapes[i % len(marker_shapes)] for i, src in enumerate(unique_sources)}


def _add_dual_legends(
    ax,
    *,
    color_palette,
    color_values,
    color_title,
    marker_dict,
    marker_title,
    color_loc="upper left",
    color_bbox=None,
    shape_loc="lower left",
    shape_bbox=None,
):
    # Color legend
    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color_palette[c],
            markeredgecolor="k",
            markersize=8,
            label=c,
        )
        for c in color_values
        if c in color_palette
    ]
    leg1 = ax.legend(
        handles=color_handles,
        title=color_title,
        loc=color_loc,
        bbox_to_anchor=color_bbox,
        frameon=False,
        fontsize=11,
        title_fontsize=12,
    )
    ax.add_artist(leg1)

    # Shape legend (source)
    shape_handles = [
        Line2D(
            [0],
            [0],
            marker=m,
            linestyle="None",
            markerfacecolor="lightgray",
            markeredgecolor="k",
            markersize=8,
            label=s,
        )
        for s, m in marker_dict.items()
    ]
    ax.legend(
        handles=shape_handles,
        title=marker_title,
        loc=shape_loc,
        bbox_to_anchor=shape_bbox,
        frameon=False,
        fontsize=11,
        title_fontsize=12,
    )


# =========================
# Plot 1 – Emissions vs Flow
# =========================
def plot_emissions_vs_flow_ax(ax, df, color_palette, markers_dict, legend_cfg):
    data = df.copy()
    data["source"] = data["source"].fillna("Unknown")
    data["has_ad_norm"] = data["has_ad"].astype(str).str.lower()
    data["ad_label"] = np.where(
        data["has_ad_norm"] == "yes", "Anaerobic digestion", "No anaerobic digestion"
    )
    data = data[(data["flow_m3_per_day"] > 0) & (data["ch4_kg_per_hr"] > 0)].copy()

    sns.scatterplot(
        ax=ax,
        data=data,
        x="flow_m3_per_day",
        y="ch4_kg_per_hr",
        hue="ad_label",
        palette=color_palette,
        style="source",
        markers=markers_dict,
        s=80,
        edgecolor="k",
        alpha=0.85,
        legend=False,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Flow (m³/day)", fontsize=16)
    ax.set_ylabel("CH₄ Emissions (kg/hr)", fontsize=16)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    # Trendlines (gray)
    for label, color in color_palette.items():
        sub = data[data["ad_label"] == label]
        if len(sub) < 3:
            continue
        fit = _powerlaw_fit(sub["flow_m3_per_day"], sub["ch4_kg_per_hr"])
        xv = np.geomspace(sub["flow_m3_per_day"].min(), sub["flow_m3_per_day"].max(), 200)
        ax.plot(xv, fit["a"] * xv ** fit["b"], lw=2.5, color=color, label="_nolegend_")

    _add_dual_legends(
        ax,
        color_palette=color_palette,
        color_values=list(color_palette.keys()),
        color_title=legend_cfg["color_title"],
        marker_dict=markers_dict,
        marker_title=legend_cfg["shape_title"],
        color_loc=legend_cfg["color"]["loc"],
        color_bbox=legend_cfg["color"]["bbox_to_anchor"],
        shape_loc=legend_cfg["shape"]["loc"],
        shape_bbox=legend_cfg["shape"]["bbox_to_anchor"],
    )


# =========================
# Plot 2 – Production-normalized CH₄ vs Biogas
# =========================
def plot_prod_norm_vs_biogas_ax(ax, df, color_palette, markers_dict, legend_cfg):
    data = df.copy()
    data["source"] = data["source"].fillna("Unknown")

    sns.scatterplot(
        ax=ax,
        data=data,
        x="biogas_production_used_kgCH4_per_hr",
        y="production_normalized_CH4_percent",
        hue="data_availability",
        palette=color_palette,
        style="source",
        markers=markers_dict,
        s=80,
        edgecolor="k",
        alpha=0.85,
        legend=False,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(_percent_formatter)
    ax.set_xlabel("Biogas production rate (kg CH₄/hr)", fontsize=16)
    ax.set_ylabel("Production Normalized CH₄ Emissions (%)", fontsize=16)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    # Trendlines
    for label, color in color_palette.items():
        sub = data[data["data_availability"] == label]
        if len(sub) < 3:
            continue
        fit = _powerlaw_fit(
            sub["biogas_production_used_kgCH4_per_hr"],
            sub["production_normalized_CH4_percent"],
        )
        xv = np.geomspace(
            sub["biogas_production_used_kgCH4_per_hr"].min(),
            sub["biogas_production_used_kgCH4_per_hr"].max(),
            200,
        )
        ax.plot(xv, fit["a"] * xv ** fit["b"], lw=2.5, color=color, label="_nolegend_")

    _add_dual_legends(
        ax,
        color_palette=color_palette,
        color_values=list(color_palette.keys()),
        color_title=legend_cfg["color_title"],
        marker_dict=markers_dict,
        marker_title=legend_cfg["shape_title"],
        color_loc=legend_cfg["color"]["loc"],
        color_bbox=legend_cfg["color"]["bbox_to_anchor"],
        shape_loc=legend_cfg["shape"]["loc"],
        shape_bbox=legend_cfg["shape"]["bbox_to_anchor"],
    )


# =========================
# Main
# =========================
def main():
    save_dir = pathlib.Path("03_figures")
    save_dir.mkdir(parents=True, exist_ok=True)

    color_pal_left, color_pal_right = _make_color_palettes()
    markers_dict = _make_markers_for_sources(measurement_data, measurement_data_ad)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    plot_emissions_vs_flow_ax(
        ax=axes[0],
        df=measurement_data,
        color_palette=color_pal_left,
        markers_dict=markers_dict,
        legend_cfg=LEGEND_POS["left_subplot"],
    )

    plot_prod_norm_vs_biogas_ax(
        ax=axes[1],
        df=measurement_data_ad,
        color_palette=color_pal_right,
        markers_dict=markers_dict,
        legend_cfg=LEGEND_POS["right_subplot"],
    )

    plt.tight_layout()
    outpath = save_dir / "supplementary_measurement_data.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")

    # Show in GUI
    plt.show()

    print(f"\nSaved figure to: {outpath}")


if __name__ == "__main__":
    main()


#%%

# supplementary_measurement_data_stacked.py
# 4-row stacked version:
# Row 1: All data (color = AD / data availability; shape = source)
# Row 2: Moore et al. only (shapes separate "Moore 2023" vs "Moore 2025")
# Row 3: Song et al. only (shape = source)
# Row 4: Fredenslund & Gålfalk only (right plot only has data; shape = source)
# Saves: 03_figures/supplementary_measurement_data_stacked.png

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from scipy.stats import linregress
from matplotlib.lines import Line2D
from matplotlib.legend import Legend


# ---- Project utilities (as in your environment) ----
from a_my_utilities import (
    load_ch4_emissions_data,
    set_chini_dataset,
    calc_biogas_production_rate,
    load_ch4_emissions_with_ad_only,
    calculate_production_normalized_ch4,
)

# =========================
# Data loading & preparation
# =========================
measurement_data = load_ch4_emissions_data()

chini_data = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))
set_chini_dataset(
    chini_data, x_col="flow_m3_per_day", y_col="methane_gen_kgh", drop_negative=True
)

measurement_data_ad = calculate_production_normalized_ch4(
    load_data_func=load_ch4_emissions_with_ad_only,
    calc_biogas_func=calc_biogas_production_rate,
)

# Keep positive values only for AD dataframe (right plots)
measurement_data_ad = measurement_data_ad[
    (measurement_data_ad["biogas_production_used_kgCH4_per_hr"] > 0)
    & (measurement_data_ad["production_normalized_CH4_percent"] > 0)
].copy()

# Data availability label from reported_biogas_production
measurement_data_ad["data_availability"] = (
    measurement_data_ad["reported_biogas_production"]
    .fillna("")
    .apply(
        lambda x: "Biogas data available"
        if str(x).lower() == "yes"
        else "Biogas production interpolated from flow"
    )
)

# Common cleanups
for df in (measurement_data, measurement_data_ad):
    df["source"] = df["source"].fillna("Unknown").astype(str)

# =========================
# Helpers
# =========================
def _powerlaw_fit(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 2:
        return None
    res = linregress(np.log(x[m]), np.log(y[m]))
    return {"a": float(np.exp(res.intercept)), "b": float(res.slope), "r2_loglog": float(res.rvalue**2)}

def _percent_formatter(y, _):
    pct = y * 100
    if pct >= 1: return f"{pct:.0f}%"
    if pct >= 0.1: return f"{pct:.1f}%"
    return f"{pct:.2f}%"

# Color palettes for the *color* encodings
COLOR_AD = {"Anaerobic digestion": "#1f77b4", "No anaerobic digestion": "#ff7f0e"}
COLOR_AVAIL = {
    "Biogas data available": "#E24A33",
    "Biogas production interpolated from flow": "#226f90",
}

# Marker (shape) assignment for *source* encodings (cycled safely)
MARKER_CYCLE = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<", ">"]
def markers_for_sources(unique_sources):
    return {src: MARKER_CYCLE[i % len(MARKER_CYCLE)] for i, src in enumerate(unique_sources)}

# Legend builder: separate box for color and shape
def add_dual_legends(
    ax, *, color_palette, color_values, color_title, marker_dict, marker_title,
    color_loc="upper left", color_bbox=None, shape_loc="lower left", shape_bbox=None,
    fontsize=10
):
    color_handles = [
        Line2D([0],[0], marker="o", linestyle="None",
               markerfacecolor=color_palette[c], markeredgecolor="k",
               markersize=7, label=c)
        for c in color_values if c in color_palette
    ]
    leg1 = ax.legend(
        handles=color_handles, title=color_title, loc=color_loc, bbox_to_anchor=color_bbox,
        frameon=False, fontsize=fontsize, title_fontsize=fontsize+1
    )
    ax.add_artist(leg1)

    shape_handles = [
        Line2D([0],[0], marker=m, linestyle="None",
               markerfacecolor="lightgray", markeredgecolor="k",
               markersize=7, label=s)
        for s, m in marker_dict.items()
    ]
    ax.legend(
        handles=shape_handles, title=marker_title, loc=shape_loc, bbox_to_anchor=shape_bbox,
        frameon=False, fontsize=fontsize, title_fontsize=fontsize+1
    )

# Normalize AD label on *left* dataframes
def add_ad_label(df_left):
    df = df_left.copy()
    if "has_ad" in df.columns:
        df["has_ad_norm"] = df["has_ad"].astype(str).str.lower()
        df["ad_label"] = np.where(df["has_ad_norm"] == "yes",
                                  "Anaerobic digestion", "No anaerobic digestion")
    else:
        df["ad_label"] = "Unknown"
    return df

# =========================
# Row filters / labelers
# =========================
def filter_all_left(df):     # measurement_data
    return df[(df["flow_m3_per_day"] > 0) & (df["ch4_kg_per_hr"] > 0)].copy()

def filter_all_right(df):    # measurement_data_ad
    return df.copy()

def filter_moore(df):
    # Select any "Moore" appearance
    return df[df["source"].str.contains(r"\bMoore\b", case=False, na=False)].copy()

def label_moore_variant(df):
    # Collapse source variants to shape labels Moore 2023 vs Moore 2025 (fallback "Moore (other)")
    src = df["source"].astype(str)
    variant = np.where(src.str.contains("2023"), "Moore 2023",
                       np.where(src.str.contains("2025"), "Moore 2025", "Moore (other)"))
    df = df.copy()
    df["source_label"] = variant
    return df

def filter_song(df):
    return df[df["source"].str.contains(r"\bSong\b", case=False, na=False)].copy()

def label_source_simple(df):
    # Use the source string directly as shape label (good for Song-only or general)
    df = df.copy()
    df["source_label"] = df["source"].astype(str)
    return df

def filter_fred_galfalk(df):
    # Fredenslund and Gålfalk*; handle 'Galfalk' without diacritic as well
    return df[
        df["source"].str.contains("Fredenslund", case=False, na=False)
        | df["source"].str.contains("Gålfalk", case=False, na=False)
        | df["source"].str.contains("Galfalk", case=False, na=False)
    ].copy()

def label_fred_galfalk(df):
    df = df.copy()
    src = df["source"].astype(str)
    lab = np.where(src.str.contains("Fredenslund", case=False), "Fredenslund et al.",
                   np.where(src.str.contains("Gålfalk", case=False) | src.str.contains("Galfalk", case=False),
                            "Gålfalk et al.", "Other"))
    df["source_label"] = lab
    return df

# =========================
# Plot functions (per axis)
# =========================
def plot_left(ax, df_left, color_by="ad_label", color_palette=COLOR_AD,
              shape_by="source_label", markers_map=None, title=None):
    if df_left.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No flow data for this subset", ha="center", va="center", fontsize=11)
        return

    df = add_ad_label(df_left)
    if shape_by not in df.columns:
        df[shape_by] = df["source"]  # default

    # Scatter: color = AD; shape = source_label
    sns.scatterplot(
        ax=ax,
        data=df,
        x="flow_m3_per_day",
        y="ch4_kg_per_hr",
        hue=color_by,
        palette=color_palette,
        style=shape_by,
        markers=markers_map,
        s=55,
        edgecolor="k",
        alpha=0.85,
        legend=False,
    )
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Flow (m³/day)", fontsize=12)
    ax.set_ylabel("CH₄ (kg/hr)", fontsize=12)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=10)

    # Trendlines by AD color
    for label, col in color_palette.items():
        sub = df[df[color_by] == label]
        fit = _powerlaw_fit(sub["flow_m3_per_day"], sub["ch4_kg_per_hr"])
        if fit is None: continue
        xv = np.geomspace(sub["flow_m3_per_day"].min(), sub["flow_m3_per_day"].max(), 200)
        ax.plot(xv, fit["a"] * xv**fit["b"], lw=2, color=col)

    if title:
        ax.set_title(title, fontsize=13)

    # Legends
    marker_dict = markers_map or {}
    add_dual_legends(
        ax,
        color_palette=color_palette,
        color_values=list(color_palette.keys()),
        color_title="Anaerobic Digestion",
        marker_dict=marker_dict,
        marker_title="Source",
        color_loc="upper left",
        shape_loc="lower left",
        fontsize=9
    )

def plot_right(ax, df_right, color_by="data_availability", color_palette=COLOR_AVAIL,
               shape_by="source_label", markers_map=None, title=None):
    if df_right.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No data for this subset", ha="center", va="center", fontsize=11)
        return

    df = df_right.copy()
    if shape_by not in df.columns:
        df[shape_by] = df["source"]

    sns.scatterplot(
        ax=ax,
        data=df,
        x="biogas_production_used_kgCH4_per_hr",
        y="production_normalized_CH4_percent",
        hue=color_by,
        palette=color_palette,
        style=shape_by,
        markers=markers_map,
        s=55,
        edgecolor="k",
        alpha=0.85,
        legend=False,
    )
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.yaxis.set_major_formatter(_percent_formatter)
    ax.set_xlabel("Biogas (kg CH₄/hr)", fontsize=12)
    ax.set_ylabel("Prod-normalized CH₄ (%)", fontsize=12)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=10)

    # Trendlines by availability color
    for label, col in color_palette.items():
        sub = df[df[color_by] == label]
        fit = _powerlaw_fit(sub["biogas_production_used_kgCH4_per_hr"], sub["production_normalized_CH4_percent"])
        if fit is None: continue
        xv = np.geomspace(sub["biogas_production_used_kgCH4_per_hr"].min(),
                          sub["biogas_production_used_kgCH4_per_hr"].max(), 200)
        ax.plot(xv, fit["a"] * xv**fit["b"], lw=2, color=col)

    if title:
        ax.set_title(title, fontsize=13)

    marker_dict = markers_map or {}
    add_dual_legends(
        ax,
        color_palette=color_palette,
        color_values=list(color_palette.keys()),
        color_title="Data Availability",
        marker_dict=marker_dict,
        marker_title="Source",
        color_loc="upper left",
        shape_loc="lower left",
        fontsize=9
    )

# =========================
# Build figure with 4 rows × 2 columns
# =========================
def main():
    save_dir = pathlib.Path("03_figures")
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(17, 22), constrained_layout=True)

    # ---------- Row 1: All data ----------
    left1 = filter_all_left(measurement_data)
    right1 = filter_all_right(measurement_data_ad)

    # Shapes = all unique sources in each side (can differ slightly)
    markers_left1 = markers_for_sources(left1["source"].unique())
    markers_right1 = markers_for_sources(right1["source"].unique())

    plot_left(axes[0,0], left1, color_palette=COLOR_AD, shape_by="source_label",
              markers_map={s: markers_left1.get(s, "o") for s in left1["source"].unique()},
              title="All Data: Emissions vs Flow")
    # attach source_label for shapes
    left1 = label_source_simple(left1)

    # we need to replot after labeling shapes (fix: label first)
    axes[0,0].cla()
    plot_left(axes[0,0], add_ad_label(left1), color_palette=COLOR_AD, shape_by="source_label",
              markers_map=markers_left1, title="All Data: Emissions vs Flow")

    right1 = label_source_simple(right1)
    plot_right(axes[0,1], right1, color_palette=COLOR_AVAIL, shape_by="source_label",
               markers_map=markers_right1, title="All Data: Prod-Normalized vs Biogas")

    # ---------- Row 2: Moore et al. ----------
    left2 = filter_moore(measurement_data)
    right2 = filter_moore(measurement_data_ad)

    left2 = label_moore_variant(left2)
    right2 = label_moore_variant(right2)

    markers_left2 = markers_for_sources(left2["source_label"].unique())
    markers_right2 = markers_for_sources(right2["source_label"].unique())

    plot_left(axes[1,0], left2, color_palette=COLOR_AD, shape_by="source_label",
              markers_map=markers_left2, title="Moore et al. 2023 and 2025: Emissions vs Flow")
    plot_right(axes[1,1], right2, color_palette=COLOR_AVAIL, shape_by="source_label",
               markers_map=markers_right2, title="Moore et al. 2023 and 2025: Prod-Normalized vs Biogas")

    # ---------- Row 3: Song et al. ----------
    left3 = filter_song(measurement_data)
    right3 = filter_song(measurement_data_ad)

    left3 = label_source_simple(left3)
    right3 = label_source_simple(right3)

    markers_left3 = markers_for_sources(left3["source_label"].unique())
    markers_right3 = markers_for_sources(right3["source_label"].unique())

    plot_left(axes[2,0], left3, color_palette=COLOR_AD, shape_by="source_label",
              markers_map=markers_left3, title="Song et al.: Emissions vs Flow")
    plot_right(axes[2,1], right3, color_palette=COLOR_AVAIL, shape_by="source_label",
               markers_map=markers_right3, title="Song et al.: Prod-Normalized vs Biogas")

    # ---------- Row 4: Fredenslund & Gålfalk ----------
    left4 = filter_fred_galfalk(measurement_data)
    right4 = filter_fred_galfalk(measurement_data_ad)

    left4 = label_fred_galfalk(left4)
    right4 = label_fred_galfalk(right4)

    markers_left4 = markers_for_sources(left4["source_label"].unique())
    markers_right4 = markers_for_sources(right4["source_label"].unique())

    # Left plot likely empty (no flow data) — function handles gracefully
    plot_left(axes[3,0], left4, color_palette=COLOR_AD, shape_by="source_label",
              markers_map=markers_left4, title="Fredenslund et al. and Gålfalk et al.: No reported flow data")
    plot_right(axes[3,1], right4, color_palette=COLOR_AVAIL, shape_by="source_label",
               markers_map=markers_right4, title="Fredenslund et al. and Gålfalk et al.: Prod-Normalized vs Biogas")
    

    # Remove legends from bottom-right subplot
    for leg in axes[3,0].get_children():
        if isinstance(leg, Legend):
            leg.remove()

    # --- Ensure all subplots share the same axes limits ---
    # Left (flow vs CH4)
    xlims_left = [np.inf, -np.inf]
    ylims_left = [np.inf, -np.inf]
    for ax in axes[:, 0]:
        if ax.get_visible():
            x1, x2 = ax.get_xlim()
            y1, y2 = ax.get_ylim()
            xlims_left[0] = min(xlims_left[0], x1)
            xlims_left[1] = max(xlims_left[1], x2)
            ylims_left[0] = min(ylims_left[0], y1)
            ylims_left[1] = max(ylims_left[1], y2)
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_xlim(xlims_left)
            ax.set_ylim(ylims_left)

    # Right (biogas vs normalized CH4)
    xlims_right = [np.inf, -np.inf]
    ylims_right = [np.inf, -np.inf]
    for ax in axes[:, 1]:
        if ax.get_visible():
            x1, x2 = ax.get_xlim()
            y1, y2 = ax.get_ylim()
            xlims_right[0] = min(xlims_right[0], x1)
            xlims_right[1] = max(xlims_right[1], x2)
            ylims_right[0] = min(ylims_right[0], y1)
            ylims_right[1] = max(ylims_right[1], y2)
    for ax in axes[:, 1]:
        if ax.get_visible():
            ax.set_xlim(xlims_right)
            ax.set_ylim(ylims_right)


    plt.tight_layout()
    # Save + show
    outpath = pathlib.Path("03_figures") / "supplementary_measurement_data_stacked.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")

    # Show in GUI
    plt.show()

    print(f"Saved figure to: {outpath}")

if __name__ == "__main__":
    main()
