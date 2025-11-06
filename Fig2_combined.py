#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from scipy.stats import linregress
import json
import matplotlib.ticker as mtick
from scipy import stats
import matplotlib.patches as mpatches


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

# For facilities that do not report biogas production, use Chini et al 2023 data to model it 
chini_data = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))
set_chini_dataset(
    chini_data,
    x_col="flow_m3_per_day",
    y_col="methane_gen_kgh",
    drop_negative=True,
)

# Load data with production normalized AD calculations 
measurement_data_ad = calculate_production_normalized_ch4(
    load_data_func=load_ch4_emissions_with_ad_only,
    calc_biogas_func=calc_biogas_production_rate,
)

# Filter to remove any non-positive values 
measurement_data_ad = measurement_data_ad[
    (measurement_data_ad['biogas_production_used_kgCH4_per_hr'] > 0) &
    (measurement_data_ad['production_normalized_CH4_percent'] > 0)
].copy()

# Label biogas availability based on data source
# measurement_data_ad['data_availability'] = (
#     measurement_data_ad['source']
#     .fillna('')
#     .apply(lambda x: 'Biogas data available' if 'Fredenslund et al., 2023' in x else 'Biogas production interpolated from flow')
# )

# Label biogas availability based on reported biogas production
measurement_data_ad['data_availability'] = (
    measurement_data_ad['reported_biogas_production']
    .fillna('')
    .apply(lambda x: 'Biogas data available' if x.lower() == 'yes' else 'Biogas production interpolated from flow')
)


measurement_data_ad.to_csv(pathlib.Path("02_clean_data", "measurement_data_ad.csv"), index=False)

# =========================
# Helpers
# =========================

# Function for polotting a power plaw fit
def _powerlaw_fit(x, y):
    """
    Fit y = a * x^b by linear regression on (log x, log y).
    Returns dict with a, b, r2, stderr, intercept_stderr.
    """
    logx = np.log(x)
    logy = np.log(y)
    res = linregress(logx, logy)  # slope=b, intercept=log(a)
    b = res.slope
    a = float(np.exp(res.intercept))
    r2 = float(res.rvalue**2)
    return {
        "model": "power",
        "a": a,
        "b": float(b),
        "r2_loglog": r2,
        "slope_stderr": float(res.stderr),
        "intercept_stderr": float(res.intercept_stderr)
    }

def _powerlaw_fit_with_intervals(x, y, xfit=None, alpha=0.05, use_smearing=True):
    """
    Fit y = a * x^b by linear regression on (log x, log y) and compute
    confidence and prediction intervals on the original (y) scale.

    Returns:
      {
        "a": a, "b": b, "r2_loglog": R^2,
        "xfit": xfit (sorted),
        "yfit": yfit,
        "ci_lower": ci_lower, "ci_upper": ci_upper,
        "pi_lower": pi_lower, "pi_upper": pi_upper,
        "stderr": s, "tcrit": tcrit, "smearing": smear
      }
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[m]; y = y[m]

    if x.size < 3:
        return None  # need at least 3 to estimate variance and df=n-2

    logx = np.log(x)
    logy = np.log(y)

    res = linregress(logx, logy)  # slope=b, intercept=log(a)
    b = float(res.slope)
    a = float(np.exp(res.intercept))
    r2 = float(res.rvalue**2)

    # Fitted values & residuals in log-space
    logy_hat = res.intercept + res.slope * logx
    resid = logy - logy_hat
    n = logx.size
    # Unbiased residual variance with df = n - 2 for simple linear regression
    s2 = float(np.sum(resid**2) / (n - 2))
    s = float(np.sqrt(s2))

    xbar = float(np.mean(logx))
    Sxx = float(np.sum((logx - xbar)**2))

    # X points where to evaluate the line & intervals
    if xfit is None:
        xfit = np.geomspace(np.min(x), np.max(x), 200)
    else:
        xfit = np.asarray(xfit, dtype=float)
    xfit = np.sort(xfit)  # helps with fill_between on log axes
    logxfit = np.log(xfit)

    # Standard errors in log-space
    se_mean = s * np.sqrt(1.0 / n + (logxfit - xbar) ** 2 / Sxx)
    se_pred = s * np.sqrt(1.0 + 1.0 / n + (logxfit - xbar) ** 2 / Sxx)

    # t critical
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 2))

    # Fitted curve (log-space), then back-transform
    logyfit = res.intercept + res.slope * logxfit

    # Optional Duan smearing to correct E[exp(ε)] bias on back-transform
    smear = float(np.mean(np.exp(resid))) if use_smearing else 1.0

    yfit = np.exp(logyfit) * smear
    ci_lower = yfit * np.exp(-tcrit * se_mean)
    ci_upper = yfit * np.exp(+tcrit * se_mean)
    pi_lower = yfit * np.exp(-tcrit * se_pred)
    pi_upper = yfit * np.exp(+tcrit * se_pred)

    return {
        "a": a,
        "b": b,
        "r2_loglog": r2,
        "xfit": xfit,
        "yfit": yfit,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pi_lower": pi_lower,
        "pi_upper": pi_upper,
        "stderr": s,
        "tcrit": tcrit,
        "smearing": smear,
        "n": int(n),
    }


# For making plots pretty 
# Formatting text 
def _format_sci_tex(num, precision=2):
    """
    LaTeX-friendly scientific notation string: '1.23 × 10^{-4}'
    (for use inside math mode: $...$)
    """
    num = float(num)
    s = f"{num:.{precision}e}"   # e.g., '1.23e-04'
    base, exp = s.split("e")
    exp = int(exp)
    return rf"{base} \times 10^{{{exp}}}"


def save_coeffs_as_py(module_path, coeffs_dict, var_name="EMISSIONS_FLOW_COEFFS"):
    """
    Writes a small Python module with a top-level dict you can import.
    Example import: 
        from coefficients_emissions import EMISSIONS_FLOW_COEFFS
    """
    text = (
        "# Auto-generated coefficients module\n"
        f"{var_name} = {json.dumps(coeffs_dict, indent=2)}\n"
    )
    with open(module_path, "w", encoding="utf-8") as f:
        f.write(text)

def print_trendline_results(title, coeffs_like, percent=False):
    """
    Pretty-print power-law fits for caption copying.
    Accepts either:
      - {"model": "...", "coefficients": {"Has AD": {...}, ...}}
      - {"Has AD": {...}, "No AD": {...}, "All": {...}}
    """
    # Normalize input
    if isinstance(coeffs_like, dict) and "coefficients" in coeffs_like:
        coeffs = coeffs_like["coefficients"]
    else:
        coeffs = coeffs_like

    print(f"\n=== {title} ===")
    # Provide a consistent order if those keys exist
    preferred_order = ["Anaerobic digestion", "No anaerobic digestion", "All",
                       "Biogas data available", "Biogas production interpolated from flow"]
    keys = [k for k in preferred_order if k in coeffs] + [k for k in coeffs if k not in preferred_order]

    for label in keys:
        fit = coeffs.get(label)
        if not fit:
            continue
        # Expecting dict with "a", "b", "r2_loglog"
        a = fit.get("a")
        b = fit.get("b")
        r2 = fit.get("r2_loglog")
        if a is None or b is None or r2 is None:
            continue
        if percent:
            a_for_print = 100.0 * float(a)
            eqn = f"y(%) = {a_for_print:.2e} · x^{float(b):.2f}, R² = {float(r2):.3f}"
        else:
            eqn = f"y = {float(a):.2e} · x^{float(b):.2f}, R² = {float(r2):.3f}"
        print(f"{label}: {eqn}")


# =========================
# Plot 1 (left): Emissions vs Flow (log–log)
# =========================

def plot_emissions_vs_flow_ax(
    ax,
    data,
    group_col,
    group_label_map=None,
    group_label_func=None,
    palette=None,
    linewidth=2,
    # title="Methane Emissions by Anaerobic Digestion (Log-Log)",
    legend_precision=2,
):
    """
    Draw CH4 vs Flow (log–log) with 3 power-law trendlines on the given Axes.
    Returns a dict of fit coeffs for Has AD / No AD / All data.
    """

    # legend storage for confidence bands (avoid duplicates)
    if not hasattr(ax, "_ci_legend_entries"):
        ax._ci_legend_entries = []   # list of (handle, label)


    # Only include values greater than zero (filter out zero values or NaN values) 
    filtered = data[(data['flow_m3_per_day'] > 0) & (data['ch4_kg_per_hr'] > 0)].copy()

    # Apply group labels
    if group_label_func is not None:
        filtered['group'] = filtered[group_col].apply(group_label_func)
    elif group_label_map is not None:
        filtered['group'] = filtered[group_col].map(group_label_map)
    else:
        filtered['group'] = filtered[group_col]
    filtered = filtered.dropna(subset=['group'])

    # Scatter
    sns.scatterplot(
        ax=ax,
        data=filtered,
        x='flow_m3_per_day',
        y='ch4_kg_per_hr',
        hue='group',
        palette=palette,
        edgecolor='k',
        s=80,
        alpha=0.8,
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Flow (m³/day)", fontsize=16)
    ax.set_ylabel("CH₄ Emissions (kg/hr)", fontsize=16)
    
    # Spine settings 
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    # ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax.tick_params(axis="both", which="major", direction="in", length=6, width=1, labelsize=14, pad=8)
    ax.tick_params(axis="both", which="minor", direction="in", length=3)

    coeffs_out = {}

    def _color_for(label, fallback="black"):
        if palette and label in palette:
            return palette[label]
        return fallback

    # def _fit_and_plot(x, y, label, color, lw=linewidth, alpha_band=0.20, draw_pi=False):
    #     # Compute fit and intervals on the same x-range we plot
    #     xv = np.asarray(x, dtype=float)
    #     xfit = np.geomspace(np.nanmin(xv[xv > 0]), np.nanmax(xv), 200)

    #     out = _powerlaw_fit_with_intervals(x, y, xfit=xfit, alpha=0.05, use_smearing=True)
    #     if out is None:
    #         return None

    #     ax.plot(out["xfit"], out["yfit"], linewidth=lw, color=color, label="_nolegend_")
    #     # Confidence band
    #     ax.fill_between(out["xfit"], out["ci_lower"], out["ci_upper"],
    #                     color=color, alpha=alpha_band, linewidth=0)
    #     # Optional prediction band (wider)
    #     if draw_pi:
    #         ax.fill_between(out["xfit"], out["pi_lower"], out["pi_upper"],
    #                         color=color, alpha=alpha_band * 0.5, linewidth=0)

    #     # Return everything so you can export/use later if desired
    #     return {
    #         "model": "power",
    #         "a": out["a"],
    #         "b": out["b"],
    #         "r2_loglog": out["r2_loglog"],
    #         "stderr_log": out["stderr"],
    #         "tcrit": out["tcrit"],
    #         "smearing": out["smearing"]
    #     }

    def _fit_and_plot(x, y, label, color, lw=linewidth, alpha_band=0.20):
        xv = np.asarray(x, dtype=float)
        xfit = np.geomspace(np.nanmin(xv[xv > 0]), np.nanmax(xv), 200)

        out = _powerlaw_fit_with_intervals(x, y, xfit=xfit, alpha=0.05, use_smearing=True)
        if out is None:
            return None

        # line
        ax.plot(out["xfit"], out["yfit"], linewidth=lw, color=color, label="_nolegend_")
        # 95% confidence band (shaded)
        ax.fill_between(out["xfit"], out["ci_lower"], out["ci_upper"],
                        color=color, alpha=alpha_band, linewidth=0, label="_nolegend_")

        # per-group legend patch for CI
        ci_label = f"{label} – 95% CI"
        ci_patch = mpatches.Patch(facecolor=color, edgecolor='none', alpha=alpha_band, label=ci_label)
        ax._ci_legend_entries.append((ci_patch, ci_label))

        return {
            "model": "power",
            "a": out["a"],
            "b": out["b"],
            "r2_loglog": out["r2_loglog"],
            "stderr_log": out["stderr"],
            "tcrit": out["tcrit"],
            "smearing": out["smearing"]
        }



    mask_ad = filtered['group'] == 'Anaerobic digestion'
    if mask_ad.any():
        fit_ad = _fit_and_plot(
            filtered.loc[mask_ad, 'flow_m3_per_day'],
            filtered.loc[mask_ad, 'ch4_kg_per_hr'],
            "Anaerobic digestion",
            _color_for('Anaerobic digestion', '#1f77b4'),
        )
        coeffs_out["Has AD"] = fit_ad

    mask_no = filtered['group'] == 'No anaerobic digestion'
    if mask_no.any():
        fit_no = _fit_and_plot(
            filtered.loc[mask_no, 'flow_m3_per_day'],
            filtered.loc[mask_no, 'ch4_kg_per_hr'],
            "No anaerobic digestion",
            _color_for('No anaerobic digestion', '#ff7f0e'),
        )
        coeffs_out["No anaerobic digestion"] = fit_no

    fit_all = _fit_and_plot(
        filtered['flow_m3_per_day'],
        filtered['ch4_kg_per_hr'],
        "All data",
        "black",
    )
    coeffs_out["All"] = fit_all

 # Clean legend (avoid duplicate hue entries) + add CI patches
    handles, labels = ax.get_legend_handles_labels()
    seen = set(); new_h, new_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen and l != "_nolegend_":
            new_h.append(h); new_l.append(l); seen.add(l)

    # add CI patches (one per group, de-duplicated)
    for patch, lab in getattr(ax, "_ci_legend_entries", []):
        if lab not in seen:
            new_h.append(patch); new_l.append(lab); seen.add(lab)

    ax.legend(new_h, new_l, fontsize=13, frameon=False, handlelength=1.5, handletextpad=0.5)


    return {"model": "power", "coefficients": coeffs_out}


# =========================
# Plot 2 (right): Production-normalized CH4 vs Biogas Production (AD only)
# =========================

def _percent_formatter(y, _):
    pct = y * 100
    if pct >= 1:
        return f"{pct:.0f}%"
    elif pct >= 0.1:
        return f"{pct:.1f}%"
    else:
        return f"{pct:.2f}%"


def _sci_tex(num, precision=2):
    s = f"{num:.{precision}e}"
    base, exp = s.split("e")
    return rf"{base} \times 10^{{{int(exp)}}}"


def plot_prod_norm_vs_biogas_ax(
    ax,
    df,
    palette=None,
    line_width=3.5,
    title="Production Normalized Methane Emissions vs. Biogas Production (AD Only)",
):
    if palette is None:
        palette = {
            'Biogas data available': '#E24A33',
            'Biogas production interpolated from flow': '#226f90',
        }

    # legend storage for confidence bands (avoid duplicates)
    if not hasattr(ax, "_ci_legend_entries"):
        ax._ci_legend_entries = []

    sns.scatterplot(
        ax=ax,
        data=df,
        x='biogas_production_used_kgCH4_per_hr',
        y='production_normalized_CH4_percent',
        hue='data_availability',
        palette=palette,
        edgecolor='k',
        s=80,
        alpha=0.8,
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(_percent_formatter)

    # def _fit_and_plot(sub_df, label, color, lw=line_width, alpha_band=0.20, draw_pi=False):
    #     x = sub_df['biogas_production_used_kgCH4_per_hr'].to_numpy()
    #     y = sub_df['production_normalized_CH4_percent'].to_numpy()  # fraction 0–1
    #     m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    #     x = x[m]; y = y[m]
    #     if x.size < 3:
    #         return None

    #     xfit = np.geomspace(x.min(), x.max(), 200)
    #     out = _powerlaw_fit_with_intervals(x, y, xfit=xfit, alpha=0.05, use_smearing=True)
    #     if out is None:
    #         return None

    #     ax.plot(out["xfit"], out["yfit"], lw=lw, color=color, label="_nolegend_")
    #     ax.fill_between(out["xfit"], out["ci_lower"], out["ci_upper"],
    #                     color=color, alpha=alpha_band, linewidth=0)
    #     if draw_pi:
    #         ax.fill_between(out["xfit"], out["pi_lower"], out["pi_upper"],
    #                         color=color, alpha=alpha_band * 0.5, linewidth=0)

    #     return {
    #         "a": out["a"],
    #         "b": out["b"],
    #         "r2_loglog": out["r2_loglog"],
    #         "n": int(out["n"]),
    #         "stderr_log": out["stderr"],
    #         "tcrit": out["tcrit"],
    #         "smearing": out["smearing"]
    #     }

    def _fit_and_plot(sub_df, label, color, lw=line_width, alpha_band=0.20):
        x = sub_df['biogas_production_used_kgCH4_per_hr'].to_numpy()
        y = sub_df['production_normalized_CH4_percent'].to_numpy()  # fraction 0–1
        m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        x = x[m]; y = y[m]
        if x.size < 3:
            return None

        xfit = np.geomspace(x.min(), x.max(), 200)
        out = _powerlaw_fit_with_intervals(x, y, xfit=xfit, alpha=0.05, use_smearing=True)
        if out is None:
            return None

        ax.plot(out["xfit"], out["yfit"], lw=lw, color=color, label="_nolegend_")
        ax.fill_between(out["xfit"], out["ci_lower"], out["ci_upper"],
                        color=color, alpha=alpha_band, linewidth=0, label="_nolegend_")

        ci_label = f"{label} – 95% CI"
        ci_patch = mpatches.Patch(facecolor=color, edgecolor='none', alpha=alpha_band, label=ci_label)
        ax._ci_legend_entries.append((ci_patch, ci_label))

        return {
            "a": out["a"], "b": out["b"], "r2_loglog": out["r2_loglog"],
            "n": int(out["n"]), "stderr_log": out["stderr"],
            "tcrit": out["tcrit"], "smearing": out["smearing"]
        }



    # Subsets
    df_available = df[df['data_availability'] == 'Biogas data available']
    df_interp = df[df['data_availability'] == 'Biogas production interpolated from flow']

    # Trendlines
    _fit_and_plot(df_available, 'Biogas data available', palette.get('Biogas data available', '#1f77b4'))
    _fit_and_plot(df_interp, 'Biogas production interpolated from flow', palette.get('Biogas production interpolated from flow', '#ff7f0e'))
    _fit_and_plot(df, 'All', 'black')


    results = {}
    results['Biogas data available'] = _fit_and_plot(df_available, 
                                                     'Biogas data available', 
                                                     palette.get('Biogas data available', '#1f77b4'))
    results['Biogas production interpolated from flow'] = _fit_and_plot(df_interp, 'Biogas production interpolated from flow', palette.get('Biogas production interpolated from flow', '#ff7f0e'))
    results['All'] = _fit_and_plot(df, 'All', 'black')


    # Legend 
    # handles, labels = ax.get_legend_handles_labels()
    # seen = set(); new_h, new_l = [], []
    # for h, l in zip(handles, labels):
    #     if l not in seen and l != "_nolegend_":
    #         new_h.append(h); new_l.append(l); seen.add(l)
   
    # ax.legend(new_h, new_l)
    # ax.legend(fontsize=13)
    
    handles, labels = ax.get_legend_handles_labels()
    seen = set(); new_h, new_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen and l != "_nolegend_":
            new_h.append(h); new_l.append(l); seen.add(l)

    # add CI patches (one per group, de-duplicated)
    for patch, lab in getattr(ax, "_ci_legend_entries", []):
        if lab not in seen:
            new_h.append(patch); new_l.append(lab); seen.add(lab)

    ax.legend(new_h, new_l, fontsize=13, frameon=False, handlelength=1.5, handletextpad=0.5)


    # Labels and spine settings 
    ax.set_xlabel("Biogas production rate (kg CH₄/hr)", fontsize=16)
    ax.set_ylabel("Production Normalized CH₄ Emissions (%)", fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    # ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis="both", which="major", direction="in", length=6, width=1, labelsize=14, pad=8)
    ax.tick_params(axis="both", which="minor", direction="in", length=3)
  
    
    return results 


# =========================
# Main: build combined figure with subplots
# =========================

def main():
    save_dir = pathlib.Path("03_figures")
    save_dir.mkdir(parents=True, exist_ok=True)

    # palettes & labels for left plot
    group_label_map = {'yes': 'Anaerobic digestion', 
                       'no': 'No anaerobic digestion'}
    palette_left = {'Anaerobic digestion': '#1f77b4', 
                    'No anaerobic digestion': '#ff7f0e'}

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    # Left subplot (Figure 2a)
    coeffs = plot_emissions_vs_flow_ax(
        ax=axes[0],
        data=measurement_data,
        group_col='has_ad',
        group_label_map=group_label_map,
        palette=palette_left,
        linewidth=4,
        # title="Methane Emissions vs. Flow (Log-Log)",
        legend_precision=2,
    )

    # Export coefficients as in original script
    export_coeffs_py = pathlib.Path("coefficients_emissions.py")
    export_coeffs_json = pathlib.Path("coefficients_emissions.json")
    with open(export_coeffs_json, "w", encoding="utf-8") as f:
        json.dump(coeffs, f, indent=2)
    save_coeffs_as_py(export_coeffs_py, coeffs, var_name="EMISSIONS_FLOW_COEFFS")

    # Right subplot (Figure 2c)
    palette_right = {
        'Biogas data available': '#E24A33',
        'Biogas production interpolated from flow': '#226f90',
    }
    plot_prod_norm_vs_biogas_ax(
        ax=axes[1],
        df=measurement_data_ad,
        palette=palette_right,
        # title="Production Normalized CH₄ vs. Biogas Production (AD Only)",
    )
    

    # Overall layout & save combined
    plt.tight_layout()
    combined_path = save_dir / "Figure_2a_2c_combined.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Print trendline equations
    prod_norm_coeffs = plot_prod_norm_vs_biogas_ax(
        ax=axes[1],
        df=measurement_data_ad,
        palette=palette_right,
    )

    # ===== Print trendline equations (clean caption-ready text) =====
    print_trendline_results("Figure 2a – Emissions vs Flow", coeffs, percent=False)
    print_trendline_results("Figure 2c – Production-normalized CH₄ vs Biogas", prod_norm_coeffs, percent=True)


main()
