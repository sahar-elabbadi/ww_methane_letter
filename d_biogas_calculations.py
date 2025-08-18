"""
Script to: 
1. Load Chini data
2. Remove outliers 
3. Saved cleaned data in 02_clean_data > "chini_cleaned.csv"
4. Plot Chini data with fixed y-intercept at origin

Note: does not depend on utility files for loading Chini regression 
"""

#%%
# Imports
import pathlib
import pandas as pd
from a_my_utilities import BIOGAS_FRACTION_CH4, METHANE_KG_PER_SCF, convert_to_scf, mgd_to_m3_per_day
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load utility data from Chini et al 
# File saved in "01_raw_data", "chini-biogas", "chini_for_coding.csv"

chini_data_path = pathlib.Path("01_raw_data", "chini-biogas", "chini_for_coding.csv")
chini_data = pd.read_csv(chini_data_path)

chini_data['flow_m3_per_day'] = chini_data.apply(lambda row: mgd_to_m3_per_day(row['facility_size_MGD']), axis=1)
chini_data['biogas_gen_scf'] = chini_data.apply(lambda row: convert_to_scf(row['biogas_gen_value'], row['biogas_gen_units']), axis=1)
chini_data['methane_gen_kgh'] = chini_data['biogas_gen_scf'] * METHANE_KG_PER_SCF * BIOGAS_FRACTION_CH4 / 366 / 24 # Convert from biogas scf generated in 2012 to kg CH4/hr

# Remove facilities with outlier data 

remove_list = ['Stickney', # Stickney biogas consumption listed is too low to be total production (maybe rest is flared and not included in data facility provided)
               'Back River WWTP', # Baltimore, Black River facility has implausibly high biogas production for a facility of its size. 
               'EBMUD Main WWTP', # Flow data much lower than expected give our knowledge of this facility
               'TE Maxson WWTP', # Biogas production is very high - facility may be accepting additional high-strenght waste streams
               'Toledo Bay View', # Biogas production is very high - facility may be accepting additional high-strenght waste stream
               ]

# Keep only rows where facility_name is NOT in the list
chini_data = chini_data[~chini_data['facility_name'].isin(remove_list)]

chini_data.to_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"), index=False)

#%%
######## Compute through-origin regression ##########

def compute_through_origin_regression(
    data,
    x_col="flow_m3_per_day",
    y_col="methane_gen_kgh",
    *,
    drop_nonfinite=True,
    drop_negative=False
):
    """
    Compute slope and through-origin R² (Excel-style) for y ~ m * x.

    Cleans rows where x or y is missing (and optionally non-finite / negative).

    Returns:
        dict with keys:
            slope (float): least-squares slope through the origin
            r2_origin (float): 1 - SS_res / sum(y^2)  (Excel when intercept=0)
            n (int): number of points used
            x (np.ndarray), y (np.ndarray): cleaned arrays (for plotting/prediction)
    """
    df = data[[x_col, y_col]].copy()

    # base cleaning: drop NaN
    mask = df.notna().all(axis=1)

    if drop_nonfinite:
        mask &= np.isfinite(df).all(axis=1)

    if drop_negative:
        mask &= (df[x_col] >= 0) & (df[y_col] >= 0)

    dfc = df.loc[mask]
    x = dfc[x_col].to_numpy(dtype=float)
    y = dfc[y_col].to_numpy(dtype=float)

    if len(x) == 0:
        raise ValueError("No valid rows after cleaning.")

    # Through-origin least squares:
    # slope = sum(x*y) / sum(x^2)
    denom = np.sum(x**2)
    if denom == 0:
        raise ValueError("sum(x^2) is zero; cannot fit a through-origin line.")

    slope = np.sum(x * y) / denom
    y_pred = slope * x

    # Excel-style through-origin R²
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot_origin = np.sum(y ** 2)
    r2_origin = 1 - ss_res / ss_tot_origin if ss_tot_origin > 0 else np.nan

    return {
        "slope": float(slope),
        "r2_origin": float(r2_origin),
        "n": int(len(x)),
        "x": x,
        "y": y,
    }


########## MAKE PLOT ##########

def plot_through_origin_regression(
    stats,
    *,
    color="#1f77b4",
    title="Methane Generation vs. Flow (Through-Origin Regression)",
    xlabel="Flow (m³/day)",
    ylabel="Methane Generation (kg/h)",
    save_dir=pathlib.Path("03_figures"),
    filename="methane_vs_flow_through_origin.png"
):
    """
    Plot cleaned data and the through-origin best-fit line using precomputed stats.
    """
    x = stats["x"]
    y = stats["y"]
    slope = stats["slope"]
    r2 = stats["r2_origin"]

    # Build line from the origin to max(x)
    x_line = np.linspace(0, np.max(x), 200)
    y_line = slope * x_line

    # Styling to match your other figures
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))

    # Scatter with black edge, export-friendly
    plt.scatter(x, y, s=80, edgecolor="k", alpha=0.7, color=color, label="Data")

    # Regression line
    plt.plot(x_line, y_line, linewidth=2, color=color,
             label=f"y = {slope:.2f} x\nR² (origin) = {r2:.3f}")

    # Axes from origin to match the physical assumption
    plt.xlim(0, None)
    plt.ylim(0, None)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    outpath = save_dir / filename
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()

    return outpath


# Compute linear regression stats
stats = compute_through_origin_regression(
    chini_data,
    x_col="flow_m3_per_day",
    y_col="methane_gen_kgh",
    drop_negative=True   # keep if you want to enforce nonnegative physics
)

# Plot figure 
plot_through_origin_regression(
    stats,
    color="#1f77b4",
    title="Methane Generation vs. Flow (Through-Origin Regression)"
)
# %%
