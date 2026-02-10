#%%

#Setup
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from a_my_utilities import get_chini_se_slope, set_chini_dataset, load_ch4_emissions_data, calc_biogas_production_rate, load_ch4_emissions_with_ad_only, calculate_production_normalized_ch4, calc_annual_revenue
from a_my_utilities import solve_leak_rate_for_value, get_chini_slope, METHANE_MJ_PER_KG, get_chini_r2_centered, get_chini_r2_origin, annualized_cost, ENGINES, METHANE_KG_PER_SCF
import matplotlib.ticker as mtick
import matplotlib.ticker as ticker

# Load Chini dataset and cache linear regression 
chini_data = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))  # or csv

set_chini_dataset(
    chini_data,
    x_col="flow_m3_per_day",
    y_col="methane_gen_kgh",
    drop_negative=True
)

import numpy as np

def compute_linear_regression(
    data,
    x_col="flow_m3_per_day",
    y_col="methane_gen_kgh",
    *,
    drop_negative=False,
):
    """
    Compute slope, intercept, Excel-style R², and standard errors for y ~ b0 + b1*x.

    Returns:
        dict with keys:
            slope (float): least-squares slope
            intercept (float): least-squares intercept
            r2 (float): 1 - ss_res/ss_tot (centered)
            n (int): number of points used
            se_slope (float): standard error of slope
            se_intercept (float): standard error of intercept
    """
    x, y = _clean_xy(data, x_col, y_col, drop_negative=drop_negative)

    n = len(x)
    if n < 2:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r2": float("nan"),
            "n": int(n),
            "se_slope": float("nan"),
            "se_intercept": float("nan"),
        }

    x_mean = float(x.mean())
    y_mean = float(y.mean())

    sxx = float(np.sum((x - x_mean) ** 2))
    sxy = float(np.sum((x - x_mean) * (y - y_mean)))

    slope = (sxy / sxx) if sxx > 0.0 else float("nan")
    intercept = y_mean - slope * x_mean if np.isfinite(slope) else float("nan")

    y_pred = intercept + slope * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

    # Standard errors (simple linear regression, homoskedastic)
    # df = n - 2 because we estimated 2 parameters
    df = n - 2
    sigma2 = ss_res / df if df > 0 else float("nan")  # residual variance estimate

    se_slope = (sigma2 / sxx) ** 0.5 if (sxx > 0.0 and np.isfinite(sigma2)) else float("nan")
    se_intercept = (
        (sigma2 * (1.0 / n + (x_mean ** 2) / sxx)) ** 0.5
        if (sxx > 0.0 and np.isfinite(sigma2))
        else float("nan")
    )

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
        "n": int(n),
        "se_slope": float(se_slope),
        "se_intercept": float(se_intercept),
    }

#%%
### LINEAR REGRESSION CALCLATIONS - Fixed y-intercept at origin ####

# What is the slope of the Chini dataset in MJ biogas per m3? 

slope = get_chini_slope()          # kg CH4/h per (m^3 wastewater/day)
slope_kg_per_m3 = slope * 24  # Convert to kg CH4 per m3 wastewater
slope_MJ_per_m3 = slope_kg_per_m3 *  METHANE_MJ_PER_KG # Convert to MJ biogas per m3 wastewater

# print
print(f"Chini slope: {slope:.4f} kg CH4/h per m3 wastewater")
print(f"Chini slope: {slope_kg_per_m3:.4f} kg CH4 per m3 wastewater")
print(f"Chini slope: {slope_MJ_per_m3:.4f} MJ biogas per m3 wastewater")

# What is the R2 of the Chini dataset?
r2_centered = get_chini_r2_centered()
print(f"Chini R2 (centered, typical calculations): {r2_centered:.5f}")

# What is R2 for through-origin fit (uncentered R2)? (excel calculation style): 
r2_origin = get_chini_r2_origin()
print(f"Chini R2 (through-origin, uncentered): {r2_origin:.5f}")

# What is standard error of the mean on the slope?: 
se_slope = get_chini_se_slope()
print(f"Chini standard error of slope: {se_slope:.5f}")

#%%
### LINEAR REGRESSION CALCLATIONS - Y-intercept not fixed ####


