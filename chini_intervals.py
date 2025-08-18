#%% 
import numpy as np
import pandas as pd
from functools import lru_cache
from scipy import stats
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
from a_my_utilities import set_chini_dataset, calc_biogas_production_rate

chini_data = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))  

set_chini_dataset(
    chini_data,
    x_col="flow_m3_per_day",
    y_col="methane_gen_kgh",
    drop_negative=True
)


def chini_confidence_intervals(flow_m3_per_day, alpha=0.05):
    """
    Calculate predicted biogas production and 95% confidence intervals
    from the Chini dataset regression.

    Parameters
    ----------
    flow_m3_per_day : array-like
        Input flow rates (m^3/day).
    alpha : float, optional
        Significance level (default=0.05 → 95% confidence interval).

    Returns
    -------
    dict with keys:
        - "estimate": predicted mean value (kg CH4/h)
        - "lower": lower CI bound
        - "upper": upper CI bound
    """
    # Get dataset and slope
    df = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))  
    if df is None:
        raise RuntimeError("Chini dataset not set. Call set_chini_dataset(df, ...) first.")
    x = df['flow_m3_per_day']
    y = df["methane_gen_kgh"]

    n = len(x)

    # Estimate slope
    slope = float(np.sum(x * y) / np.sum(x**2))

    # Residuals and variance
    y_hat = slope * x
    residuals = y - y_hat
    sigma2 = np.sum(residuals**2) / (n - 1)

    # Variance of slope
    var_slope = sigma2 / np.sum(x**2)

    # t critical value
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)

    # Predictions
    flow = np.asarray(flow_m3_per_day, dtype=float)
    est = slope * flow
    
    # Standard error of the mean 
    se_mean = np.sqrt(var_slope * flow**2)

    # Standard error for predictions
    se_pred = np.sqrt(var_slope * flow**2)

    # Confidence intervals for mean 
    lower = est - t_crit * se_mean
    upper = est + t_crit * se_mean

    # Confidence intervals for predictions
    pi_lower = est - t_crit * se_pred
    pi_upper = est + t_crit * se_pred

    return {
        "estimate": est,
        "lower_ci": lower,
        "upper_ci": upper,
        "lower_pi": pi_lower,
        "upper_pi": pi_upper
    }

def plot_chini_regression_with_intervals(alpha=0.05, n_points=200):
    # Get dataset
    # Get dataset and slope
    df = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))  
    if df is None:
        raise RuntimeError("Chini dataset not set. Call set_chini_dataset(df, ...) first.")
    x = df['flow_m3_per_day']
    y = df["methane_gen_kgh"]

    n = len(x)

    # Fit slope through origin
    Sxx = np.sum(x**2)
    Sxy = np.sum(x*y)
    slope = Sxy / Sxx
    y_hat = slope * x

    # Residuals & variance
    residuals = y - y_hat
    sigma2 = np.sum(residuals**2) / (n - 1)
    var_slope = sigma2 / Sxx
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)

    # Range of x values for plotting
    x_grid = np.linspace(0, x.max()*1.05, n_points)
    est = slope * x_grid

    # SE of mean response
    se_mean = np.sqrt(var_slope * x_grid**2)
    ci_lower = np.maximum(0, est - t_crit * se_mean)
    ci_upper = est + t_crit * se_mean

    # SE of prediction (mean uncertainty + residual scatter)
    se_pred = np.sqrt(se_mean**2 + sigma2)
    pi_lower = np.maximum(0, est - t_crit * se_pred)
    pi_upper = est + t_crit * se_pred

    eq_text = f"$\\hat{{y}} = {slope:.5f}\\,x$"

    # --- Plot ---
    plt.figure(figsize=(8,6))
    plt.scatter(x, y, color="black", alpha=0.7, label="Observed data")
    plt.plot(x_grid, est, "b-", label=f"Regression: {eq_text}", linewidth=2)
    
    # Confidence interval (narrow band)
    plt.fill_between(x_grid, ci_lower, ci_upper, color="blue", alpha=0.3,
                     label="Confidence interval (95%) of linear fit")

    # Prediction interval (wide band)
    plt.fill_between(x_grid, pi_lower, pi_upper, color="orange", alpha=0.2,
                     label="Prediction interval (95%)")
    
        # --- regression equation text ---
    # plt.text(0.05 * np.max(x_grid), 0.6 * np.max(y), eq_text,
    #          fontsize=12, color="red")

    plt.xlabel("Flow (m³/day)")
    plt.ylabel("Biogas production (kg CH₄/h)")
    plt.title("Chini regression with confidence & prediction intervals")
    plt.legend()
    plt.show()


plot_chini_regression_with_intervals(alpha=0.05, n_points=200)
# %%
