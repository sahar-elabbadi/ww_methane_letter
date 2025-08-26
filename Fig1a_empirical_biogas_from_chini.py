#%% 
import numpy as np
import pandas as pd
from functools import lru_cache
from scipy import stats
import pathlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from a_my_utilities import set_chini_dataset, get_chini_stats, chini_confidence_intervals, get_chini_xy

from a_my_utilities import get_chini_xy


# --- 1) Load Chini dataset ---
chini_data = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))
set_chini_dataset(
    chini_data,
    x_col="flow_m3_per_day",
    y_col="methane_gen_kgh",
    drop_negative=True,
)

# --- 2) Function for plotting  ---
def plot_chini_regression_with_intervals(alpha=0.05, n_points=200, save_path=None):
    
    # Chini stats for annotation 
    stats_cached = get_chini_stats()
    slope = stats_cached["slope"]
    r2_origin = stats_cached["r2_origin"]

    # Load chini data 
    x, y = get_chini_xy()
    n = len(x)

    # x-grid for the curve and bands
    x_grid = np.linspace(0.0, x.max() * 1.05, n_points)

    lower_floor = 1e-6  # use 1e-6 for plotting purposes

    ci = chini_confidence_intervals(x_grid, alpha=alpha)
    ci_lower = np.maximum(lower_floor, ci["lower_ci"])
    pi_lower = np.maximum(lower_floor, ci["lower_pi"])

    eq_text = (
        f"$\\hat{{y}} = {slope:.5f}\\,x$  "
        f"[R$^2$ = {r2_origin:.3f}, n={n}]"
    )

    # --- plotting ---
    fig, ax = plt.subplots(figsize=(9, 6))

    # mean-response CI band
    ax.fill_between(
        x_grid, ci_lower, ci["upper_ci"], alpha=0.30,
        label=f"{int((1-alpha)*100)}% confidence interval (mean)", 
        color="#60C1CF", 
    )

    # prediction band (mean + residual scatter)
    ax.fill_between(
        x_grid, pi_lower, ci["upper_pi"], alpha=0.20,
        label=f"{int((1-alpha)*100)}% prediction interval", 
        color="#79BF82"
    )

    # Equation of best fit 
    ax.plot(x_grid, ci["estimate"], linewidth=4, label=f"{eq_text}")

    # Chini data points - fill (semi-transparent)
    ax.scatter(
        x, y,
        s=50,
        facecolors="black",
        edgecolors="none",
        alpha=0.4
    )

    # Chini data points - outline (solid, on top)
    ax.scatter(
        x, y,
        s=50,
        facecolors="none",
        edgecolors="black",
        linewidths=1.2
    )


    # labels and title
    ax.set_xlabel("Flow (m³/day)", fontsize=16)
    ax.set_ylabel("Biogas production (kg CH₄/h)", fontsize=16)
    # ax.set_title("Chini regression with confidence & prediction intervals", fontsize=16)

    # format tick labels
    ax.tick_params(axis="both", labelsize=15, pad=8)  # increase tick font size
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:,.0f}"))

    # Tick marks orientation and length 
    ax.tick_params(axis="both", which="both", direction="in", length=6, width=1)

    # Axis length of formatting 
    # ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, x_grid[-1])

    # Remove top and right axes: 
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # increase thickness of bottom & left spines
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)


    ax.legend(fontsize=13, frameon=False)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# Make plot
plot_chini_regression_with_intervals(alpha=0.05, n_points=200)
#%%


# chini_data = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))  

# # -- Load and register the dataset ONCE --
# chini_data = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))

# # Set x and y columns, drop negative values (there shouldn't be any)
# X_COL = "flow_m3_per_day"
# Y_COL = "methane_gen_kgh"
# DROP_NEGATIVE = True

# set_chini_dataset(
#     chini_data,
#     x_col=X_COL,
#     y_col=Y_COL,
#     drop_negative=DROP_NEGATIVE
# )


# def chini_confidence_intervals(flow_m3_per_day, alpha=0.05):
#     """
#     Calculate predicted biogas production and 95% confidence intervals
#     from the Chini dataset regression.

#     Parameters
#     ----------
#     flow_m3_per_day : array-like
#         Input flow rates (m^3/day).
#     alpha : float, optional
#         Significance level (default=0.05 → 95% confidence interval).

#     Returns
#     -------
#     dict with keys:
#         - "estimate": predicted mean value (kg CH4/h)
#         - "lower": lower CI bound
#         - "upper": upper CI bound
#     """
#     # Get dataset and slope
#     df = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))  
#     if df is None:
#         raise RuntimeError("Chini dataset not set. Call set_chini_dataset(df, ...) first.")
#     x = df['flow_m3_per_day']
#     y = df["methane_gen_kgh"]

#     n = len(x)

#     # Estimate slope
#     slope = float(np.sum(x * y) / np.sum(x**2))

#     # Residuals and variance
#     y_hat = slope * x
#     residuals = y - y_hat
#     sigma2 = np.sum(residuals**2) / (n - 1)

#     # Variance of slope
#     var_slope = sigma2 / np.sum(x**2)

#     # t critical value
#     t_crit = stats.t.ppf(1 - alpha/2, df=n-1)

#     # Predictions
#     flow = np.asarray(flow_m3_per_day, dtype=float)
#     est = slope * flow
    
#     # Standard error of the mean 
#     se_mean = np.sqrt(var_slope * flow**2)

#     # Standard error for predictions
#     se_pred = np.sqrt(var_slope * flow**2)

#     # Confidence intervals for mean 
#     lower = est - t_crit * se_mean
#     upper = est + t_crit * se_mean

#     # Confidence intervals for predictions
#     pi_lower = est - t_crit * se_pred
#     pi_upper = est + t_crit * se_pred

#     return {
#         "estimate": est,
#         "lower_ci": lower,
#         "upper_ci": upper,
#         "lower_pi": pi_lower,
#         "upper_pi": pi_upper
#     }

# def plot_chini_regression_with_intervals(alpha=0.05, n_points=200):
#     # Get dataset
#     # Get dataset and slope
#     df = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))  
#     if df is None:
#         raise RuntimeError("Chini dataset not set. Call set_chini_dataset(df, ...) first.")
#     x = df['flow_m3_per_day']
#     y = df["methane_gen_kgh"]

#     n = len(x)

#     # Fit slope through origin
#     Sxx = np.sum(x**2)
#     Sxy = np.sum(x*y)
#     slope = Sxy / Sxx
#     y_hat = slope * x

#     # Residuals & variance
#     residuals = y - y_hat
#     sigma2 = np.sum(residuals**2) / (n - 1)
#     var_slope = sigma2 / Sxx
#     t_crit = stats.t.ppf(1 - alpha/2, df=n-1)

#     # Range of x values for plotting
#     x_grid = np.linspace(0, x.max()*1.05, n_points)
#     est = slope * x_grid

#     # SE of mean response
#     se_mean = np.sqrt(var_slope * x_grid**2)
#     ci_lower = np.maximum(0, est - t_crit * se_mean)
#     ci_upper = est + t_crit * se_mean

#     # SE of prediction (mean uncertainty + residual scatter)
#     se_pred = np.sqrt(se_mean**2 + sigma2)
#     pi_lower = np.maximum(0, est - t_crit * se_pred)
#     pi_upper = est + t_crit * se_pred

#     eq_text = f"$\\hat{{y}} = {slope:.5f}\\,x$"

#     # --- Plot ---
#     plt.figure(figsize=(8,6))
#     plt.scatter(x, y, color="black", alpha=0.7, label="Observed data")
#     plt.plot(x_grid, est, "b-", label=f"Regression: {eq_text}", linewidth=2)
    
#     # Confidence interval (narrow band)
#     plt.fill_between(x_grid, ci_lower, ci_upper, color="blue", alpha=0.3,
#                      label="Confidence interval (95%) of linear fit")

#     # Prediction interval (wide band)
#     plt.fill_between(x_grid, pi_lower, pi_upper, color="orange", alpha=0.2,
#                      label="Prediction interval (95%)")
    
#         # --- regression equation text ---
#     # plt.text(0.05 * np.max(x_grid), 0.6 * np.max(y), eq_text,
#     #          fontsize=12, color="red")

#     plt.xlabel("Flow (m³/day)")
#     plt.ylabel("Biogas production (kg CH₄/h)")
#     plt.title("Chini regression with confidence & prediction intervals")
#     plt.legend()
#     plt.show()


# plot_chini_regression_with_intervals(alpha=0.05, n_points=200)
# # %%
