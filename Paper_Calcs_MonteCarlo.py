#%% 
import numpy as np
import pandas as pd
import pathlib
from a_my_utilities import calc_annual_revenue, Engine, ENGINES, set_chini_dataset
import matplotlib.pyplot as plt


###### Set up data ######

# Import EIA data dictionaries 
from a_my_utilities import eia_industrial_tariffs_2023 as elec_price_dict
from a_my_utilities import eia_industrial_natural_gas_2023 as ng_price_dict

# Load Chini dataset
chini_data = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))  # or csv

set_chini_dataset(
    chini_data,
    x_col="flow_m3_per_day",
    y_col="methane_gen_kgh",
    drop_negative=True
)

# Load measurement dataset with biogas normalized production rate 
measurement_data_ad_all = pd.read_csv(
    pathlib.Path("02_clean_data", "measurement_data_ad.csv")
)

measurement_data_ad_reported_biogas = pd.read_csv(
    pathlib.Path("02_clean_data", "measurement_data_ad.csv")
).query("reported_biogas_production == 'yes'")

# Distrubtions for leak rates 

# For bootstrapping data from existing measurement datasets
def make_bootstrap_sampler_from_df(df: pd.DataFrame, col: str = "leak_rate"):
    """
    Returns a sampler function f(n) that bootstraps n leak-rate values from df[col].
    - Drops NaNs and keeps only values strictly between 0 and 1.
    - Raises a clear error if no valid data remain.
    """
    series = pd.to_numeric(df[col], errors="coerce")
    data = series[(series > 0) & (series < 1)].dropna().to_numpy()

    if data.size == 0:
        raise ValueError(f"No valid leak-rate data found in column '{col}' (0<value<1).")

    # Optional: you can print/log the sample size to be aware of data support
    # print(f"Bootstrap base size for '{col}': {data.size}")

    def sampler(n: int):
        return np.random.choice(data, size=n, replace=True)

    return sampler

# For making a heavy tail distribution about a median value 
def make_heavy_tail_leak_dist(median=0.05, sigma=0.8):
    """
    Returns a sampler for a heavy-tailed leak-rate distribution.
    Distribution is lognormal with given median and spread.

    Parameters
    ----------
    median : float
        Desired median leak rate (fraction, e.g. 0.05 = 5%).
    sigma : float
        Spread parameter of the lognormal. Larger = heavier tail.

    Returns
    -------
    sampler : function
        sampler(n) gives n leak-rate draws.
    """
    mu = np.log(median)  # ensures median is as specified
    def sampler(n):
        return np.random.lognormal(mean=mu, sigma=sigma, size=n)
    return sampler

# Monte Carlo function applied to a single plant 
def monte_carlo_annual_revenue(
    *,
    plant_size,
    state_abbr,
    leak_rate_dist,
    leak_fraction_capturable_dist,
    engine: Engine,
    n_iter=10000,
    ogi_cost=100000,
    random_seed=None
):
    """
    Run a Monte Carlo simulation to estimate annual revenue from capturing methane leaks.

    Parameters
    ----------
    plant_size : float
        Size of the plant in m3/day (fixed).
    n_iter : int
        Number of Monte Carlo iterations.
    leak_rate_dist : callable
        Function that returns samples of leak_rate (fraction).
    leak_fraction_capturable_dist : callable
        Function that returns samples of leak_fraction_capturable (fraction).
    electricity_price_dist : callable
        Function that returns samples of electricity price (USD/kWh).
    nat_gas_price_dist : callable
        Function that returns samples of natural gas price (USD/MJ).
    engine : Engine
        Engine object with attributes efficiency and power_to_heat_ratio.
    ogi_cost : float
        Cost of OGI survey (USD).
    random_seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of annual revenue values (USD).
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    # --- State means ---
    elec_mean = elec_price_dict[state_abbr]  # $/kWh
    ng_mean   = ng_price_dict[state_abbr]    # $/MJ

    # --- Normal distributions around state means ---
    elec_sigma = 0.1 * elec_mean / 1.96
    ng_sigma   = 0.1 * ng_mean / 1.96

    elec_price_dist = lambda n: np.random.normal(elec_mean, elec_sigma, n)
    ng_price_dist   = lambda n: np.random.normal(ng_mean, ng_sigma, n)

    # Draw random samples
    leak_rates = leak_rate_dist(n_iter)
    leak_fracs = leak_fraction_capturable_dist(n_iter)
    elec_prices = elec_price_dist(n_iter)
    gas_prices = ng_price_dist(n_iter)

    # Run the simulation
    revenues = np.array([
        calc_annual_revenue(
            plant_size,
            leak_rates[i],
            leak_fracs[i],
            engine=engine,
            electricity_price_per_kWh=elec_prices[i],
            nat_gas_price_per_MJ=gas_prices[i],
            ogi_cost=ogi_cost
        )
        for i in range(n_iter)
    ])

    return revenues, {"elec_mu_usd_per_kwh": elec_mean, "ng_mu_usd_per_mj": ng_mean}

# Function for running Monte Carlo simulation on all plants in a DataFrame
def run_mc_for_all_plants(
    df: pd.DataFrame,
    *,
    leak_rate_samplers: dict,  # {"bootstrap_all": f, "bootstrap_biogas": g, "heavy_tail": h}
    leak_fraction_capturable_dist,  # e.g., lambda n: np.random.uniform(0.3, 0.8, n)
    engine: Engine,
    n_iter: int = 10000,
    base_seed: int = 123,       # determinism across runs
    ogi_cost: float = 100000,
    save_csv_path: str | None = None
):
    """
    Reads clean_data/chp_data.csv with columns:
      - state (2-letter abbr)
      - flow_m3_per_day (float)
    Runs MC per plant for each leak-rate sampler, returns a summary DataFrame.
    """

    # Prepare results table 
    out_rows = []

    for i, row in df.iterrows():
        plant_state = row["state"]
        flow = float(row["flow_m3_per_day"])

        row_result = {
            "plant_index": i,
            "state": plant_state,
            "flow_m3_per_day": flow,
        }

        # For determinism per plant & per sampler
        for name, sampler in leak_rate_samplers.items():
            seed = None if base_seed is None else (base_seed + 1000*i + hash(name) % 9973)

            revenues, price_means = monte_carlo_annual_revenue(
                plant_size=flow,
                state_abbr=plant_state,
                leak_rate_dist=sampler,
                leak_fraction_capturable_dist=leak_fraction_capturable_dist,
                engine=engine,
                n_iter=n_iter,
                ogi_cost=ogi_cost,
                random_seed=seed
            )

            # Calculate 95% CI 
            z = 1.96  # for normal approximation
            ci_half_width = z * (np.std(revenues) / np.sqrt(n_iter))
            ci_lower = np.mean(revenues) - ci_half_width
            ci_upper = np.mean(revenues) + ci_half_width

            # Summaries
            row_result.update({
                f"median_{name}": float(np.median(revenues)),
                f"p2_5_{name}":  float(np.percentile(revenues, 2.5)),
                f"p97_5_{name}": float(np.percentile(revenues, 97.5)),
                f"mean_{name}":  float(np.mean(revenues)),
                f"ci_lower_{name}": float(ci_lower), # adding this to see what it looks like
                f"ci_upper_{name}": float(ci_upper), # adding this to see what it looks like
                f"elec_mu_usd_per_kwh_{name}": price_means["elec_mu_usd_per_kwh"],
                f"ng_mu_usd_per_mj_{name}":    price_means["ng_mu_usd_per_mj"],
            })

        out_rows.append(row_result)

    summary = pd.DataFrame(out_rows)

    if save_csv_path:
        summary.to_csv(save_csv_path, index=False)

    return summary

def summarize_national_from_mc(summary_df, leak_rate_samplers, n_iter=10000):
    """
    Summarize national-level results by summing across all plants
    for each leak-rate assumption.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Plant-level Monte Carlo results (output from run_mc_for_all_plants).
        Must include plant_index so we can line up runs if needed.
    leak_rate_samplers : dict
        Dict of {"name": sampler_function} so we know which leak-rate models to summarize.
    n_iter : int
        Number of iterations used in MC (same as used in plant-level runs).
    
    Returns
    -------
    pd.DataFrame
        National-level summary with median, 95% CI, mean for each leak-rate model.
    """
    national_summaries = []
    
    for name in leak_rate_samplers.keys():
        # Extract plant-level summaries
        med = summary_df[f"median_{name}"].sum()
        mean = summary_df[f"mean_{name}"].sum()
        lo = summary_df[f"ci_lower_{name}"].sum() # previously: p2_5_{name}
        hi = summary_df[f"ci_upper_{name}"].sum() #previously: p97_5_{name}
        p2_5 = summary_df[f"p2_5_{name}"].sum()
        p97_5 = summary_df[f"p97_5_{name}"].sum()

        national_summaries.append({
            "distribution": name,
            "median_sum": med,
            "mean_sum": mean,
            "ci_lower_sum": lo, # previously: p2_5_sum
            "ci_upper_sum": hi, # previously: p97_5_sum
            "p2_5_sum": p2_5,
            "p97_5_sum": p97_5, 
        })
    
    return pd.DataFrame(national_summaries)


## SETUP FOR MONTE CARLO SIMULATION ## 

# Load facility-level data 
chp_facilities_data = pd.read_csv(pathlib.Path("02_clean_data", "chp_data.csv"))

# Build sampler functions for leak rate distrubtion
leak_rate_all_dist = make_bootstrap_sampler_from_df(measurement_data_ad_all, col="production_normalized_CH4_percent")
leak_rate_biogas_dist = make_bootstrap_sampler_from_df(measurement_data_ad_reported_biogas, col="production_normalized_CH4_percent")
leak_rate_heavy_tail_dist = make_heavy_tail_leak_dist(median=0.05, sigma=0.8)

leak_rate_samplers = {
    "bootstrap_all": leak_rate_all_dist,
    "bootstrap_biogas": leak_rate_biogas_dist,
    "heavy_tail": leak_rate_heavy_tail_dist,
}

# Additional inputs 
leak_fraction_capturable_dist = lambda n: np.random.uniform(0.5, 0.9, n)  # Fraction of gas capturable: 50–90%
engine = ENGINES['reciprocating_lean_burn'] # SET ENGINE TYPE HERE 

# Run simulation on all data 

summary_df = run_mc_for_all_plants(
    df=chp_facilities_data,
    leak_rate_samplers=leak_rate_samplers,
    leak_fraction_capturable_dist=leak_fraction_capturable_dist,
    engine=engine,
    n_iter=10000,
    base_seed=123,
    save_csv_path="02_clean_data/chp_mc_summary.csv"  # or None
)


national_summary = summarize_national_from_mc(
    summary_df, 
    leak_rate_samplers=leak_rate_samplers, 
    n_iter=10000
)

national_summary.to_csv("02_clean_data/chp_national_mc_summary.csv", index=False)

# Print for text 

def print_national_summary_table(national_summary_df):
    """
    Print a nicely formatted table of national MC results.

    Parameters
    ----------
    national_summary_df : pd.DataFrame
        Must contain columns: distribution, median, mean, p2_5, p97_5
        (output from run_mc_for_all_plants_national).
    """
    print("\nNational Monte Carlo Summary (USD per year)")
    print("-" * 70)
    print(f"{'Distribution':<20} {'Median [95% range]':<30} {'Mean [95% CI]':<30}") # We are displaying 2.5th to 97.5th percentiles here
    print("-" * 70)

    for _, row in national_summary_df.iterrows():
        dist = row["distribution"]
        med = row["median_sum"]
        mean = row["mean_sum"]
        p2_5 = row["p2_5_sum"]
        p97_5 = row["p97_5_sum"]
        lo = row["ci_lower_sum"] # previously: p2_5_sum
        hi = row["ci_upper_sum"] # previously: p97_5_sum

        med_str = f"{med:,.0f} [{p2_5:,.0f} – {p97_5:,.0f}]"
        mean_str = f"{mean:,.0f} [{lo:,.0f} – {hi:,.0f}]"

        print(f"{dist:<20} {med_str:<30} {mean_str:<30}")
    print("-" * 70)


# Print nicely formatted summary
print_national_summary_table(national_summary.reset_index(drop=True))
