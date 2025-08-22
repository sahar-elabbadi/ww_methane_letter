#%% 
#Setup
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from a_my_utilities import set_chini_dataset, load_ch4_emissions_data, calc_biogas_production_rate, load_ch4_emissions_with_ad_only, calculate_production_normalized_ch4, calc_annual_revenue
from a_my_utilities import solve_leak_rate_for_value, get_chini_slope, METHANE_MJ_PER_KG, get_chini_r2

import matplotlib.ticker as mtick

###### LOAD DATA ######

# Load Chini dataset and cache linear regression 
chini_data = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))  # or csv

set_chini_dataset(
    chini_data,
    x_col="flow_m3_per_day",
    y_col="methane_gen_kgh",
    drop_negative=True
)

# Load measurement data
measurement_data_ad = calculate_production_normalized_ch4(
    load_data_func=load_ch4_emissions_with_ad_only,
    calc_biogas_func=calc_biogas_production_rate
)


# All measurement data 
measurement_data = load_ch4_emissions_data()

# Filter measurement data to ensure values are > 0 and not NaN
measurement_data_ad = measurement_data_ad[
    (measurement_data_ad['biogas_production_used_kgCH4_per_hr'] > 0) &
    (measurement_data_ad['production_normalized_CH4_percent']> 0)]

# Load El Abbadi, Feng et al 2025 data on facilities with CHP 
chp_data = pd.read_csv(pathlib.Path("02_clean_data", "chp_data.csv"))
wwtp_data = pd.read_csv(pathlib.Path("02_clean_data", "wwtp_data.csv"))


#%% 
######### Section: Comparison of measurement-based emissions factors from WRRFs #######

# How many facilities do not report biogas production?
no_witout_biogas = measurement_data_ad[measurement_data_ad['reported_biogas_production'] == 'no'].shape[0]
print(f"Number of facilities that do not report biogas production: {no_witout_biogas}")
# Percent of facilities that do not report biogas production:
print(f"Percent of facilities that do not report biogas production: {no_witout_biogas / measurement_data_ad.shape[0] * 100:.2f}%")

# What is the slope of the Chini dataset in MJ biogas per m3? 

slope = get_chini_slope()          # kg CH4/h per (m^3 wastewater/day)
slope_kg_per_m3 = slope * 24  # Convert to kg CH4 per m3 wastewater
slope_MJ_per_m3 = slope_kg_per_m3 *  METHANE_MJ_PER_KG # Convert to MJ biogas per m3 wastewater

# print
print(f"Chini slope: {slope:.4f} kg CH4/h per m3 wastewater")
print(f"Chini slope: {slope_kg_per_m3:.4f} kg CH4 per m3 wastewater")
print(f"Chini slope: {slope_MJ_per_m3:.4f} MJ biogas per m3 wastewater")

# What is the R2 of the Chini dataset?
r2 = get_chini_r2()
print(f"Chini R2: {r2:.4f}")

#%%
from scipy import stats

########## Discussion of Figure 2a #######

# calculate methane emissions per m3 wastewater treated 
measurement_data['ch4_kg_per_m3'] = measurement_data['ch4_kg_per_hr'] / (measurement_data['flow_m3_per_day'] / 24)

# facilities with AD 
measurement_data_has_ad = measurement_data[measurement_data['has_ad']=='yes']
# facilities without AD
measurement_data_no_ad = measurement_data[measurement_data['has_ad']=='no']

# --- With AD ---
print(f"Mean emissions rate for facilities with AD: {measurement_data_has_ad['ch4_kg_per_hr'].mean():.2f} kg/hr")
print(f"Median emissions rate for facilities with AD: {measurement_data_has_ad['ch4_kg_per_hr'].median():.2f} kg/hr")
print(f"Std Dev emissions rate for facilities with AD: {measurement_data_has_ad['ch4_kg_per_hr'].std():.2f} kg/hr\n")

# --- Without AD ---
print(f"Mean emissions rate for facilities without AD: {measurement_data_no_ad['ch4_kg_per_hr'].mean():.2f} kg/hr")
print(f"Median emissions rate for facilities without AD: {measurement_data_no_ad['ch4_kg_per_hr'].median():.2f} kg/hr")
print(f"Std Dev emissions rate for facilities without AD: {measurement_data_no_ad['ch4_kg_per_hr'].std():.2f} kg/hr\n")

# Add normalized emissions column (kg CH4 per m3 wastewater)

# --- With AD ---
print(f"Mean normalized emissions for facilities with AD: {measurement_data_has_ad['ch4_kg_per_m3'].mean():.4f} kg/m3")
print(f"Median normalized emissions for facilities with AD: {measurement_data_has_ad['ch4_kg_per_m3'].median():.4f} kg/m3")
print(f"Std Dev normalized emissions for facilities with AD: {measurement_data_has_ad['ch4_kg_per_m3'].std():.4f} kg/m3\n")

# --- Without AD ---
print(f"Mean normalized emissions for facilities without AD: {measurement_data_no_ad['ch4_kg_per_m3'].mean():.4f} kg/m3")
print(f"Median normalized emissions for facilities without AD: {measurement_data_no_ad['ch4_kg_per_m3'].median():.4f} kg/m3")
print(f"Std Dev normalized emissions for facilities without AD: {measurement_data_no_ad['ch4_kg_per_m3'].std():.4f} kg/m3")

# t-test for normalized emissions (kg/m3)
t_stat_norm, p_val_norm = stats.ttest_ind(
    measurement_data_has_ad['ch4_kg_per_m3'].dropna(),
    measurement_data_no_ad['ch4_kg_per_m3'].dropna(),
    equal_var=False #Welch's t-test does not assume equal variance
)
print(f"T-test (kg/m3): t = {t_stat_norm:.3f}, p = {p_val_norm:.4f}")


########## Discussion of Figure 2b #######

measurement_data_ad = calculate_production_normalized_ch4(
    load_data_func=load_ch4_emissions_with_ad_only,
    calc_biogas_func=calc_biogas_production_rate
)

has_biogas_data = measurement_data_ad[measurement_data_ad["reported_biogas_production"]=='yes']
no_biogas_data = measurement_data_ad[measurement_data_ad["reported_biogas_production"]=='no']

print(f'\nCALCULATING LEAK RATES BASED ON BIOGAS AVAILABILITY\n')

# --- With Biogas data ---
print(f"Mean normalized emissions for facilities with biogas data: {has_biogas_data['production_normalized_CH4_percent'].mean()*100:.4f}%")
print(f"Median normalized emissions for facilities with biogas data: {has_biogas_data['production_normalized_CH4_percent'].median()*100:.4f}%")
print(f"Std Dev normalized emissions for facilities with biogas data: {has_biogas_data['production_normalized_CH4_percent'].std()*100:.4f}%")
print(f'\n')

# --- Without AD ---
print(f"Mean normalized emissions for facilities without biogas data: {no_biogas_data['production_normalized_CH4_percent'].mean()*100:.4f}%")
print(f"Median normalized emissions for facilities without biogas data: {no_biogas_data['production_normalized_CH4_percent'].median()*100:.4f}%")
print(f"Std Dev normalized emissions for facilities without biogas data: {no_biogas_data['production_normalized_CH4_percent'].std()*100:.4f}%")


#%%
########## Discussion of Figure 2b #######



# Load production normalized biogas 
measurement_data_ad = calculate_production_normalized_ch4(
    load_data_func=load_ch4_emissions_with_ad_only,
    calc_biogas_func=calc_biogas_production_rate
)

# Filter measurement data to ensure values are > 0 and not NaN
measurement_data_ad = measurement_data_ad[
    (measurement_data_ad['biogas_production_used_kgCH4_per_hr'] > 0) &
    (measurement_data_ad['production_normalized_CH4_percent']> 0)]
# Filter measurement data to ensure values are > 0 and not NaN
measurement_data_ad = measurement_data_ad[
    (measurement_data_ad['flow_m3_per_day'] > 0) &
    (measurement_data_ad['production_normalized_CH4_percent']> 0)]

min_leak_rate = measurement_data_ad['production_normalized_CH4_percent'].min()
print(f"Minimum production normalized leak rate: {min_leak_rate*100:.2f}%")

max_leak_rate = measurement_data_ad['production_normalized_CH4_percent'].max()
print(f"Minimum production normalized leak rate: {max_leak_rate*100:.2f}%")

median_leak_rate = measurement_data_ad['production_normalized_CH4_percent'].median()
print(f"Minimum production normalized leak rate: {max_leak_rate*100:.2f}%")


# %%

########## Section: Economic opportunities from leak repairs #######

measurement_data_has_biogas = measurement_data_ad[measurement_data_ad['reported_biogas_production']=='yes']
# Range of production normalized CH4 leaks (percent)
print(f'Average leak rate for facilities with reported biogas production: {measurement_data_has_biogas["production_normalized_CH4_percent"].mean():.2%}')
print(f'Range of leak rates for facilities with reported biogas production: {measurement_data_has_biogas["production_normalized_CH4_percent"].min():.2%} - {measurement_data_has_biogas["production_normalized_CH4_percent"].max():.2%} ')

# Mean and SD of production normalized CH4 leak rates (percent): 
mean_leak_rate = measurement_data_has_biogas['production_normalized_CH4_percent'].mean() 
median_leak_rate = measurement_data_has_biogas['production_normalized_CH4_percent'].median()
std_leak_rate = measurement_data_has_biogas['production_normalized_CH4_percent'].std()
print(f'Median leak rate: {median_leak_rate:.2%}, Standard deviation: {std_leak_rate:.2%}')
      # %%
# What leak fraction is needed to offset OGI costs for a "large" facility of 0.5 Mm3/day? 


target_annual = 100_000  # USD/year
plant_size = 500_000  # m³/day
leak_fraction_capturable = 0.8
engine_efficiency = 0.35
electricity_price = 0.08  # USD/kWh

required_leak_rate = solve_leak_rate_for_value(
    target_annual, 
    plant_size, 
    leak_fraction_capturable, 
    engine_efficiency, 
    electricity_price
)


print(f"Fraction gas capturable: {leak_fraction_capturable}")
print(f"Required leak rate for a plant that is {plant_size/1e6:.1f}Mm3/day: {required_leak_rate:.3%}")
leak_fraction_capturable = 0.5

required_leak_rate = solve_leak_rate_for_value(
    target_annual, 
    plant_size, 
    leak_fraction_capturable, 
    engine_efficiency, 
    electricity_price
)

print(f"Fraction gas capturable: {leak_fraction_capturable}")
print(f"Required leak rate for a plant that is {plant_size/1e6:.1f}Mm3/day: {required_leak_rate:.3%}")

# %%
########## Section: Applying to real world plants #######

from a_my_utilities import calc_leak_value

# How many facilities are there with CHP in the United States? 
count_chp = chp_data.shape[0]
print(f"Facilities with CHP: {count_chp}")

# How big are these facilities? 
mean_facility_size_Mm3_per_day = chp_data['flow_m3_per_day'].mean()*1e-6
print(f"Mean facility size: {mean_facility_size_Mm3_per_day}")

median_facility_size_Mm3_per_day = chp_data['flow_m3_per_day'].median()*1e-6
print(f"Median facility size: {median_facility_size_Mm3_per_day}")

std_dev_facility_size_Mm3_per_day = chp_data['flow_m3_per_day'].std()*1e-6
print(f"Stdev facility size: {std_dev_facility_size_Mm3_per_day}")

## CONSERVATIVE CONDITIONS 
chp_data['annual_revenue_conservative'] = chp_data['flow_m3_per_day'].apply(lambda x: calc_annual_revenue(plant_size=x, leak_rate=0.05, leak_fraction_capturable=0.5, engine_efficiency=0.45, electricity_price_per_kWh=0.08))

############ MAKE TABLE 1 ###############


import pandas as pd

# --- Inputs / constants ---
threshold = 100_000  # $100k
leak_rates = [0.05, 0.15, 0.3]
capturable_fracs = [0.5, 0.8]

# Total national flow (all WWTPs, not only CHP subset)
total_national_flow = wwtp_data['flow_m3_per_day'].sum()

# Helper to compute metrics for a scenario
def scenario_metrics(leak_rate, capturable):
    # Compute annual revenue for each CHP facility under this scenario
    annual_rev = chp_data['flow_m3_per_day'].apply(
        lambda x: calc_annual_revenue(
            plant_size=x,
            leak_rate=leak_rate,
            leak_fraction_capturable=capturable,
            engine_efficiency=0.45,
            electricity_price_per_kWh=0.08
        )
    )
    # Mask of facilities above the threshold
    mask = annual_rev > threshold

    # Outputs
    n_facilities = int(mask.sum())
    share_national_flow = (
        chp_data.loc[mask, 'flow_m3_per_day'].sum() / total_national_flow
        if total_national_flow > 0 else float('nan')
    )
    return n_facilities, share_national_flow

# Collect results
rows = []
for lr in leak_rates:
    for cap in capturable_fracs:
        n_fac, flow_share = scenario_metrics(lr, cap)
        rows.append({
            "Leak rate": lr,
            "Capturable fraction": cap,
            "Facilities > $100k": n_fac,
            "Share of national flow": flow_share
        })

# Build and print a readable table
df = pd.DataFrame(rows)

# Formatters for pretty printing in plain text (good for Word)
formatters = {
    "Leak rate": lambda v: f"{v:.0%}",
    "Capturable fraction": lambda v: f"{v:.0%}",
    "Facilities > $100k": lambda v: f"{v:,}",
    "Share of national flow": lambda v: f"{v:.2%}"
}

print("\n=== Revenue > $100,000 — Facilities and National Flow Share by Scenario ===")
print(df.to_string(index=False, formatters=formatters))

# Optional: also print each scenario on its own line (easy to paste inline in text)
print("\n--- Scenario summaries ---")
for _, r in df.iterrows():
    print(
        f"Leak rate {r['Leak rate']:.0%}, Capturable {r['Capturable fraction']:.0%}: "
        f"Facilities > $100k = {int(r['Facilities > $100k']):,}; "
        f"Share of national flow = {r['Share of national flow']:.2%}"
    )

# --- Mean and median flow for >$100k facilities in each scenario ---
rows_flow = []
for lr in leak_rates:
    for cap in capturable_fracs:
        # Recompute annual revenue for this scenario
        annual_rev = chp_data['flow_m3_per_day'].apply(
            lambda x: calc_annual_revenue(
                plant_size=x,
                leak_rate=lr,
                leak_fraction_capturable=cap,
                engine_efficiency=0.45,
                electricity_price_per_kWh=0.08
            )
        )
        mask = annual_rev > threshold
        selected = chp_data.loc[mask, 'flow_m3_per_day']

        if not selected.empty:
            mean_flow = selected.mean() * 1e-6  # convert to Mm³/day
            median_flow = selected.median() * 1e-6
        else:
            mean_flow, median_flow = float('nan'), float('nan')

        rows_flow.append({
            "Leak rate": lr,
            "Capturable fraction": cap,
            "Mean flow (Mm³/day)": mean_flow,
            "Median flow (Mm³/day)": median_flow
        })

df_flow = pd.DataFrame(rows_flow)

# Pretty print table
formatters_flow = {
    "Leak rate": lambda v: f"{v:.0%}",
    "Capturable fraction": lambda v: f"{v:.0%}",
    "Mean flow (Mm³/day)": lambda v: f"{v:.2f}",
    "Median flow (Mm³/day)": lambda v: f"{v:.2f}"
}

print("\n=== Mean and Median Flow of Facilities with Revenue > $100,000 ===")
print(df_flow.to_string(index=False, formatters=formatters_flow))

# Optional: line-by-line summaries (good for pasting into text)
print("\n--- Scenario flow summaries ---")
for _, r in df_flow.iterrows():
    print(
        f"Leak rate {r['Leak rate']:.0%}, Capturable {r['Capturable fraction']:.0%}: "
        f"Mean flow = {r['Mean flow (Mm³/day)']:.2f} Mm³/day; "
        f"Median flow = {r['Median flow (Mm³/day)']:.2f} Mm³/day"
    )


