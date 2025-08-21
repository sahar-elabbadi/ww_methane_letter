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
