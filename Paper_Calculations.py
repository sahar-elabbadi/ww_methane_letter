#%% 
#Setup
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from a_my_utilities import set_chini_dataset, calc_biogas_production_rate, load_ch4_emissions_with_ad_only, calculate_production_normalized_ch4, calc_annual_revenue
from a_my_utilities import solve_leak_rate_for_value

import matplotlib.ticker as mtick

chini_data = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))  # or csv

set_chini_dataset(
    chini_data,
    x_col="flow_m3_per_day",
    y_col="methane_gen_kgh",
    drop_negative=True
)



measurement_data_ad = calculate_production_normalized_ch4(
    load_data_func=load_ch4_emissions_with_ad_only,
    calc_biogas_func=calc_biogas_production_rate
)

# Filter measurement data to ensure values are > 0 and not NaN
measurement_data_ad = measurement_data_ad[
    (measurement_data_ad['biogas_production_used_kgCH4_per_hr'] > 0) &
    (measurement_data_ad['production_normalized_CH4_percent']> 0)]

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

print(f"Required leak rate for a plant that is {plant_size/1e6:.1f}Mm3/day: {required_leak_rate:.3%}")
# %%
