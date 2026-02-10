#%% 
#Setup
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from a_my_utilities import get_chini_sample_size, get_chini_se_slope, get_chini_sigma, set_chini_dataset, load_ch4_emissions_data, calc_biogas_production_rate, load_ch4_emissions_with_ad_only, calculate_production_normalized_ch4, calc_annual_revenue
from a_my_utilities import solve_leak_rate_for_value, get_chini_slope, METHANE_MJ_PER_KG, get_chini_r2_centered, get_chini_r2_origin, annualized_cost, ENGINES, METHANE_KG_PER_SCF
import matplotlib.ticker as mtick
import matplotlib.ticker as ticker


#%%
###### LOAD DATA ######

# Load Chini dataset and cache linear regression 
chini_data = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))  # or csv

set_chini_dataset(
    chini_data,
    x_col="flow_m3_per_day",
    y_col="methane_gen_kgh",
    drop_negative=True
)

# All measurement data 
measurement_data = load_ch4_emissions_data()

# # Load measurement data
# measurement_data_ad = calculate_production_normalized_ch4(
#     load_data_func=load_ch4_emissions_with_ad_only,
#     calc_biogas_func=calc_biogas_production_rate
# )

# # Filter measurement data to ensure values are > 0 and not NaN
# measurement_data_ad = measurement_data_ad[
#     (measurement_data_ad['biogas_production_used_kgCH4_per_hr'] > 0) &
#     (measurement_data_ad['production_normalized_CH4_percent']> 0)]

measurement_data_ad_raw = calculate_production_normalized_ch4(
    load_data_func=load_ch4_emissions_with_ad_only,
    calc_biogas_func=calc_biogas_production_rate
)
measurement_data_ad_filt = measurement_data_ad_raw.query(
    "biogas_production_used_kgCH4_per_hr > 0 and production_normalized_CH4_percent > 0"
)

# Rows that were filtered out
filtered_out = measurement_data_ad_raw.loc[
    ~measurement_data_ad_raw.index.isin(measurement_data_ad_filt.index)
]

measurement_data_has_biogas_data = measurement_data_ad_filt.query("reported_biogas_production == 'yes'")


# Load El Abbadi, Feng et al 2025 data on facilities with CHP 
chp_data = pd.read_csv(pathlib.Path("02_clean_data", "chp_data.csv"))
wwtp_data = pd.read_csv(pathlib.Path("02_clean_data", "wwtp_data.csv"))


#%% 

######### Section: Introduction Calculations #######

# Convert the MMT CO2-eq from El Abbadi, Feng et al 2025 to MMT CH4
total_ch4_mmtCO2eq_low = 15 # Low estimate is 15 MMT CO2-eq/year 
total_ch4_mmtCO2eq_high = 25 # High estimate is 25 MMT CO2-eq/year

# GWP of methane over a 100 year time frame 
gwp_ch4 = 29.8 # Median value from EL Abbadi, Feng et al 2025

total_ch4_mmtCH4_low = total_ch4_mmtCO2eq_low / gwp_ch4
total_ch4_mmtCH4_high = total_ch4_mmtCO2eq_high / gwp_ch4

# Print
print(f"Total methane emissions from WWTPs in the US (low estimate): {total_ch4_mmtCH4_low:.2f} MMT CH4/year")
print(f"Total methane emissions from WWTPs in the US (high estimate): {total_ch4_mmtCH4_high:.2f} MMT CH4/year")

# California's annual natural gas production in 2023 (according to EIA: https://www.eia.gov/dnav/ng/hist/na1160_sca_2a.htm)
ca_ng_production2023 = 124_024 # Units: million cubic feet (MMcf)

# Convert to MMT CH4
ca_ng_production_mmtCH4 = (ca_ng_production2023 * 1e6 * METHANE_KG_PER_SCF) / 1e9  # Convert to MMT CH4
print(f"California's annual natural gas production in 2023: {ca_ng_production_mmtCH4:.2f} MMT CH4/year")

# What fraction of CA's natural gas production is equivalent to national WRRF emissions? 
fraction_equiv_high = total_ch4_mmtCH4_high / ca_ng_production_mmtCH4
fraction_equiv_low = total_ch4_mmtCH4_low / ca_ng_production_mmtCH4

print(f"Fraction of California's natural gas production equivalent to national WRRF methane emissions (low estimate): {fraction_equiv_low:.2%}")
print(f"Fraction of California's natural gas production equivalent to national WRRF methane emissions (high estimate): {fraction_equiv_high:.2%}")

#%%
######### Section: Comparison of measurement-based emissions factors from WRRFs #######

# How many facilities do not report biogas production?
measurement_data_no_biogas = measurement_data_ad_raw[measurement_data_ad_raw['reported_biogas_production'] == 'no'].shape[0]
print(f"Number of facilities that do not report biogas production: {measurement_data_no_biogas}")
# Percent of facilities that do not report biogas production:
print(f"Percent of facilities that do not report biogas production: {measurement_data_no_biogas / measurement_data_ad_raw.shape[0] * 100:.2f}%")

#%%
### LINEAR REGRESSION CALCLATIONS ####

# What is the slope of the Chini dataset in MJ biogas per m3? 

slope = get_chini_slope()          # kg CH4/h per (m^3 wastewater/day)
slope_kg_per_m3 = slope * 24  # Convert to kg CH4 per m3 wastewater
slope_MJ_per_m3 = slope_kg_per_m3 *  METHANE_MJ_PER_KG # Convert to MJ biogas per m3 wastewater

# print
print(f"Chini slope: {slope:.4f} kg CH4/h per (m^3/day) wastewater")
print(f"Chini slope: {slope_kg_per_m3:.4f} kg CH4 per m3 wastewater")
print(f"Chini slope: {slope_MJ_per_m3:.4f} MJ biogas per m3 wastewater")

# What is the R2 of the Chini dataset?
r2_centered = get_chini_r2_centered()
print(f"Chini R2 (centered, typical calculations): {r2_centered:.5f}")

# What is R2 for through-origin fit (uncentered R2)? (excel calculation style): 
r2_origin = get_chini_r2_origin()
print(f"Chini R2 (through-origin, uncentered): {r2_origin:.5f}")

# What is standard error of the regression on the slope?: 
se_slope = get_chini_sigma()
print(f"Chini standard error of regression: {se_slope:.5f}")

# What is standard error of the mean on the slope?: 
se_slope = get_chini_se_slope()
print(f"Chini standard error of slope: {se_slope:.5f}")


# Print sample size of Chini dataset
sample_size = get_chini_sample_size()
print(f"Sample size of Chini dataset: {sample_size}")

# %%

#%% 
########## Discussion of Figure 2a #######

from scipy import stats

# Data setup 
# calculate methane emissions per m3 wastewater treated 
measurement_data['ch4_kg_per_m3'] = measurement_data['ch4_kg_per_hr'] / (measurement_data['flow_m3_per_day'] / 24)

# facilities with AD 
measurement_data_all_has_ad = measurement_data[measurement_data['has_ad']=='yes']
# facilities without AD
measurement_data_all_no_ad = measurement_data[measurement_data['has_ad']=='no']


print("Fig 2a: kg/hr emissions rate")

# --- With AD ---
print("Facilities with AD:")
with_ad_mean = measurement_data_all_has_ad['ch4_kg_per_hr'].mean()
with_ad_median = measurement_data_all_has_ad['ch4_kg_per_hr'].median()
with_ad_std = measurement_data_all_has_ad['ch4_kg_per_hr'].std() 
print(f"Mean emissions rate for facilities with AD: {with_ad_mean:.2f} kg/hr")
print(f"Median emissions rate for facilities with AD: {with_ad_median:.2f} kg/hr")
print(f"Std Dev emissions rate for facilities with AD: {with_ad_std:.2f} kg/hr")

# print(f"Mean emissions rate for facilities with AD: {measurement_data_all_has_ad['ch4_kg_per_hr'].mean():.2f} kg/hr")
# print(f"Median emissions rate for facilities with AD: {measurement_data_all_has_ad['ch4_kg_per_hr'].median():.2f} kg/hr")
# print(f"Std Dev emissions rate for facilities with AD: {measurement_data_all_has_ad['ch4_kg_per_hr'].std():.2f} kg/hr\n")

# Parameters for 95% Confidence Interval calculations
C = 0.95 # 95% confidence interval
z = 1.95996 # z-score for 95% confidence interval, Hazra et al 2017

# Calculations for 95% CI for facilities with AD 
with_ad_sem = measurement_data_all_has_ad['ch4_kg_per_hr'].sem()  # Standard error of the mean
with_ad_CI_low = with_ad_mean - z * with_ad_sem
with_ad_CI_upper = with_ad_mean + z * with_ad_sem 
print(f"95% Confidence interval for facilities with AD: ({with_ad_CI_low:.2f}, {with_ad_CI_upper:.2f}) kg/hr\n")


# --- Without AD ---
print("Facilities without AD:")

print(f"Mean emissions rate for facilities without AD: {measurement_data_all_no_ad['ch4_kg_per_hr'].mean():.2f} kg/hr")
print(f"Median emissions rate for facilities without AD: {measurement_data_all_no_ad['ch4_kg_per_hr'].median():.2f} kg/hr")
print(f"Std Dev emissions rate for facilities without AD: {measurement_data_all_no_ad['ch4_kg_per_hr'].std():.2f} kg/hr")

# Calculations for 95% CI for facilities without AD 
no_ad_sem = measurement_data_all_no_ad['ch4_kg_per_hr'].sem()  # Standard error of the mean
no_ad_CI_low = measurement_data_all_no_ad['ch4_kg_per_hr'].mean() - z * no_ad_sem
no_ad_CI_upper = measurement_data_all_no_ad['ch4_kg_per_hr'].mean() + z * no_ad_sem

print(f"95% Confidence interval for facilities without AD: ({no_ad_CI_low:.4f}, {no_ad_CI_upper:.4f}) kg/hr\n")

######## Normalized by flow rate ##########
# Add normalized emissions column (kg CH4 per m3 wastewater)

print("Fig 2a: kg/m3 emissions rate")
# --- With AD ---
print("Facilities with AD:")
# print(f"Mean normalized emissions for facilities with AD: {measurement_data_all_has_ad['ch4_kg_per_m3'].mean():.4f} kg/m3")
# print(f"Median normalized emissions for facilities with AD: {measurement_data_all_has_ad['ch4_kg_per_m3'].median():.4f} kg/m3")
# print(f"Std Dev normalized emissions for facilities with AD: {measurement_data_all_has_ad['ch4_kg_per_m3'].std():.4f} kg/m3\n")

with_ad_mean = measurement_data_all_has_ad['ch4_kg_per_m3'].mean()
with_ad_median = measurement_data_all_has_ad['ch4_kg_per_m3'].median()
with_ad_std = measurement_data_all_has_ad['ch4_kg_per_m3'].std() 
print(f"Mean normalized emissions for facilities with AD: {with_ad_mean:.4f} kg/m3")
print(f"Median normalized emissions for facilities with AD: {with_ad_median:.4f} kg/m3")
print(f"Std Dev normalized emissions for facilities with AD: {with_ad_std:.4f} kg/m3\n")

# Calculations for 95% CI for facilities with AD 
with_ad_sem = measurement_data_all_has_ad['ch4_kg_per_m3'].sem()  # Standard error of the mean
with_ad_CI_low = with_ad_mean - z * with_ad_sem
with_ad_CI_upper = with_ad_mean + z * with_ad_sem 

print(f"95% Confidence interval for facilities with AD: ({with_ad_CI_low:.4f}, {with_ad_CI_upper:.4f}) kg/m3\n")


# --- Without AD ---
print("Facilities without AD:")
no_ad_mean = measurement_data_all_no_ad['ch4_kg_per_m3'].mean()
no_ad_median = measurement_data_all_no_ad['ch4_kg_per_m3'].median()
no_ad_std = measurement_data_all_no_ad['ch4_kg_per_m3'].std() 
print(f"Mean normalized emissions for facilities without AD: {no_ad_mean:.4f} kg/m3")
print(f"Median normalized emissions for facilities without AD: {no_ad_median:.4f} kg/m3")
print(f"Std Dev normalized emissions for facilities without AD: {no_ad_std:.4f} kg/m3\n")

# Calculations for 95% CI for facilities without AD 
no_ad_sem = measurement_data_all_no_ad['ch4_kg_per_m3'].sem()  # Standard error of the mean
no_ad_CI_low = no_ad_mean - z * no_ad_sem
no_ad_CI_upper = no_ad_mean + z * no_ad_sem 

print(f"95% Confidence interval for facilities without AD: ({no_ad_CI_low:.4f}, {no_ad_CI_upper:.4f}) kg/m3\n")

# print(f"Mean normalized emissions for facilities without AD: {measurement_data_all_no_ad['ch4_kg_per_m3'].mean():.4f} kg/m3")
# print(f"Median normalized emissions for facilities without AD: {measurement_data_all_no_ad['ch4_kg_per_m3'].median():.4f} kg/m3")
# print(f"Std Dev normalized emissions for facilities without AD: {measurement_data_all_no_ad['ch4_kg_per_m3'].std():.4f} kg/m3")

#%%
########## Statistical Testing ##########

# t-test for normalized emissions (kg/m3)
t_stat_norm, p_val_norm = stats.ttest_ind(
    measurement_data_all_has_ad['ch4_kg_per_m3'].dropna(),
    measurement_data_all_no_ad['ch4_kg_per_m3'].dropna(),
    equal_var=False #Welch's t-test does not assume equal variance
)
print(f"T-test (kg/m3): t = {t_stat_norm:.3f}, p = {p_val_norm:.4f}")

# What is the average size of facilities with AD and average size of facilities without AD? 
print(f"\nHow do facility sizes compare?")
print(f"Mean facility size for facilities with AD: {measurement_data_all_has_ad['flow_m3_per_day'].mean():,.2f} m3/day")
print(f"Mean facility size for facilities witout AD: {measurement_data_all_no_ad['flow_m3_per_day'].mean():,.2f} m3/day")

# t-test for normalized emissions (kg/m3)
t_stat_norm, p_val_norm = stats.ttest_ind(
    measurement_data_all_has_ad['flow_m3_per_day'].dropna(),
    measurement_data_all_no_ad['flow_m3_per_day'].dropna(),
    equal_var=False #Welch's t-test does not assume equal variance
)
print(f"T-test (kg/m3): t = {t_stat_norm:.3f}, p = {p_val_norm:.4f}")

#%%
########## Discussion of Figure 2b #######

# measurement_data_ad = calculate_production_normalized_ch4(
#     load_data_func=load_ch4_emissions_with_ad_only,
#     calc_biogas_func=calc_biogas_production_rate
# )
# measurement_data_ad = measurement_data_ad_filt


has_biogas_data = measurement_data_ad_filt[measurement_data_ad_filt["reported_biogas_production"]=='yes']
no_biogas_data = measurement_data_ad_filt[measurement_data_ad_filt["reported_biogas_production"]=='no']

print(f'\nCALCULATING LEAK RATES BASED ON BIOGAS AVAILABILITY\n')

# --- With Biogas data ---
print(f"Mean normalized emissions for facilities with biogas data: {has_biogas_data['production_normalized_CH4_percent'].mean()*100:.4f}%")
# Confidence intervals for the mean 
z=1.95996 # z-score for 95% confidence interval, Hazra et al 2017
lower_ci = has_biogas_data['production_normalized_CH4_percent'].mean() - z * has_biogas_data['production_normalized_CH4_percent'].sem()
upper_ci = has_biogas_data['production_normalized_CH4_percent'].mean() + z * has_biogas_data['production_normalized_CH4_percent'].sem()
print(f"95% Confidence interval for facilities with biogas data: ({lower_ci*100:.4f}%, {upper_ci*100:.4f}%)")

print(f"Median normalized emissions for facilities with biogas data: {has_biogas_data['production_normalized_CH4_percent'].median()*100:.4f}%")
print(f"Std Dev normalized emissions for facilities with biogas data: {has_biogas_data['production_normalized_CH4_percent'].std()*100:.4f}%")
print(f'\n')

# --- Without AD ---
print(f"Mean normalized emissions for facilities without biogas data: {no_biogas_data['production_normalized_CH4_percent'].mean()*100:.4f}%")
lower_ci = no_biogas_data['production_normalized_CH4_percent'].mean() - z * no_biogas_data['production_normalized_CH4_percent'].sem()
upper_ci = no_biogas_data['production_normalized_CH4_percent'].mean() + z * no_biogas_data['production_normalized_CH4_percent'].sem()
print(f"95% Confidence interval for facilities without biogas data: ({lower_ci*100:.4f}%, {upper_ci*100:.4f}%)")

print(f"Median normalized emissions for facilities without biogas data: {no_biogas_data['production_normalized_CH4_percent'].median()*100:.4f}%")
print(f"Std Dev normalized emissions for facilities without biogas data: {no_biogas_data['production_normalized_CH4_percent'].std()*100:.4f}%")


#%%
########## Discussion of Figure 2b #######


# # Load production normalized biogas 
# measurement_data_ad = calculate_production_normalized_ch4(
#     load_data_func=load_ch4_emissions_with_ad_only,
#     calc_biogas_func=calc_biogas_production_rate
# )

# # Filter measurement data to ensure values are > 0 and not NaN
# measurement_data_ad = measurement_data_ad[
#     (measurement_data_ad['biogas_production_used_kgCH4_per_hr'] > 0) &
#     (measurement_data_ad['production_normalized_CH4_percent']> 0)]

# # Filter measurement data to ensure values are > 0 and not NaN
# measurement_data_ad = measurement_data_ad[
#     (measurement_data_ad['flow_m3_per_day'] > 0) &
#     (measurement_data_ad['production_normalized_CH4_percent']> 0)]

min_leak_rate = measurement_data_ad_filt['production_normalized_CH4_percent'].min()
print(f"Minimum production normalized leak rate: {min_leak_rate*100:.2f}%")

max_leak_rate = measurement_data_ad_filt['production_normalized_CH4_percent'].max()
print(f"Maximum production normalized leak rate: {max_leak_rate*100:.2f}%")

median_leak_rate = measurement_data_ad_filt['production_normalized_CH4_percent'].median()
print(f"Median production normalized leak rate: {median_leak_rate*100:.2f}%")


# %% 
####### Examine Electricity Data (EIA) #######

# Assign electricity value to chp_data based on EIA industrial tariffs 
# chp_data["electricity_cost"] = chp_data["state"].map(eia_industrial_tariffs_2023)

# Print results
print(f"Electricity price data for facilities with CHP:")
print(chp_data["electricity_cost"].describe(percentiles=[0.25, 0.5, 0.75]))

# Histogram
plt.hist(chp_data["electricity_cost"], bins=20, edgecolor="black")
plt.xlabel("Electricity Cost")
plt.ylabel("Frequency")
plt.title("Distribution of Electricity Costs for Facilities with CHP")
plt.show()
# Note: median electricity price calculated is $0.09 / kWh


# Print results
print(f"Natural gas price data for facilities with CHP:")
print(chp_data["natural_gas_cost"].describe(percentiles=[0.25, 0.5, 0.75]))

# Histogram
plt.hist(chp_data["natural_gas_cost"], bins=20, edgecolor="black")
plt.xlabel("Natural Gas Cost")
plt.ylabel("Frequency")
plt.title("Distribution of Natural Gas Costs for Facilities with CHP")
plt.show()

# Note: median natural gas price calculated is $0.0079 / MJ

#%% 
### Plot industrial electricity tariffs by state
import matplotlib.pyplot as plt 
# Plot electricity data histogram 
industrial_electricity_data = pathlib.Path("02_clean_data", "eia_industrial_tariffs_2023.csv")
eia2023 = pd.read_csv(industrial_electricity_data)

# Sort the dataframe by price
eia2023_sorted = eia2023.sort_values(by="Price ($/kWh)").reset_index(drop=True)

# Scatter plot with y-axis starting at 0
plt.figure(figsize=(12,6))
plt.scatter(range(len(eia2023_sorted)), eia2023_sorted["Price ($/kWh)"], color="blue", alpha=0.7)

# Keep all state labels
plt.xticks(range(len(eia2023_sorted)), eia2023_sorted["State"], rotation=90, fontsize=10)
plt.yticks(fontsize=12)

plt.xlabel("State (sorted by tariff)", fontsize=14)
plt.ylabel("Price ($/kWh)", fontsize=14)
plt.title("Industrial Electricity Tariffs by State (2023) - Sorted", fontsize=16)

# Remove grid and set y-axis to start at 0
plt.grid(False)
plt.ylim(bottom=0)

plt.tight_layout()
plt.show()

# Get summary statistics of the tariff data
summary_stats = eia2023["Price ($/kWh)"].describe()
summary_stats

#%% 
### Plot industrial natural gas tariffs by state
import matplotlib.pyplot as plt 
# Plot natural gas data histogram 
industrial_natural_gas_data = pathlib.Path("02_clean_data", "eia_industrial_tariffs_natural_gas_2023.csv")
eia2023_ng = pd.read_csv(industrial_natural_gas_data)

# Sort the dataframe by price
eia2023_ng_sorted = eia2023_ng.sort_values(by="Price ($/MJ)").reset_index(drop=True)

# Scatter plot with y-axis starting at 0
plt.figure(figsize=(12,6))
plt.scatter(range(len(eia2023_ng_sorted)), eia2023_ng_sorted["Price ($/MJ)"], color="blue", alpha=0.7)

# Keep all state labels
plt.xticks(range(len(eia2023_ng_sorted)), eia2023_ng_sorted["State"], rotation=90, fontsize=10)
plt.yticks(fontsize=12)

plt.xlabel("State (sorted by tariff)", fontsize=14)
plt.ylabel("Price ($/MJ)", fontsize=14)
plt.title("Industrial Natural Gas Tariffs by State (2023) - Sorted", fontsize=16)

# Remove grid and set y-axis to start at 0
plt.grid(False)
plt.ylim(bottom=0)

plt.tight_layout()
plt.show()

# Get summary statistics of the tariff data
summary_stats_ng = eia2023_ng["Price ($/MJ)"].describe()
summary_stats_ng

#%%
# Examine size of facilities with CHP 
plt.hist(chp_data["flow_m3_per_day"], bins=20, edgecolor="black")
plt.xlabel("Flow (m3/day)")
plt.ylabel("Frequency")
plt.title("Distribution of Facility Size")
plt.show()

#%% 
########## Section: Economic opportunities from leak repairs #######

measurement_data_has_biogas_data = measurement_data_ad_filt[measurement_data_ad_filt['reported_biogas_production']=='yes']
# print(measurement_data_has_biogas_data["production_normalized_CH4_percent"].head())
# Range of production normalized CH4 leaks (percent)
print(f'Average leak rate for facilities with reported biogas production: {measurement_data_has_biogas_data["production_normalized_CH4_percent"].mean():.2%}')
print(f'Median leak rate for facilities with reported biogas production: {measurement_data_has_biogas_data["production_normalized_CH4_percent"].median():.2%}')

print(f'Range of leak rates for facilities with reported biogas production: {measurement_data_has_biogas_data["production_normalized_CH4_percent"].min():.2%} - {measurement_data_has_biogas_data["production_normalized_CH4_percent"].max():.2%} ')

#%%
# What fraction of facilities in the US have flow up to 1.6 Mm3/day? 
# chp_data, wwtp_data loaded earlier

flow_threshold_m3_per_day = 1_600_000  # 1.6 Mm3/day
num_facilities_below_threshold_chp = chp_data[chp_data['flow_m3_per_day'] <= flow_threshold_m3_per_day].shape[0]
total_chp_facilities = chp_data.shape[0]
fraction_below_threshold = num_facilities_below_threshold_chp / total_chp_facilities
print(f"Fraction of CHP facilities with flow up to {flow_threshold_m3_per_day/1e6:.1f} Mm3/day: {fraction_below_threshold:.2%} ({num_facilities_below_threshold_chp} out of {total_chp_facilities})")

num_facilities_below_threshold_all = wwtp_data[wwtp_data['flow_m3_per_day'] <= flow_threshold_m3_per_day].shape[0]
total_facilities = wwtp_data.shape[0]
fraction_below_threshold = num_facilities_below_threshold_all / total_facilities
print(f"Fraction of U.S. facilities with flow up to {flow_threshold_m3_per_day/1e6:.1f} Mm3/day: {fraction_below_threshold:.2%} ({num_facilities_below_threshold_all} out of {total_facilities})")

flow_threshold_m3_per_day = 500_000  # 0.5 Mm3/day

num_facilities_above_threshold_chp = chp_data[chp_data['flow_m3_per_day'] >= flow_threshold_m3_per_day].shape[0]
fraction_below_threshold = num_facilities_above_threshold_chp / total_chp_facilities
print(f"Fraction of CHP facilities with flow over {flow_threshold_m3_per_day/1e6:.1f} Mm3/day: {fraction_below_threshold:.2%} ({num_facilities_above_threshold_chp} out of {total_chp_facilities})")



#%% 
# Mean and SD of production normalized CH4 leak rates (percent): 
mean_leak_rate = measurement_data_has_biogas_data['production_normalized_CH4_percent'].mean() 
median_leak_rate = measurement_data_has_biogas_data['production_normalized_CH4_percent'].median()
std_leak_rate = measurement_data_has_biogas_data['production_normalized_CH4_percent'].std()
print(f'Median leak rate: {median_leak_rate:.2%}, Standard deviation: {std_leak_rate:.2%}')

# What leak fraction is needed to offset OGI costs for a "large" facility of 0.5 Mm3/day? 

target_annual = 100_000  # USD/year
plant_size = 500_000  # m³/day
leak_fraction_capturable = 0.8
engine = ENGINES['reciprocating_lean_burn'] # SET ENGINE TYPE HERE 
electricity_price = 0.09  # USD/kWh 
nat_gas_price = 0.008  # USD/MJ

required_leak_rate = solve_leak_rate_for_value(
    target_annual, 
    plant_size, 
    leak_fraction_capturable, 
    engine=engine, 
    electricity_price_per_kWh = electricity_price,
    nat_gas_price_per_MJ=nat_gas_price
)

### TEXT: Facilities over 0.5 Mm3 /day
print(f"\nFACILITIES OVER 0.5 Mm3 per day")
print(f"Fraction gas capturable: {leak_fraction_capturable}")
print(f"Required leak rate for a plant that is {plant_size/1e6:.1f}Mm3/day: {required_leak_rate:.3%}")
leak_fraction_capturable = 0.5

required_leak_rate = solve_leak_rate_for_value(
    target_annual, 
    plant_size, 
    leak_fraction_capturable, 
    engine=engine, 
    electricity_price_per_kWh = electricity_price,
    nat_gas_price_per_MJ=nat_gas_price
)

print(f"Fraction gas capturable: {leak_fraction_capturable}")
print(f"Required leak rate for a plant that is {plant_size/1e6:.1f}Mm3/day: {required_leak_rate:.3%}")

# %%
########## Section: Applying to real world plants #######

from a_my_utilities import calc_leak_value_CHP

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

chp_data['annual_revenue_conservative'] = chp_data['flow_m3_per_day'].apply(lambda x: calc_annual_revenue(plant_size=x, leak_rate=0.05, leak_fraction_capturable=0.5, engine=engine, 
                                                                                                          electricity_price_per_kWh=electricity_price, nat_gas_price_per_MJ=nat_gas_price))

#%%
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
    annual_rev = chp_data.apply(
        lambda r: calc_annual_revenue(
            plant_size=r['flow_m3_per_day'],
            leak_rate=leak_rate,
            leak_fraction_capturable=capturable,
            engine=engine, 
            electricity_price_per_kWh=r['electricity_cost'], 
            nat_gas_price_per_MJ=r['natural_gas_cost']
        ),
        axis=1
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
        annual_rev = chp_data.apply(
            lambda r: calc_annual_revenue(
                plant_size=r['flow_m3_per_day'],
                leak_rate=lr,
                leak_fraction_capturable=cap,
                engine=engine, 
                electricity_price_per_kWh=r['electricity_cost'], 
                nat_gas_price_per_MJ=r['natural_gas_cost']
            ), 
            axis=1
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



#%% Annualized cost of OGI camera 

capital_cost = 200_000
lifetime_years = 10 
discount_rate = 0.07
ogi_annualized_cost = annualized_cost(capital_cost, lifetime_years, discount_rate)
print(f"Annualized OGI camera costs with lifetimes of {lifetime_years}, discount rate of {discount_rate*100}%, and capital cost of {capital_cost}\n")
print(f"${ogi_annualized_cost:,.2f}")

#%% 
import pathlib 
import pandas as pd

# Fredenslund et al 2023 cost of repairs 
dkk_per_usd_2021 = pd.read_excel(pathlib.PurePath('01_raw_data','danish_kroner_to_usd.xls'))
print(f"Average DKK per USD in 2021: {dkk_per_usd_2021['DEXDNUS'].mean():.2f}")

repair_costs_dkk_low = 100_000
repair_costs_dkk_high = 22_500_000
print(f"Low costs of repairs in 2021: ${repair_costs_dkk_low / dkk_per_usd_2021['DEXDNUS'].mean():,.2f} ")
print(f"Low costs of repairs in 2021: ${repair_costs_dkk_high / dkk_per_usd_2021['DEXDNUS'].mean():,.2f} ")

# Convert to 2023 dollars: 
# data saved in 01_raw_data/CPI_2021-2023.xlsx
cpi_2021 = 270.970
cpi_2023 = 304.702 
print(f"Low costs of repairs in 2023: ${repair_costs_dkk_low / dkk_per_usd_2021['DEXDNUS'].mean()*cpi_2023/cpi_2021:,.2f} ")
print(f"Low costs of repairs in 2023: ${repair_costs_dkk_high / dkk_per_usd_2021['DEXDNUS'].mean()*cpi_2023/cpi_2021:,.2f} ")
#%%
usd_per_eu_2023 = pd.read_excel(pathlib.PurePath('01_raw_data','EU_to_USD.xls'))
print(f"Average DKK per USD in 2023: {usd_per_eu_2023['DEXUSEU'].mean():.2f}")

# Hurtig et al 2025 Table 2, currency year assumed to be 2023 based on timeline of paper submission in early 2024
# Cost of repairing connection From Hurtig et al 2025, Table 2
repair_costs_eu_connections_low = 10
repair_costs_eu_connections_high = 300
print(f"Low costs of connection repairs in 2023: ${repair_costs_eu_connections_low * usd_per_eu_2023['DEXUSEU'].mean():,.2f} ")
print(f"High costs of repairs in 2023: ${repair_costs_eu_connections_high * usd_per_eu_2023['DEXUSEU'].mean():,.2f} ")

# Cost of repairing flanges From Hurtig et al 2025, Table 2
repair_costs_eu__flanges_low = 25
repair_costs_eu__flanges_high = 1000
print(f"Low costs of flange repairs in 2023: ${repair_costs_eu__flanges_low * usd_per_eu_2023['DEXUSEU'].mean():,.2f} ")
print(f"High costs of flange repairs in 2023: ${repair_costs_eu__flanges_high * usd_per_eu_2023['DEXUSEU'].mean():,.2f} ")

# Cost of repairing dome From Hurtig et al 2025, Table 2
repair_costs_eu__dome_low = 30_000
repair_costs_eu__dome_high = 35_000
print(f"Low costs of AD dome repairs in 2023: ${repair_costs_eu__dome_low * usd_per_eu_2023['DEXUSEU'].mean():,.2f} ")
print(f"High costs of AD dome repairs in 2023: ${repair_costs_eu__dome_high * usd_per_eu_2023['DEXUSEU'].mean():,.2f} ")


# Cost of repairing gas storage From Hurtig et al 2025, Table 2
repair_costs_eu__storage_low = 15_000
repair_costs_eu__storage_high = 25_000
print(f"Low costs of membrane storage repairs in 2023: ${repair_costs_eu__storage_low * usd_per_eu_2023['DEXUSEU'].mean():,.2f} ")
print(f"High costs of membrane storage repairs in 2023: ${repair_costs_eu__storage_high * usd_per_eu_2023['DEXUSEU'].mean():,.2f} ")


# Cost of leak survey: 400 euro to 1200 euro 
survey_cost_low = 400
survey_cost_high = 1200
print(f"Survey cost low (IR camera, gas analyser, flow meters): ${survey_cost_low * usd_per_eu_2023['DEXUSEU'].mean():,.2f} ")
print(f"Survey cost high (OGI, tuneable diode laser, etc.): ${survey_cost_high * usd_per_eu_2023['DEXUSEU'].mean():,.2f} ")
