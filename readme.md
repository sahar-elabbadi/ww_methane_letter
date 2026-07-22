# Read Me

Author: Sahar El Abbadi 
Date last modified: July 13, 2026

### Directories 

01_raw_data: contains all raw data used in analysis. Raw data files are never directly edited, but rather loaded and used as inputs to data cleaning code, which saves cleaned data as a separate file. 

02_clean_data: contains all cleaned data files, which are used as inputs to analysis. Cleaned data files are: 
- ad_data.csv: subset of El Abbadi, Feng et al., 2025 dataset, filtered to only include wastewater treatment plants with anaerobic digestion. Additional columns added during data cleaning are: 
    - has_chp: yes / no to indicate presence or absence of electricity recovery via CHP
    - has_ad: yes / no to indicate presence or absence of anaerobic digestion for solids treatment
    - flow_m3_per_day: flow rate of wastewater treated, converted from million gallons per day to m3 per day
    - electricity_cost: state average price of electricity according to EIA, in units of USD per kWh, currency year 2023
    - natural_gas_cost: state average industrial price of natural gas according to EIA, in units of USD per MJ, currency year 2023
- chini_cleaned.csv: cleaned file of compiled data from Chini and Stillwell (2018). Includes calculated values of flow rate in m3 per day and biogas generated as methane in units of kg CH4 / hr. Extractign data from the raw files provided by Chini and Stillwell was performed manually by the research team. Final data cleaning and unit conversions for generating this file are in d_clean_chini_data.py 
- chp_data.csv: subset of El Abbadi, Feng et al., 2025 dataset, filtered to only include wastewater treatment plants with combined heat and power for energy recovery from biogas. Additional columns added during data cleaning are: 
    - has_chp: yes / no to indicate presence or absence of electricity recovery via CHP
    - has_ad: yes / no to indicate presence or absence of anaerobic digestion for solids treatment
    - flow_m3_per_day: flow rate of wastewater treated, converted from million gallons per day to m3 per day
    - electricity_cost: state average price of electricity according to EIA, in units of USD per kWh, currency year 2023
    - natural_gas_cost: state average industrial price of natural gas according to EIA, in units of USD per MJ, currency year 2023
- chp_mc_summary.csv: each row shows the results of estimated revenue from Monte Carlo analysis (median, 2.5 and 97.5 percentile, mean, and upper and lower CI values), run across three different leak rate scenarios. Option to also print the electricity price. Generated through the function run_mc_for_all_plants in Paper_Calcs_MonteCarlo.py. 
- chp_national_mc_summary.csv: Summarize results of chp_mc_summary by summing across all plants. 
- eia_industrial_tariffs_2023.csv: EIA industrial tarrifs for electricity in $2023 / kWh. Generated in function load_eia_data in a_my_utilities.py  
- eia_industrial_tariffs_natural_gas_2023.csv:  EIA industrial tarrifs for electricity in $2023 / MJ. Generated in a_my_utilities.py by function make_eia_industrial_ng_2023 
- measurement_data_ad.csv: measurement dataset (measurement_data.csv) filtered based on facilities with biogas production. Generated in Fig2_combined.py 
- measurement_data.csv: measurement data from all measurement data sources. Generated in b_data_cleaning.py 
- wwtp_data.csv: all WWTPs reported in El Abbadi, Feng et al 2025. 

03_figures: figures used in manuscript

zArchive: old scripts not used in final analysis. 

### Overview of scripts 

1. a_my_utilities.py: contains functions used across different scripts (unit conversions, loading data, etc.)

2. b_data_cleaning.py
    - Functions to clean all measurement datasets, in the following order: Moore et al., 2023; Moore et al., 2025; Song et al., 2023; Fredenslund et al., 2023 (including person equivalent data from Wechselberger et al); Galfalk et al, 2025. Each script generates a cleaned dataset with standardized columns: source, flow_m3_per_day, size_PE, ch4_kg_per_hr, has_ad, reported_biogas_production, and biogas_production_kgCH4_per_hr
    - Script then combines all measurement datasets and sames them to 02_clean_data > measurement_data.csv 
    - Load and clean data from El Abbadi, Feng et al., 2025. Calls on functions in a_my_utilities.py to load all facilities, those with CHP, and those with AD 

3. c_plotting_functions.py: contains functions used to generate plots
4. calc_digester_costs.py - script for calculating cost of a new AD. I don't think I used this, it might have been playing around. Delete?  
5. chini_stats.py - scrit to load Chini dataset and cache stats 
6. d_clean_chini_data.py - clean Chini dataset (maybe rename so that this is before chini_stats.py in alpha order?)
7. e_compare_biogas_methods.py - compare Chini and Tarallo biogas methods. Not used in paper, was for internal checking 
8. Fig1_combined.py: generates Figure 1 
9. Fig2_combined.py: generates Figure 2
10. Fig4_economic_analysis: generates Fig4
11. Fig5_ad_chp_facilities.py: makes map of facilities with AD and CHP based on El Abbadi, Feng et al., 2025. (Not in main manuscript anymore)
12. FigS1_sources.py: generates Figures S1 
13. Paper_Calcs_MonteCarlo.py: script for all Monte Carlo calculations in the main manuscript 
14. Paper_Calculations: runs all calculations in the main manuscript 
15. Presentation_Figs.py: generates versions of the script used in presentations 
16. requirements.txt: requirements file 