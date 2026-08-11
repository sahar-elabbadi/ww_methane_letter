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

# Load utility data from Chini et al 
# File saved in "01_raw_data", "chini-biogas", "chini_for_coding.csv"
chini_data_path = pathlib.Path("01_raw_data", "chini-biogas", "chini_for_coding.csv")
chini_data = pd.read_csv(chini_data_path)

# Unit conversions 
chini_data['flow_m3_per_day'] = chini_data.apply(lambda row: mgd_to_m3_per_day(row['facility_size_MGD']), axis=1)
chini_data['biogas_gen_scf'] = chini_data.apply(lambda row: convert_to_scf(row['biogas_gen_value'], row['biogas_gen_units']), axis=1)

# Convert from biogas scf generated in 2012 to kg CH4/hr
chini_data['methane_gen_kgh'] = chini_data['biogas_gen_scf'] * METHANE_KG_PER_SCF * BIOGAS_FRACTION_CH4 / 366 / 24 

# Remove facilities with outlier data. Rationales provided in comments and discussed in main text. 
remove_list = ['Stickney', # Chicago, Stickney facility: biogas consumption listed is too low to be total production (maybe rest is flared and not included in data facility provided)
               'Back River WWTP', # Baltimore, Black River facility: has implausibly high biogas production for a facility of its size. 
               'EBMUD Main WWTP', # Oakland, EBMUD: Flow data much lower than expected give our knowledge of this facility
               'TE Maxson WWTP',  # Memphis TE, TE MAxon: Biogas production is very high - facility may be accepting additional high-strenght waste streams
               'Toledo Bay View', # Toledo, OH: Biogas production is very high - facility may be accepting additional high-strenght waste stream
               ]

# Keep only rows where facility_name is NOT in the list
chini_data = chini_data[~chini_data['facility_name'].isin(remove_list)]

# Drop rows where flow_m3_per_day or methane_gen_kgh is NaN or negative
chini_data = chini_data[(chini_data['flow_m3_per_day'] > 0) & (chini_data['methane_gen_kgh'] > 0)]

chini_data.to_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"), index=False)