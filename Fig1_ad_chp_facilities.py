
# %% Load data for CHP facilities 

import pandas as pd 
import matplotlib.pyplot as plt
import pathlib

from a_my_utilities import load_chp_facilities, load_ad_facilities, mgd_to_m3_per_day

# Load data 

data_path = pathlib.Path("01_raw_data", "supplementary_database_C.xlsx")
wwtp_data = pd.read_excel(data_path)

chp_data = load_chp_facilities()
ad_data = load_ad_facilities()

# Data concversion from MGD to m3/day
chp_data['flow_m3_per_day'] = chp_data['flow_mgd'].apply(mgd_to_m3_per_day)
ad_data['flow_m3_per_day'] = ad_data['flow_mgd'].apply(mgd_to_m3_per_day)


## NEXT
# Plot chp facilities and ad facilities on map, flow rate in mm3 /day 
# AD facilities in orange
# CHP facilities in green

# For Brian: 
# chp_data.sort_values(by='flow_mgd', inplace=True, ascending=False)
# chp_data.to_csv(pathlib.PurePath("02_clean_data", "chp_facilities.csv"), index=False)
