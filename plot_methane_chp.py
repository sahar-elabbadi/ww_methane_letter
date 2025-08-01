
# %% Load data for CHP facilities 

import pandas as pd 
import matplotlib.pyplot as plt
import pathlib

from a_my_utilities import load_chp_facilities, load_and_clean_facility_data

# Load data 

data_path = pathlib.Path("01_raw_data", "supplementary_database_C.xlsx")
wwtp_data = pd.read_excel(data_path)

chp_data = load_chp_facilities()

# %%

# NEXT: 
# Plot data for facilities with CHP - how much methane are they producing with X% leak rates? 

# 
