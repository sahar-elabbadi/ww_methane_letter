#%% 
import pandas as pd
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from a_my_utilities import set_chini_dataset, calc_biogas_production_rate, load_ch4_emissions_data

chini_data = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))  # or csv

set_chini_dataset(
    chini_data,
    x_col="flow_m3_per_day",
    y_col="methane_gen_kgh",
    drop_negative=True
)
# Load the measurement dataset 

measurement_data = load_ch4_emissions_data()

# Calclate biogas production using both methods
measurement_data['biogas_chini']   = calc_biogas_production_rate(measurement_data['flow_m3_per_day'], "chini_data")   # kg CH4/h
measurement_data['biogas_tarallo'] = calc_biogas_production_rate(measurement_data['flow_m3_per_day'], "tarallo_model") # kg CH4/h


# Plot biogas production 
# plot measurement_data['biogas_chini'] vs measurement_data['biogas_tarallo']

sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
plt.scatter(measurement_data['biogas_tarallo'], measurement_data['biogas_chini'])

plt.xlim(0, max(measurement_data['biogas_chini'].max(), measurement_data['biogas_tarallo'].max()) * 1.05)
plt.ylim(0, max(measurement_data['biogas_chini'].max(), measurement_data['biogas_tarallo'].max()) * 1.05)
plt.xlabel('Tarallo Biogas Production Estimate (kg/h)')
plt.ylabel('Chini Biogas Production Estimate (kg/h)')
plt.title('Compare Chini and Tarallo Biogas Production Estimates for Facilities with Measurement Data')
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()



# %%
