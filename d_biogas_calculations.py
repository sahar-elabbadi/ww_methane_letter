#%%
# Imports
from pathlib import Path
import pandas as pd
from a_my_utilities import BIOGAS_FRACTION_CH4, METHANE_KG_PER_SCF, convert_to_scf, mgd_to_m3_per_day

# Load utility data from Chini et al 
# File saved in "01_raw_data", "chini-biogas", "chini_for_coding.csv"

chini_data_path = Path("01_raw_data", "chini-biogas", "chini_for_coding.csv")
chini_data = pd.read_csv(chini_data_path)

chini_data['flow_m3_per_day'] = chini_data.apply(lambda row: mgd_to_m3_per_day(row['facility_size_MGD']), axis=1)
chini_data['biogas_gen_scf'] = chini_data.apply(lambda row: convert_to_scf(row['biogas_gen_value'], row['biogas_gen_units']), axis=1)
chini_data['methane_gen_kgh'] = chini_data['biogas_gen_scf'] * METHANE_KG_PER_SCF * BIOGAS_FRACTION_CH4 / 366 / 24 # Convert from biogas scf generated in 2012 to kg/hr

#%%

# Remove facilities with outlier data 

remove_list = ['Stickney', # Stickney biogas consumption listed is too low to be total production (maybe rest is flared and not included in data facility provided)
               'Back River WWTP', # Baltimore, Black River facility has implausibly high biogas production for a facility of its size. 
               'EBMUD Main WWTP', # Flow data much lower than expected give our knowledge of this facility
               'TE Maxson WWTP', # Biogas production is very high - facility may be accepting additional high-strenght waste streams
               'Toledo Bay View', # Biogas production is very high - facility may be accepting additional high-strenght waste stream
               ]

# Keep only rows where facility_name is NOT in the list
chini_data = chini_data[~chini_data['facility_name'].isin(remove_list)]

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- 1. Clean data ---
mask = (
    chini_data[['flow_m3_per_day', 'methane_gen_kgh']].notna().all(axis=1) &
    np.isfinite(chini_data[['flow_m3_per_day', 'methane_gen_kgh']]).all(axis=1)
)
chini_clean = chini_data.loc[mask].copy()

X = chini_clean[['flow_m3_per_day']]
y = chini_clean['methane_gen_kgh']

# --- 2. Fit through-origin model ---
model = LinearRegression(fit_intercept=False)
model.fit(X, y)
y_pred = model.predict(X)

# Excel-style through-origin R²
ss_res = np.sum((y - y_pred)**2)
ss_tot_origin = np.sum(y**2)
r2_origin = 1 - ss_res / ss_tot_origin

print(f"Slope: {model.coef_[0]:.6f}")
print(f"Through-origin R²: {r2_origin:.6f}")

# --- 3. Plot ---
plt.figure(figsize=(6, 4))
plt.scatter(X, y, color="blue", alpha=0.6, label="Data")

# Regression line through origin
x_vals = np.linspace(0, X.max()[0], 100)
y_vals = model.coef_[0] * x_vals
plt.plot(x_vals, y_vals, color="red",
         label=f"y = {model.coef_[0]:.2f}x\nR² = {r2_origin:.3f}")

# Force axes to start at origin
plt.xlim(0, None)
plt.ylim(0, None)

plt.xlabel("Flow (m³/day)")
plt.ylabel("Methane generation (kg/h)")
plt.title("Methane vs. Flow (Through-Origin Regression)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
