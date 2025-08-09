# %% Load data for CHP facilities 

# Setup 
import pandas as pd 
import matplotlib.pyplot as plt
import pathlib
from a_my_utilities import mgd_to_m3_per_day, g_per_s_to_kg_per_hour

#%% 
####################### Data cleaning for Moore et al., 2023 ######################

def clean_moore2023_data():
    # File and sheet names
    filepath = pathlib.PurePath("01_raw_data", "Moore2023_SI-data.xlsx")
    flow_sheet = 'tableS5_bod_flow'
    measurement_sheet = 'tableS6_measurements'
    
    # Columns of interest
    location_column = 'site'
    flow_cols_numeric = ['flow_mgd', 'flow_mgd_uncertainty']
    ad_column = 'has_ad'
    measurement_cols = [
        'ch4_g_per_s', 'ch4_g_per_s_uncertainty', 'detection_limit_g_per_s',
        'no_of_transects', 'pressure_mb', 'temp_C', 'wind_direction_degrees',
        'true_plume_direct_degrees1', 'u2_m_per_s'
    ]
    flow_column = 'flow_m3_per_day'
    measurement_column = 'ch4_kg_per_hr'

    # Load Excel data
    flow_data = pd.read_excel(filepath, sheet_name=flow_sheet)
    meas_data = pd.read_excel(filepath, sheet_name=measurement_sheet)
    
    # Clean column names
    flow_data.columns = flow_data.columns.str.strip()
    meas_data.columns = meas_data.columns.str.strip()

    # Coerce numeric columns
    flow_data[flow_cols_numeric] = flow_data[flow_cols_numeric].apply(pd.to_numeric, errors='coerce')
    meas_data[measurement_cols] = meas_data[measurement_cols].apply(pd.to_numeric, errors='coerce')

    # Keep categorical column as string
    flow_data[ad_column] = flow_data[ad_column].astype(str)

    # Unit conversions
    flow_data[flow_column] = flow_data['flow_mgd'].apply(mgd_to_m3_per_day)
    meas_data[measurement_column] = meas_data['ch4_g_per_s'].apply(g_per_s_to_kg_per_hour)

    # Select and merge
    df_meas_selected = meas_data[[location_column, measurement_column]]
    df_flow_selected = flow_data[[location_column, flow_column, ad_column]]
    
    merged = pd.merge(df_meas_selected, df_flow_selected, on=location_column, how='inner')

    # Add 'source' column to Moore data
    merged['source'] = "Moore et al., 2023"

    # Add column for 'reported_biogas_production' 
    merged['reported_biogas_production'] = "no"


    # Reorder columns
    reorder_merged = merged[['source', flow_column, measurement_column, ad_column, 'reported_biogas_production']]
    
    return reorder_merged

# Call the function
moore_data = clean_moore2023_data()


#%% 

####################### Data cleaning for Song et al, 2023 ######################

def clean_song_data():
    # File path
    filepath = pathlib.PurePath("01_raw_data", "Song2023_raw-data.xlsx")
    
    # Columns
    numeric_cols = ['flow_m3_per_day', 'ch4_kg_per_day']
    ad_column = 'has_ad'
    flow_column = 'flow_m3_per_day'
    measurement_column = 'ch4_kg_per_hr'
    location_column = 'site'  # If needed for future consistency

    # Load data
    df = pd.read_excel(filepath)
    
    # Clean column names
    df.columns = df.columns.str.strip()

    # Coerce numeric values
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Ensure 'has_ad' remains a string
    df[ad_column] = df[ad_column].astype(str)

    # Unit conversion: kg/day → kg/hour
    df[measurement_column] = df['ch4_kg_per_day'] / 24

    # Add column for 'reported_biogas_production' 
    df['reported_biogas_production'] = "no"

    save_cols = ['source', 'flow_m3_per_day', 'ch4_kg_per_hr', 'has_ad', 'reported_biogas_production']

    return df[save_cols]

song_data = clean_song_data()

# %%
####################### Data cleaning for Fredenslund et al., 2023 ######################

import pandas as pd 
import matplotlib.pyplot as plt
import pathlib

import re
import pandas as pd
from pathlib import Path

def load_fredenslund_data(excel_path: Path, last_row_excel: int = 72, sheet: str = "Data") -> pd.DataFrame:
    """Load Fredenslund2023 data up to a specified last row, with units in column names standardized."""
    header_row_idx = 1
    units_row_idx = 2

    # Load raw for headers/units
    raw = pd.read_excel(excel_path, sheet_name=sheet, header=None, nrows=last_row_excel)
    headers = raw.iloc[header_row_idx].astype(str).str.strip().tolist()
    units_raw = raw.iloc[units_row_idx].astype(str).tolist()

    def normalize_unit(u: str) -> str:
        if not isinstance(u, str):
            return ""
        s = u.strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return ""
        s_norm = s.replace("\u2212", "-")
        s_norm = re.sub(r"\s+", "", s_norm).lower()
        if re.fullmatch(r"kgch4h-?1", s_norm):
            return "kgCH4_per_hour"
        if "%" in s:
            return "percent"
        return s

    units_norm = [normalize_unit(u) for u in units_raw]
    headers_with_units = [f"{h} ({u})" if u else h for h, u in zip(headers, units_norm)]

    # Load data after the units row
    data_rows_to_read = last_row_excel - (units_row_idx + 1)
    df = pd.read_excel(excel_path, sheet_name=sheet, header=None, skiprows=3, nrows=data_rows_to_read)
    df.columns = headers_with_units

    # snake_case
    def to_snake(name: str) -> str:
        name = name.strip().lower()
        name = name.replace("#", "num ").replace("%", "percent")
        name = re.sub(r"[^\w\s]", "", name)
        name = re.sub(r"\s+", "_", name)
        return name

    df.columns = [to_snake(c) for c in df.columns]

    return df


def clean_fredenslund_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for wastewater plants and return a simplified dataframe with specified columns."""
    filtered = df[df["plant_type"] == "Wastewater"].copy()

    # Build the new dataframe
    result = pd.DataFrame({
        "source": "Fredenslund et al., 2023",
        "flow_m3_per_day": pd.NA,  # No flow data provided
        "ch4_kg_per_hr": filtered["total_methane_emission_kgch4_per_hour"],
        "has_ad": 'yes', 
        "reported_biogas_production": 'yes',
    })

    return result


excel_path = pathlib.Path("01_raw_data", "Fredenslund2023_SI-data.xlsx")
raw_df = load_fredenslund_data(excel_path)
fredenslund_data = clean_fredenslund_data(raw_df)
# %%
#%%

####################### Combine datasets ######################

# Combine into one long DataFrame
measurement_data = pd.concat([moore_data, song_data, fredenslund_data], ignore_index=True)
save_path = pathlib.Path("02_clean_data", "measurement_data.csv")
measurement_data.to_csv(save_path, index=False)
