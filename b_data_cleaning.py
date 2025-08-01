# %% Load data for CHP facilities 

# Setup 
import pandas as pd 
import matplotlib.pyplot as plt
import pathlib
from a_my_utilities import mgd_to_m3_per_day, g_per_s_to_kg_per_hour

#%% 

# Data cleaning for Moore et al, 2023

def clean_moore2023_data(
    filepath=pathlib.PurePath("01_raw_data", "Moore2023_SI-data.xlsx" ),
    flow_sheet='tableS5_bod_flow',
    measurement_sheet='tableS6_measurements',
    location_column='site',
    flow_cols=('flow_mgd', 'flow_mgd_uncertainty'),
    measurement_cols=(
        'ch4_g_per_s', 'ch4_g_per_s_uncertainty', 'detection_limit_g_per_s',
        'no_of_transects', 'pressure_mb', 'temp_C', 'wind_direction_degrees',
        'true_plume_direct_degrees1', 'u2_m_per_s'
    ),
    flow_column='flow_m3_per_day',
    measurement_column='ch4_kg_per_hr'
):
    # Load Excel data
    excel_path = filepath
    flow_data = pd.read_excel(excel_path, sheet_name=flow_sheet)
    meas_data = pd.read_excel(excel_path, sheet_name=measurement_sheet)
    
    # Clean column names
    flow_data.columns = flow_data.columns.str.strip()
    meas_data.columns = meas_data.columns.str.strip()

    # Coerce numeric
    flow_data[list(flow_cols)] = flow_data[list(flow_cols)].apply(pd.to_numeric, errors='coerce')
    meas_data[list(measurement_cols)] = meas_data[list(measurement_cols)].apply(pd.to_numeric, errors='coerce')

    # Unit conversions
    flow_data[flow_column] = flow_data['flow_mgd'].apply(mgd_to_m3_per_day)
    meas_data[measurement_column] = meas_data['ch4_g_per_s'].apply(g_per_s_to_kg_per_hour)

    # Select and merge
    df_meas_selected = meas_data[[location_column, measurement_column]]
    df_flow_selected = flow_data[[location_column, flow_column]]
    
    merged = pd.merge(df_meas_selected, df_flow_selected, on=location_column, how='inner')

    # Add 'source' column to Moore data
    merged['source'] = "Moore et al., 2023"

    reorder_merged = merged[['source', flow_column, measurement_column]]
    
    return reorder_merged


moore_data = clean_moore2023_data()


#%% 

# Data cleaning for Song et al, 2023

def clean_song_data(
    filepath=pathlib.PurePath("01_raw_data", "Song2023_raw-data.xlsx"),
    numeric_cols=('flow_m3_per_day', 'ch4_kg_per_day')
):
    # Load Excel data
    excel_path = filepath
    df = pd.read_excel(excel_path)
    
    # Clean column names
    df.columns = df.columns.str.strip()

    # Coerce numeric
    df[list(numeric_cols)] = df[list(numeric_cols)].apply(pd.to_numeric, errors='coerce')

    # Unit conversion: kg/day → kg/hour
    df['ch4_kg_per_hr'] = df['ch4_kg_per_day'] / 24

    save_cols = ['source', 'flow_m3_per_day', 'ch4_kg_per_hr']

    return df[save_cols]

song_data = clean_song_data()

#%%

# Combine Moore data and Song data 

# Combine into one long DataFrame
measurement_data = pd.concat([moore_data, song_data], ignore_index=True)
save_path = pathlib.Path("02_clean_data", "measurement_data.csv")
measurement_data.to_csv(save_path, index=False)

