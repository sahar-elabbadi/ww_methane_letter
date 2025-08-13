

#%%
import pandas as pd
import numpy as np
import pathlib

def fredenslund_append_wastewater_pe(fredenslund_df):
    """
    Append wastewater_PE values from the Wechselberger2025_SI-data.csv dataset
    to a given fredenslund dataset, matching rows by plant_id
    
    Parameters
    ----------
    fredenslund_df : pd.DataFrame
        The Fredenslund dataset (will not be modified in place).
        
    Returns
    -------
    pd.DataFrame
        Copy of fredenslund_df with an added columns  'CH4_emission_rate_in_kg_per_hour_mean', 'wastewater_PE' from Wechselberger2025
    """
    
    # --- Load and clean Wechselberger dataset ---
    file_path = pathlib.Path("01_raw_data", "Wechselberger2025_SI-data.csv")
    df = pd.read_csv(file_path)
    
    def clean_numeric(series):
        return (
            series.astype(str)
            .str.replace('.', '', regex=False)   # remove thousand separators
            .str.replace(',', '.', regex=False)  # fix decimal commas
            .apply(lambda x: pd.to_numeric(x, errors='coerce'))  # to float
        )
    
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            continue
        if df[col].astype(str).str.match(r"^[\d\.,]+$").any():
            df[col] = clean_numeric(df[col])
    
    wechselberger_ww = df[df['data_source_citation'] == "Fredenslund et al., 2023"].copy()
    
    if 'wastewater_PE' not in wechselberger_ww.columns:
        raise KeyError("Column 'wastewater_PE' not found in wechselberger_ww")
    
    wechselberger_ww['wastewater_PE'] = pd.to_numeric(wechselberger_ww['wastewater_PE'], errors='coerce')

    merged = pd.merge(
        fredenslund_df, wechselberger_ww[['plant_id', 'CH4_emission_rate_in_kg_per_hour_mean', 'wastewater_PE']],
        left_on='plant_id', 
        right_on='plant_id',
        how='inner', # Inner merge to keep only matching rows
    )
    return merged 


from b_data_cleaning import load_fredenslund_data

excel_path = pathlib.Path("01_raw_data", "Fredenslund2023_SI-data.xlsx")
raw_fredenslund = load_fredenslund_data(excel_path)
fredenslund_ww = raw_fredenslund[raw_fredenslund["plant_type"] == "Wastewater"].copy()

fredenslund_with_pe = fredenslund_append_wastewater_pe(fredenslund_ww)

print(fredenslund_with_pe[['total_methane_emission_kgch4_per_hour', 'wastewater_PE']].head())

# %%
