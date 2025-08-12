# #%%
# from b_data_cleaning import load_fredenslund_data
# import pandas as pd
# import pathlib
# # 1) Load the CSV file
# file_path = pathlib.Path("01_raw_data", "Wechselberger2025_SI-data.csv")  # Update path if needed
# df = pd.read_csv(file_path)

# # Function to clean numeric strings
# def clean_numeric(series):
#     return (
#         series.astype(str)
#         .str.replace('.', '', regex=False)   # remove thousand separators
#         .str.replace(',', '.', regex=False)  # fix decimal commas
#         .apply(lambda x: pd.to_numeric(x, errors='coerce'))  # convert to float
#     )

# # Apply to all columns that contain numbers (either actual numbers or numeric-looking strings)
# for col in df.columns:
#     if df[col].dtype in ['float64', 'int64']:  # already numeric
#         continue
#     # Check if any value looks numeric (digits, . , ,)
#     if df[col].astype(str).str.match(r"^[\d\.,]+$").any():
#         df[col] = clean_numeric(df[col])

# # Filter again
# wechselberger_ww = df[df['data_source_citation'] == "Fredenslund et al., 2023"].copy()


# excel_path = pathlib.Path("01_raw_data", "Fredenslund2023_SI-data.xlsx")
# raw_fredenslund = load_fredenslund_data(excel_path)
# fredenslund_ww = raw_fredenslund[raw_fredenslund["plant_type"] == "Wastewater"].copy()

# import numpy as np
# import pandas as pd

# # Tolerance in kg CH4/hr
# tolerance = 0.05

# # --- keep indices in matches during matching ---
# matches = []
# for _, row in wechselberger_ww.iterrows():
#     diffs = np.abs(
#         fredenslund_ww['total_methane_emission_kgch4_per_hour'] 
#         - row['CH4_emission_rate_in_kg_per_hour_mean']
#     )
#     min_diff_idx = diffs.idxmin()
#     min_diff = diffs[min_diff_idx]
#     if min_diff <= tolerance:
#         matches.append({
#             'wechselberger_idx': row.name,  # <-- index in wechselberger_ww
#             'fredenslund_idx': min_diff_idx,  # <-- index in fredenslund_ww
#             'CH4_filtered': row['CH4_emission_rate_in_kg_per_hour_mean'],
#             'CH4_fredenslund': fredenslund_ww.loc[min_diff_idx, 'total_methane_emission_kgch4_per_hour'],
#             'diff': min_diff
#         })

# matches_df = pd.DataFrame(matches)

# # --- if multiple wechselberger rows map to the same fredenslund row, keep the closest ---
# matches_df = matches_df.sort_values('diff').drop_duplicates('fredenslund_idx', keep='first')

# # --- sanity check: make sure wastewater_PE exists and is numeric ---
# if 'wastewater_PE' not in wechselberger_ww.columns:
#     raise KeyError("Column 'wastewater_PE' not found in wechselberger_ww")

# # If needed, coerce to numeric (won't hurt if it's already numeric)
# wechselberger_ww['wastewater_PE'] = pd.to_numeric(wechselberger_ww['wastewater_PE'], errors='coerce')

# # --- build a Series aligned to fredenslund_ww's index ---
# pe_series = pd.Series(index=fredenslund_ww.index, dtype='float64')

# # Map wastewater_PE from matched wechselberger rows onto fredenslund indices
# pe_series.loc[matches_df['fredenslund_idx'].values] = (
#     wechselberger_ww.loc[matches_df['wechselberger_idx'].values, 'wastewater_PE'].values
# )

# # --- append to fredenslund_ww ---
# fredenslund_ww = fredenslund_ww.copy()
# fredenslund_ww['wastewater_PE_from_wechselberger'] = pe_series

# # --- quick diagnostics ---
# n_matched = matches_df.shape[0]
# n_total_subset = wechselberger_ww.shape[0]
# n_with_pe = fredenslund_ww['wastewater_PE_from_wechselberger'].notna().sum()

# print(f"Matched {n_matched} methane points (of {n_total_subset} in wechselberger_ww) within tolerance={tolerance}.")
# print(f"Transferred wastewater_PE onto {n_with_pe} fredenslund_ww rows.")
# print(fredenslund_ww[['total_methane_emission_kgch4_per_hour', 'wastewater_PE_from_wechselberger']].head())

# check_cols = fredenslund_ww[['total_methane_emission_kgch4_per_hour', 'wastewater_PE_from_wechselberger']]

#%%
# # %%
import pandas as pd
import numpy as np
import pathlib

def append_wastewater_pe(fredenslund_df, tolerance=0.05):
    """
    Append wastewater_PE values from the Wechselberger2025_SI-data.csv dataset
    to a given fredenslund dataset, matching rows by methane emission rate 
    within a specified tolerance.
    
    Parameters
    ----------
    fredenslund_df : pd.DataFrame
        The Fredenslund dataset (will not be modified in place).
    tolerance : float, optional
        Maximum allowable absolute difference (kg CH4/hr) for matching. Default = 0.05.
        
    Returns
    -------
    pd.DataFrame
        Copy of fredenslund_df with an added column `wastewater_PE_from_wechselberger`.
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
    
    # --- Match rows ---
    matches = []
    for _, row in wechselberger_ww.iterrows():
        diffs = np.abs(
            fredenslund_df['total_methane_emission_kgch4_per_hour'] 
            - row['CH4_emission_rate_in_kg_per_hour_mean']
        )
        min_diff_idx = diffs.idxmin()
        min_diff = diffs[min_diff_idx]
        if min_diff <= tolerance:
            matches.append({
                'wechselberger_idx': row.name,
                'fredenslund_idx': min_diff_idx,
                'diff': min_diff
            })
    
    matches_df = pd.DataFrame(matches)
    matches_df = matches_df.sort_values('diff').drop_duplicates('fredenslund_idx', keep='first')
    
    # --- Map wastewater_PE to fredenslund_df ---
    pe_series = pd.Series(index=fredenslund_df.index, dtype='float64')
    pe_series.loc[matches_df['fredenslund_idx'].values] = (
        wechselberger_ww.loc[matches_df['wechselberger_idx'].values, 'wastewater_PE'].values
    )
    
    fredenslund_out = fredenslund_df.copy()
    fredenslund_out['wastewater_PE_from_wechselberger'] = pe_series
    
    # --- Diagnostics ---
    n_matched = matches_df.shape[0]
    n_total_subset = wechselberger_ww.shape[0]
    n_with_pe = fredenslund_out['wastewater_PE_from_wechselberger'].notna().sum()
    
    print(f"Matched {n_matched} methane points (of {n_total_subset} in wechselberger_ww) within tolerance={tolerance}.")
    print(f"Transferred wastewater_PE onto {n_with_pe} fredenslund rows.")
    
    return fredenslund_out


from b_data_cleaning import load_fredenslund_data

excel_path = pathlib.Path("01_raw_data", "Fredenslund2023_SI-data.xlsx")
raw_fredenslund = load_fredenslund_data(excel_path)
fredenslund_ww = raw_fredenslund[raw_fredenslund["plant_type"] == "Wastewater"].copy()

fredenslund_with_pe = append_wastewater_pe(fredenslund_ww, tolerance=0.05)

fredenslund_with_pe[['total_methane_emission_kgch4_per_hour', 'wastewater_PE_from_wechselberger']].head()
