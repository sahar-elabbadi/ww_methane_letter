#%% 
import pandas as pd 
import pathlib
import numpy as np
from functools import lru_cache
from scipy import stats



import pandas as pd


####### GLOBAL CONSTANTS ########################################

BIOGAS_FRACTION_CH4 = 0.65  # Assume 65% CH4 in biogas. Source: Metcalf & Eddy, page 1520
METHANE_SCF_PER_THERM = 100
METHANE_MMBTU_PER_THERM = 0.1
METHANE_KG_PER_SCF = 0.019176  # kg of methane per scf (standard conditions, 15C, 1 atm) 
METHANE_KG_PER_NM3 = 0.7168 # kg of CH4 per normal m3 (Normal conditions: 0 C , 1 atm) 
M3_PER_GAL = 0.003785411784
MMBTU_TO_MJ   = 1055.056    # 1 MMBtu = 1,055.056 MJ  (EIA/DOE)
THERM_TO_MJ   = 105.5056    # 1 therm = 105.5056 MJ   (EIA/DOE)
MSCF_TO_MMBTU = 1.038       # ≈ avg 2023 heat content of NG ~1,038 Btu/ft³ → 1.038 MMBtu per Mcf (EIA)
MSCF_TO_MJ    = MSCF_TO_MMBTU * MMBTU_TO_MJ  # ≈ 1,094.093 MJ per Mscf


# Methane lower heating value - when water is not condensed 
kJ_per_mol = 802.3   # kJ/mol, reported in Harrison et al., 2010
MW_methane = 16.04    # g/mol = 0.01604 kg/mol

# Convert kJ/mol to MJ/kg
METHANE_MJ_PER_KG = (kJ_per_mol / 1000) / (MW_methane / 1000) # Result: 50 MJ / kg

####### Data Loading ########################################

######## EIA DATA #######

# Electricity data 
def load_eia_data(sector: str, year: int):
    """
    Loads EIA annual average electricity price data for a specific sector and filters by year.

    Parameters:
    sector (str): One of ["RESIDENTIAL", "COMMERCIAL", "INDUSTRIAL", "TRANSPORTATION", "TOTAL"]
    year (int): A year between 2010 and 2023.

    Returns:
    pd.DataFrame: The cleaned DataFrame for the specified sector and year.
    """
    # Define file path
    file_path = pathlib.PurePath('01_raw_data', 'EIA_annual_retail_price.xlsx')
    sheet_name = "Total Electric Industry"

    # Read the sheet into a dataframe
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=[0, 1])

    # Define sector list
    sectors = ["RESIDENTIAL", "COMMERCIAL", "INDUSTRIAL", "TRANSPORTATION", "TOTAL"]

    # Ensure the requested sector is valid
    if sector not in sectors:
        raise ValueError(f"Invalid sector. Choose from {sectors}")

    # Ensure the year is within the valid range
    if not (2010 <= year <= 2023):
        raise ValueError("Year must be between 2010 and 2023.")

    # Extract the year and state columns
    year_state_cols = df.iloc[:, :2]
    year_state_cols.columns = ["Year", "State"]

    # Check if sector exists in DataFrame
    if sector not in df.columns.get_level_values(0):
        raise ValueError(f"Sector '{sector}' not found in the data!")

    # Extract sector-specific data
    sector_df = pd.concat([year_state_cols, df.loc[:, sector]], axis=1)

    # Add a "Sector" column
    sector_df.insert(1, "Sector", sector)

    # Drop the first row (metadata)
    sector_df = sector_df.iloc[1:].reset_index(drop=True)

    # Rename columns safely
    rename_map = {
        "Revenues": "Revenues (thousand USD)",
        "Sales": "Sales (MWh)",
        "Price": "Price (Cents/kWh)"
    }

    # Rename only if the column exists
    sector_df = sector_df.rename(columns={col: rename_map[col] for col in rename_map if col in sector_df.columns})

    # Convert "Year" column to integers and filter by the specified year
    sector_df["Year"] = sector_df["Year"].astype(int)
    sector_df = sector_df[sector_df["Year"] == year]

    # Drop the US average row (keep only actual states/territories)
    sector_df = sector_df[sector_df["State"] != "US"]

    return sector_df

# Convert DataFrame column to dictionary for use in other scripts

# Load EIA data and process 
eia_industrial_tariffs_2023_df = (
    load_eia_data(sector='INDUSTRIAL', year=2023)
    .set_index('State')  # Ensure "State" is the index
    [['Price (Cents/kWh)']]  # Double brackets keep it as a DataFrame
    .div(100)  # Convert from cents to dollars
)

eia_industrial_tariffs_2023_df = eia_industrial_tariffs_2023_df.rename(columns={'Price (Cents/kWh)': 'Price ($/kWh)'})
eia_industrial_tariffs_2023_df.to_csv(pathlib.Path("02_clean_data", "eia_industrial_tariffs_2023.csv"))

eia_industrial_tariffs_2023 = eia_industrial_tariffs_2023_df['Price ($/kWh)'].to_dict() # for use elsewhere in script

def make_eia_industrial_ng_2023():
    """
    1) Load EIA data for natural gas prices
    2) Convert from $ / Mscf (thousand scf) to $/MJ
    3) Fill DC from $/therm -> $/MJ (no $/Mscf for DC)
    4) Save CSV to 02_clean_data/eia_industrial_tariffs_natural_gas_2023.csv
    5) Return dataframe
    """
    # Manual insert for DC electricity value based on their provider 
    dc_rate_per_therm = 0.40 # According to rates provided by Washington Gas, saved in 01_raw_data > DC_natural_gas_prices.pdf

    # 1) Load the data
    df = pd.read_excel(pathlib.Path('01_raw_data', 'EIA_natural_gas_prices.xlsx'), sheet_name='Clean - For Code', usecols=["Year", "State", "Price ($/Mscf)"])

    # 2) Convert to $/MJ
    df["Price ($/MJ)"] = pd.to_numeric(df["Price ($/Mscf)"], errors="coerce") / MSCF_TO_MJ

    # 3) Insert manually selected rate for DC into the main dataframe
    dc_mj = dc_rate_per_therm / THERM_TO_MJ # Unit conversion to $ / MJ 
    is_dc = df["State"].astype(str).str.upper().isin(["DC", "DISTRICT OF COLUMBIA"]) # Find row with DC, check for variations in naming
    if is_dc.any():
        df.loc[is_dc, "Price ($/MJ)"] = df.loc[is_dc, "Price ($/MJ)"].fillna(dc_mj)
    else:
        # Match rest of dataframe naming convention (abbreviation vs full name) 
        use_abbr = df["State"].astype(str).map(len).eq(2).any()
        dc_name = "DC" if use_abbr else "District of Columbia"
        df = pd.concat(
            [df, pd.DataFrame({"Year": [2023], "State": [dc_name], "Price ($/Mscf)": [pd.NA], "Price ($/MJ)": [dc_mj]})],
            ignore_index=True,
        )

    # 4) Save the spreadsheet in clean data directory 
    out_path = pathlib.Path("02_clean_data", "eia_industrial_tariffs_natural_gas_2023.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eia_industrial_tariffs_2023_df = (
        df.set_index("State")[["Price ($/MJ)"]].sort_index()
    )
    eia_industrial_tariffs_2023_df.to_csv(out_path)

    # 5) Dict for use elsewhere as needed
    # eia_industrial_tariffs_2023 = eia_industrial_tariffs_2023_df["Price ($/MJ)"].to_dict()

    return eia_industrial_tariffs_2023_df

# Load EIA natural gas data and process 
eia_industrial_natural_gas_2023_df = make_eia_industrial_ng_2023()

# Make dictionary for use elsewhere as needed 
eia_industrial_natural_gas_2023 = eia_industrial_natural_gas_2023_df['Price ($/MJ)'].to_dict() # for use elsewhere in script

####### FACILITY DATA FROM EL ABBADI, FENG ET AL 2025 #########

def load_and_clean_facility_data(filepath: str) -> pd.DataFrame:
    """
    Load and clean facility-level emissions data from an Excel file.

    Parameters:
    - filepath: str
        Path to the Excel file.

    Returns:
    - pd.DataFrame
        Cleaned DataFrame with machine-readable column names, proper data types,
        and parsed treatment train column.
    """
    
    # Define column renaming map
    column_renames = {
        'CWNS code': 'cwns_code',
        'facility': 'facility',
        'state': 'state',
        'city': 'city',
        'latitude': 'latitude',
        'longitude': 'longitude',
        'flow [MGD]': 'flow_mgd',
        'median CH4 [kg CO2-eq/day]': 'median_ch4_kgco2e_day',
        'median N2O [kg CO2-eq/day]': 'median_n2o_kgco2e_day',
        'median CO2 [kg CO2-eq/day]': 'median_co2_kgco2e_day',
        'median electricity [kg CO2-eq/day]': 'median_electricity_kgco2e_day',
        'median onsite natural gas [kg CO2-eq/day]': 'median_onsite_gas_kgco2e_day',
        'median upstream natural gas [kg CO2-eq/day]': 'median_upstream_gas_kgco2e_day',
        'median landfill CH4 [kg CO2-eq/day]': 'median_landfill_ch4_kgco2e_day',
        'median land application N2O [kg CO2-eq/day]': 'median_landapp_n2o_kgco2e_day',
        'median total emission [kg CO2-eq/day]': 'median_total_emission_kgco2e_day',
        'treatment train': 'treatment_train'
    }

    # Helper to parse treatment train list
    import numpy as np  # <- Must be global

    def parse_treatment_train(val):
        try:
            # Safely evaluate using only np and no builtins
            raw = eval(val, {"np": np, "__builtins__": {}})
            return [str(x) for x in raw]
        except Exception as e:
            print(f"Parse error: {e} on value {val}")
            return []


    # Load the Excel file
    df = pd.read_excel(filepath)

    # Rename columns
    df.rename(columns=column_renames, inplace=True)

    # Set proper data types
    df = df.astype({
        'cwns_code': 'float',
        'facility': 'string',
        'state': 'string',
        'city': 'string',
        'latitude': 'float',
        'longitude': 'float',
        'flow_mgd': 'float',
        'median_ch4_kgco2e_day': 'float',
        'median_n2o_kgco2e_day': 'float',
        'median_co2_kgco2e_day': 'float',
        'median_electricity_kgco2e_day': 'float',
        'median_onsite_gas_kgco2e_day': 'float',
        'median_upstream_gas_kgco2e_day': 'float',
        'median_landfill_ch4_kgco2e_day': 'float',
        'median_landapp_n2o_kgco2e_day': 'float',
        'median_total_emission_kgco2e_day': 'float'
    })

    # Clean treatment_train column
    df['treatment_train'] = df['treatment_train'].apply(parse_treatment_train)

    df['has_chp'] = np.where(df['treatment_train'].astype(str).str.contains('e', na=False), 'yes', 'no')
    df['has_ad']  = np.where(df['treatment_train'].astype(str).str.contains('1', na=False), 'yes', 'no')

    # Calculate flow in m3 / day 
    df['flow_m3_per_day'] = df['flow_mgd'] * M3_PER_GAL * 1e6

    # Add price of electricity based on EIA state average 
    df["electricity_cost"] = df["state"].map(eia_industrial_tariffs_2023)

    # Add price of natural gas based on EIA state average
    df["natural_gas_cost"] = df["state"].map(eia_industrial_natural_gas_2023)

    # Save files
    wwtp_save_path = pathlib.Path("02_clean_data", "wwtp_data.csv")
    df.to_csv(wwtp_save_path, index=False)

    return df

def load_all_facilities(): 
    # Load cleaned data for all facilities
    data_path = pathlib.Path("02_clean_data", "wwtp_data.csv")
    all_data = pd.read_csv(data_path)
    return all_data

def load_chp_facilities():
    # Load cleaned data for all facilities 
    wwtp_data = load_all_facilities()

    # Filter based on has_chp, save as copy 
    energy_recovery = wwtp_data[wwtp_data['has_chp']=='yes'].copy()
    return energy_recovery

def load_ad_facilities():
        # Load cleaned data for all facilities 
    wwtp_data = load_all_facilities()
        # Filter based on has_chp, save as copy 
    anaerobic_digestion = wwtp_data[wwtp_data['has_ad']=='yes'].copy()

    return anaerobic_digestion


def load_ch4_emissions_data(
    filepath=pathlib.PurePath("02_clean_data", "measurement_data.csv")  ,
    dtype_map={
        "source": str,
        "flow_m3_per_day": float,
        "ch4_kg_per_hr": float,
        'has_ad': str,
        'reported_biogas_production': str,
        'biogas_production_kgCH4_per_hr': float,
    }
):
    import pandas as pd

    df = pd.read_csv(filepath)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Ensure expected dtypes
    for col, col_type in dtype_map.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') if col_type == float else df[col].astype(col_type)

    # Reset index just in case
    df = df.reset_index(drop=True)

    return df

def load_ch4_emissions_with_ad_only():
    # Load full emissions data
    df = load_ch4_emissions_data()

    # Normalize 'has_ad' to lowercase in case there are variations like 'Yes', 'YES', etc.
    df['has_ad'] = df['has_ad'].str.strip().str.lower()

    # Filter to only rows with AD
    df_ad = df[df['has_ad'] == 'yes'].copy()

    return df_ad



####### Unit Conversions ########################################

# 1 US gallon = 0.003785411784 m³
# 1 MGD = 1e6 gallons/day → million m³/day = (1e6 * 0.003785411784) / 1e6 = 0.003785411784
# M3_PER_MG = 0.003785411784 # moved above
# mj_per_kg_ch4 = 50.4 # energy content of methane

def m3_per_mg(): 
    return M3_PER_GAL * 1e6 # Convert MG to m3 


def mgd_to_m3_per_day(mgd: float) -> float:
    """
    Convert MGD (Million Gallons per Day) to cubic meters per day.
    
    Parameters:
    - mgd: float
        Flow in Million Gallons per Day.
    
    Returns:
    - float
        Flow in cubic meters per day.
    """
    return mgd * 1e6 * M3_PER_GAL


def g_per_s_to_kg_per_hour(g_per_s):
    """
    Convert grams per second to kg per hour 
    """
    return g_per_s *3.6

def t_per_day_to_kg_per_hr(t_per_day):
    """
    Convert tonnes per day to kg per hour 
    """
    return t_per_day * (1000/24)

def t_per_yr_to_kg_per_hr(t_per_day):
    """
    Convert tonnes per day to kg per hour 
    """
    return t_per_day * (1000/24/365)

def convert_mj_to_kg_CH4(mj_ch4): 
    """
    Convert MJ of energy in methane to kg CH4
    """
    return mj_ch4 * (1/METHANE_MJ_PER_KG)


def mj_per_kg_CH4(): 
    return METHANE_MJ_PER_KG



def mj_per_kWh(): 
    return 3.6 # 3.6 MJ per kWh


def convert_to_scf(value, unit):
    """ Take in biogas production value and units
    Return value in standard cubic feet (scf) of biogas"""
    if pd.isna(value) or pd.isna(unit):
        return None
    unit = unit.lower()
    
    if unit == 'scf':
        return value
    elif unit == 'mscf':  # thousand scf
        return value * 1_000
    elif unit == 'mmscf':  # million scf
        return value * 1_000_000
    elif unit in ['therms', 'therm']:
        # Placeholder: insert actual conversion factor later
        return value * METHANE_SCF_PER_THERM / BIOGAS_FRACTION_CH4  
    elif unit in ['dtherms']: # dekatherms
        return value * 10 * METHANE_SCF_PER_THERM / BIOGAS_FRACTION_CH4
    elif unit == 'scfm':  # standard cubic feet per minute, calculated over 1 year
        return value * 60 * 24 * 366  # 366 days in 2012
    elif unit == 'mmbtu':
        # Placeholder: insert actual conversion factor later
        return value * (1/METHANE_MMBTU_PER_THERM) * METHANE_SCF_PER_THERM / BIOGAS_FRACTION_CH4  
    else:
        return None
    

def Nm3_per_year_to_kgCH4_per_hr(Nm3_per_year):
    """
    Convert normal cubic meters per year to kg CH4 per hour
    Normal conditions: 0 C , 1 atm
    """
    kgCH4_per_year = Nm3_per_year * METHANE_KG_PER_NM3 # Nm3 to kg CH4
    kgCH4_per_hr = kgCH4_per_year / (365*24) # year to hours
    return kgCH4_per_hr


####### Analysis Functions ########################################

####### SETUP FOR CHINI REGRESSION #######
# ----------------------------
# Internal registration state
# ----------------------------
_CHINI_DATA = {
    "df": None,
    "x_col": "flow_m3_per_day",
    "y_col": "methane_gen_kgh",
    "drop_negative": True,
}

def set_chini_dataset(df, x_col="flow_m3_per_day", y_col="methane_gen_kgh", drop_negative=True):
    """
    Register the dataset/columns used by method='chini_data'.
    Call this once in any process that needs the Chini method.
    """
    _CHINI_DATA["df"] = df
    _CHINI_DATA["x_col"] = x_col
    _CHINI_DATA["y_col"] = y_col
    _CHINI_DATA["drop_negative"] = drop_negative
    # Clear all caches that depend on the registered dataset
    _chini_stats_cached.cache_clear()
    # (Optional: if you still keep old slope-only cache elsewhere, clear it too)

def _clean_xy(df, x_col, y_col, drop_negative=True):
    sub = df[[x_col, y_col]].copy()
    mask = sub.notna().all(axis=1) & np.isfinite(sub).all(axis=1)
    if drop_negative:
        mask &= (sub[x_col] >= 0) & (sub[y_col] >= 0)
    sub = sub.loc[mask]
    if sub.empty:
        raise ValueError("No valid rows after cleaning for chini_data.")
    x = sub[x_col].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    if np.sum(x**2) == 0:
        raise ValueError("sum(x^2) == 0; cannot fit a through-origin line.")
    return x, y

def compute_through_origin_regression(
    data,
    x_col="flow_m3_per_day",
    y_col="methane_gen_kgh",
    *,
    drop_negative=False,
):
    """
    Compute slope and Excel-style through-origin R² for y ~ m * x.

    Returns:
        dict with keys:
            slope (float): least-squares slope through the origin
            r2: 1 - (ss_res / sum((y-y_mean)^2)) 
            r2_uncentered: 1 - (ss_res / sum(y^2))
            n (int): number of points used
    """
    x, y = _clean_xy(data, x_col, y_col, drop_negative=drop_negative)

    # Determine linear regression parameters (with forced origin intercept) 
    # Equations for linear regression through origin can be found here: https://bookdown.org/colettemair0/bookdown/the-method-of-least-squares.html
    denom = float(np.sum(x**2))
    slope = float(np.sum(x * y) / denom)
    
    # Determine R2 value 
    # Data source: Introductory Econometrics by Jeffrey Woodbridge 
    y_mean = y.mean() 
    y_pred = slope * x
    x_mean = x.mean()
    ss_res = float(np.sum((y - y_pred) ** 2)) # sum of squared residuals
    ss_tot = float(np.sum((y-y_mean) ** 2)) # total sum of squares
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

    # uncentered version: 
    ss_tot_uncentered = float(np.sum(y**2)) # total sum of squares without subtracting mean
    r2_origin = (1.0 - ss_res / ss_tot_uncentered) if ss_tot_uncentered > 0.0 else float("nan")

    # Calculate the standard error of the slope 
    sigma_squared = ss_res / (len(x) - 1) # Equation 2.61 in Woodldbridge 2009
    print(f"Debug: sigma_squared = {sigma_squared}, ss_res = {ss_res}, n = {len(x)}")
    se_denom = float(np.sum((x-x_mean)**2) ** 0.5) # Equation 2.61 in Woodldbridge 2009
    se_slope = (sigma_squared ** 0.5) / se_denom if se_denom > 0 else float("nan")
    

    return {
        "slope": slope,
        "r2": r2,
        "r2_origin": r2_origin,
        "n": int(len(x)),
        "se_slope": se_slope,
    }

# ----------------------------
# Cached stats for the registered dataset
# ----------------------------
@lru_cache(maxsize=1)
def _chini_stats_cached():
    """
    Compute and cache stats (slope, R², n) for the currently registered dataset.
    Uses Excel-style through-origin R² (intercept forced to 0).
    """
    df = _CHINI_DATA["df"]
    if df is None:
        raise RuntimeError("Chini dataset not set. Call set_chini_dataset(df, ...) first.")
    return compute_through_origin_regression(
        df,
        x_col=_CHINI_DATA["x_col"],
        y_col=_CHINI_DATA["y_col"],
        drop_negative=_CHINI_DATA["drop_negative"],
    )

# @lru_cache(maxsize=1)
# def _chini_slope_cached():
#     """
#     Compute and cache the through-origin slope for the registered dataset.
#     Units: (y units) / (x units), typically kg CH4/h per (m^3/day).
#     """
#     df = _CHINI_DATA["df"]
#     if df is None:
#         raise RuntimeError("Chini dataset not set. Call set_chini_dataset(df, ...) first.")
#     x_col = _CHINI_DATA["x_col"]
#     y_col = _CHINI_DATA["y_col"]
#     drop_negative = _CHINI_DATA["drop_negative"]

#     x, y = _clean_xy(df, x_col, y_col, drop_negative)
#     slope = float(np.sum(x * y) / np.sum(x**2))
#     return slope

# ----------------------------
# Public getters
# ----------------------------
def get_chini_xy():
    df = _CHINI_DATA["df"]
    return _clean_xy(df, _CHINI_DATA["x_col"], _CHINI_DATA["y_col"], _CHINI_DATA["drop_negative"])

def get_chini_stats():
    """
    Return a dict with keys: slope, r2_origin, n for the registered dataset.
    """
    # Return a shallow copy to avoid accidental mutation of cached dict
    stats = _chini_stats_cached()
    return dict(stats)

def get_chini_slope():
    """Convenience: slope (kg CH4/h per (m^3/day))."""
    return _chini_stats_cached()["slope"]

def get_chini_r2_origin():
    """Through-origin (uncentered) R² (intercept forced to 0)."""
    return _chini_stats_cached()["r2_origin"]

def get_chini_r2_centered():
    """Centered R² using SST about y-mean (often not used for through-origin fits)."""
    return _chini_stats_cached()["r2"]

def get_chini_se_slope():
    """Standard error of the slope in chini linear regression."""
    return _chini_stats_cached()["se_slope"]

def chini_confidence_intervals(flow_m3_per_day, alpha=0.05):
    """
    Compute predicted biogas production and confidence/prediction intervals
    for the Chini through-origin regression.

    Parameters
    ----------
    flow_m3_per_day : array-like
        Input flow rates (m^3/day).
    alpha : float
        Significance level (default=0.05 → 95% intervals).

    Returns
    -------
    dict of numpy arrays:
        - estimate : predicted mean value (kg CH4/h)
        - lower_ci, upper_ci : confidence interval of mean response
        - lower_pi, upper_pi : prediction interval (mean + scatter)
    """
    stats_cached = get_chini_stats()
    slope = stats_cached["slope"]

    # pull cleaned dataset to match regression
    df = _CHINI_DATA["df"]
    if df is None:
        raise RuntimeError("Chini dataset not set. Call set_chini_dataset(df, ...) first.")
    x, y = _clean_xy(df, _CHINI_DATA["x_col"], _CHINI_DATA["y_col"], _CHINI_DATA["drop_negative"])
    n = len(x)

    y_hat = slope * x
    residuals = y - y_hat
    sigma2 = float(np.sum(residuals**2) / (n - 1))

    Sxx = float(np.sum(x**2))
    var_slope = sigma2 / Sxx

    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)

    flow = np.asarray(flow_m3_per_day, dtype=float)
    est = calc_biogas_production_rate(flow, method="chini_data")

    se_mean = np.sqrt(var_slope * flow**2)
    se_pred = np.sqrt(se_mean**2 + sigma2)

    lower_ci = est - t_crit * se_mean
    upper_ci = est + t_crit * se_mean
    lower_pi = est - t_crit * se_pred
    upper_pi = est + t_crit * se_pred

    return {
        "estimate": est,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "lower_pi": lower_pi,
        "upper_pi": upper_pi,
    }


######## TARALLO ET AL FUNCTION #######
def _tarallo_mid_kgCH4_per_m3(mj_per_kg_ch4=None, m3_per_gal=0.003785411784):

    MJ_per_MG_mid = 6434.0  # Tarallo 2015 mid
    
    MJ_per_m3 = MJ_per_MG_mid / (1e6 * m3_per_gal)  # Convert MJ/MG → MJ/m^3
    if mj_per_kg_ch4 is None:
        mj_per_kg_ch4 = mj_per_kg_CH4()
    return MJ_per_m3 / mj_per_kg_ch4  # kg CH4 / m^3 treated wastewater

##### Calculate biogas production rate based on flow rate #####
def calc_biogas_production_rate(flow_m3_per_day, method="chini_data"):
    """
    Return kg CH4/h given flow (m^3/day) and a method: 'chini_data' or 'tarallo_model'.
    """
    flow = np.asarray(flow_m3_per_day, dtype=float)

    if method == "chini_data":
        slope = get_chini_slope()        # kg CH4/h per (m^3/day)
        return slope * flow                    # kg CH4/h

    if method == "tarallo_model":
        kgCH4_per_m3 = _tarallo_mid_kgCH4_per_m3()  # kg CH4 / m^3 treated
        return kgCH4_per_m3 * flow / 24.0           # kg CH4 / h

    raise ValueError("method must be 'chini_data' or 'tarallo_model'")


##### Calculate production-normalized CH4 emissions #####
def calculate_production_normalized_ch4(
    data: pd.DataFrame = None,
    load_data_func=None,
    calc_biogas_func=None
):
    """
    Calculate production-normalized CH4 emissions (% of biogas production).

    Parameters
    ----------
    data : pd.DataFrame, optional
        Input dataframe containing AD facility data. If None, will call `load_data_func()`.
    load_data_func : callable, optional
        Function to load the dataset if `data` is None.
    calc_biogas_func : callable
        Function to calculate biogas production rate from flow (required).

    Returns
    -------
    pd.DataFrame
        Dataframe with additional columns:
        - biogas_measured_num
        - calculated_biogas_production_kgCH4_per_hr
        - biogas_production_used_kgCH4_per_hr
        - production_normalized_CH4_percent
    """
    if data is None:
        if load_data_func is None:
            raise ValueError("Either `data` must be provided or `load_data_func` must be specified.")
        data = load_data_func()

    if calc_biogas_func is None:
        raise ValueError("`calc_biogas_func` must be provided.")

    df = data.copy()

    # 1) Ensure numeric measured biogas production
    df['biogas_measured_num'] = pd.to_numeric(
        df['biogas_production_kgCH4_per_hr'], errors='coerce'
    )

    # 2) Flag valid measured values
    use_measured = (
        df['reported_biogas_production'].astype(str).str.lower().eq('yes')
        & df['biogas_measured_num'].notna()
        & df['biogas_measured_num'].gt(0)
    )

    # 3) Check for flow data
    has_flow = df['flow_m3_per_day'].notna()

    # 4) Calculate from flow where available
    df['calculated_biogas_production_kgCH4_per_hr'] = np.nan
    df.loc[has_flow, 'calculated_biogas_production_kgCH4_per_hr'] = (
        df.loc[has_flow, 'flow_m3_per_day'].apply(calc_biogas_func)
    )

    # 5) Choose measured if valid, else calculated
    df['biogas_production_used_kgCH4_per_hr'] = (
        df['biogas_measured_num'].where(use_measured)
        .combine_first(df['calculated_biogas_production_kgCH4_per_hr'])
    )

    # 6) Calculate production-normalized CH4 (%)
    denom = df['biogas_production_used_kgCH4_per_hr'].where(lambda x: x > 0)
    df['production_normalized_CH4_percent'] = df['ch4_kg_per_hr'] / denom

    return df


#### For economic analysis #### 

from dataclasses import dataclass

@dataclass(frozen=True)
class Engine:
    name: str
    efficiency: float                 # electricity efficiency (0–1)
    power_to_heat_ratio: float        # P/H (dimensionless, >0)

    @property
    def heat_to_power_ratio(self) -> float:
        return 1.0 / self.power_to_heat_ratio
    

ENGINES = {
    "reciprocating_rich_burn": Engine("reciprocating_rich_burn", efficiency=0.291, power_to_heat_ratio=0.62), # Source: US EPA, 2011; Table 5
    "reciprocating_lean_burn":     Engine("reciprocating_lean_burn",     efficiency=0.326, power_to_heat_ratio=0.86), # Source: US EPA, 2011; Table 5
    "mircroturbine":     Engine("microbturbine",     efficiency=0.26, power_to_heat_ratio=0.88), # Source: US EPA, 2011; Table 5
    "fuel_cell":     Engine("fuel_cell",     efficiency=0.423, power_to_heat_ratio=1.26), # Source: US EPA, 2011; Table 5

}



def calc_leak_power_value(plant_size, leak_rate, leak_fraction_capturable, *, 
                          engine: Engine, 
                          electricity_price_per_kWh):
    """
    plant_size: size of the plant in m3/day
    leak_rate: leak rate as a fraction of the biogas production rate
    leak_fraction_capturable: fraction of the leak that can be captured

    """

    biogas_production_kgCH4_per_hr = calc_biogas_production_rate(plant_size, method="chini_data") # Function outputs biogas production in kg CH4/hr
    # print(f'Biogas production rate: {biogas_production_kg_per_hr} kg CH4/hr')

    methane_leakage_kg_per_hr = biogas_production_kgCH4_per_hr * leak_rate
    # print(f'Methane leakage: {methane_leakage_kg_per_hr} kg CH4/hr')

    methane_leakage_MJ_per_hr = methane_leakage_kg_per_hr * mj_per_kg_CH4() # Convert to MJ per hour
    # print(f'Methane leakage: {methane_leakage_MJ_per_hr} MJ/hr')
   
    electricity_generation_potential_kWh_per_hour = methane_leakage_MJ_per_hr *\
          leak_fraction_capturable * (1/mj_per_kWh()) * engine.efficiency  # Convert to kWh per hour and multiply by engine efficiency
    
    leak_value_usd_per_hour = electricity_generation_potential_kWh_per_hour * electricity_price_per_kWh # Convert to USD per hour
    
    return leak_value_usd_per_hour

def calc_leak_heat_value(plant_size, leak_rate, leak_fraction_capturable, *, 
                        engine: Engine, 
                        nat_gas_price_per_MJ): 
     """
    plant_size: size of the plant in m3/day
    leak_rate: leak rate as a fraction of the biogas production rate
    leak_fraction_capturable: fraction of the leak that can be captured
    power_to_heat_ratio: electrical production from CHP system divided by heat production from CHP system (units of energy) 
    nat_gas_price_per_MJ: price of natural gas in USD per MJ

    """
     biogas_production_kgCH4_per_hr = calc_biogas_production_rate(plant_size, method="chini_data") # Function outputs biogas production in kg CH4/hr
     # print(f'Biogas production rate: {biogas_production_kg_per_hr} kg CH4/hr')

     methane_leakage_kg_per_hr = biogas_production_kgCH4_per_hr * leak_rate
     # print(f'Methane leakage: {methane_leakage_kg_per_hr} kg CH4/hr')
     
     methane_leakage_MJ_per_hr = methane_leakage_kg_per_hr * mj_per_kg_CH4() # Convert to MJ per hour
     # print(f'Methane leakage: {methane_leakage_MJ_per_hr} MJ/hr')
     
     electricity_generation_potential_MJ_per_hour = methane_leakage_MJ_per_hr *\
        leak_fraction_capturable * engine.efficiency  # Energy production in MJ per hour
     
     heat_production = electricity_generation_potential_MJ_per_hour * (1/engine.power_to_heat_ratio) # Heat production in MJ per hour

     leak_value_usd_per_hour = heat_production * nat_gas_price_per_MJ # Convert to USD per hour

     return leak_value_usd_per_hour



def calc_leak_value_CHP(plant_size, leak_rate, leak_fraction_capturable, *, 
                        engine: Engine, 
                        electricity_price_per_kWh, nat_gas_price_per_MJ):
    """
    plant_size: size of the plant in m3/day
    leak_rate: leak rate as a fraction of the biogas production rate
    leak_fraction_capturable: fraction of the leak that can be captured

    """

    leak_electricity_usd = calc_leak_power_value(plant_size, leak_rate, leak_fraction_capturable, engine=engine, electricity_price_per_kWh=electricity_price_per_kWh)
    leak_heat_value = calc_leak_heat_value(plant_size, leak_rate, leak_fraction_capturable, engine=engine, nat_gas_price_per_MJ=nat_gas_price_per_MJ)

    leak_value_usd_per_hour = leak_electricity_usd + leak_heat_value
    
    return leak_value_usd_per_hour


def calc_payback_period(plant_size, leak_rate, leak_fraction_capturable, *, 
                        engine: Engine, 
                        electricity_price_per_kWh, nat_gas_price_per_MJ, 
                        ogi_cost=100000): 
    """
    Calculate the payback period (days) for a methane leak OGI survey based on the leak value.
    
    plant_size: size of the plant in m3/day
    biogas_production_rate: biogas production rate as MJ biogas per m3 treated flow 
    leak_rate: leak rate as a fraction of the biogas production rate
    leak_fraction_capturable: fraction of the leak that can be captured
    electricity_price_per_kWh: price of electricity in USD per kWh
    """
    leak_value = calc_leak_value_CHP(plant_size, leak_rate, leak_fraction_capturable, engine=engine, electricity_price_per_kWh=electricity_price_per_kWh, nat_gas_price_per_MJ=nat_gas_price_per_MJ)
    
    payback_period = ogi_cost / leak_value * (1/24) # Payback period in days
    
    return payback_period


def calc_annual_savings(plant_size, leak_rate, leak_fraction_capturable, *, 
                        engine: Engine, 
                        electricity_price_per_kWh, nat_gas_price_per_MJ, 
                        ogi_cost=100000):
    """
    Calculate the annual savings from capturing methane leaks.
    
    plant_size: size of the plant in m3/day
    biogas_production_rate: biogas production rate as MJ biogas per m3 treated flow 
    leak_rate: leak rate as a fraction of the biogas production rate
    leak_fraction_capturable: fraction of the leak that can be captured
    electricity_price_per_kWh: price of electricity in USD per kWh
    ogi_cost: cost of OGI survey in USD
    """
    
    leak_value = calc_leak_value_CHP(plant_size, leak_rate, leak_fraction_capturable, 
                                     engine=engine, electricity_price_per_kWh=electricity_price_per_kWh, nat_gas_price_per_MJ=nat_gas_price_per_MJ)
    

    annual_savings = leak_value * 24 * 365 - ogi_cost  # Annual savings in USD
    
    return annual_savings

def calc_annual_revenue(plant_size, leak_rate, leak_fraction_capturable, *, 
                        engine: Engine, 
                        electricity_price_per_kWh, nat_gas_price_per_MJ, ogi_cost=100000):
    """
    Calculate the annual savings from capturing methane leaks.
    
    plant_size: size of the plant in m3/day
    biogas_production_rate: biogas production rate as MJ biogas per m3 treated flow 
    leak_rate: leak rate as a fraction of the biogas production rate
    leak_fraction_capturable: fraction of the leak that can be captured
    electricity_price_per_kWh: price of electricity in USD per kWh
    ogi_cost: cost of OGI survey in USD
    """
    
    leak_value = calc_leak_value_CHP(plant_size, leak_rate, leak_fraction_capturable, 
                                     engine=engine, electricity_price_per_kWh=electricity_price_per_kWh, nat_gas_price_per_MJ=nat_gas_price_per_MJ)
    
    

    annual_revenue = leak_value * 24 * 365  # Annual revenue in USD
    
    return annual_revenue



def solve_leak_rate_for_value(target_value_usd_per_year, plant_size, leak_fraction_capturable, 
                              engine: Engine, 
                              electricity_price_per_kWh, nat_gas_price_per_MJ):
    """
    Solve for the methane leak rate (fraction of biogas lost) required 
    to reach a target monetary value of leaks.
    """

    # Step 1: Convert target annual value to $/hr
    target_value_usd_per_hour = target_value_usd_per_year / (24 * 365)

    # Step 2: Biogas production in kg CH4/hr
    biogas_prod_kg_hr = calc_biogas_production_rate(plant_size, method="chini_data")

    # Step 3: Conversion factor (USD/hr per unit leak_rate)
    conversion_factor = (
        # Electricity component 
        mj_per_kg_CH4()
        * leak_fraction_capturable
        * (1 / mj_per_kWh())
        * engine.efficiency
        * electricity_price_per_kWh 
        + # Heat component
        (mj_per_kg_CH4()
        * leak_fraction_capturable
        * engine.efficiency
        * (1/engine.power_to_heat_ratio)
        * nat_gas_price_per_MJ)
    )

    # Step 4: Solve for leak rate
    leak_rate = target_value_usd_per_hour / (biogas_prod_kg_hr * conversion_factor)

    return leak_rate


def annualized_cost(capital_cost, lifetime_years, discount_rate):
    """
    Calculate annualized cost from a capital investment.

    Parameters
    ----------
    capital_cost : float
        Initial capital investment ($)
    lifetime_years : int
        Project lifetime in years
    discount_rate : float
        Annual discount/interest rate (as a decimal, e.g. 0.07 for 7%)

    Returns
    -------
    float
        Annualized cost ($/year)
    """
    if discount_rate == 0:  # avoid divide by zero
        return capital_cost / lifetime_years

    i = discount_rate
    n = lifetime_years
    crf = (i * (1 + i) ** n) / ((1 + i) ** n - 1)
    return capital_cost * crf