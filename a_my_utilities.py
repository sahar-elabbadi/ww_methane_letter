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
METHANE_KG_PER_SCF = 0.019176  # kg of methane per scf
METHANE_MJ_PER_KG = 50.4  # MJ per kg of methane 

####### Data Loading ########################################

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

    return df


def load_chp_facilities():
    # Load the CSV using your custom loader
    wwtp_data = load_and_clean_facility_data(pathlib.PurePath('01_raw_data',
                                                               'supplementary_database_C.xlsx'))

    # Filter for energy recovery facilities and copy the slice
    energy_recovery = wwtp_data[
        wwtp_data['treatment_train'].apply(
            lambda treatments: any('e' in str(t) for t in treatments)
        )
    ].copy()

    return energy_recovery

def load_ad_facilities():
    # Load the CSV using your custom loader
    wwtp_data = load_and_clean_facility_data(pathlib.PurePath('01_raw_data',
                                                               'supplementary_database_C.xlsx'))

    # Filter for energy recovery facilities and copy the slice
    anaerobic_digestion = wwtp_data[
        wwtp_data['treatment_train'].apply(
            lambda treatments: any('1' in str(t) for t in treatments)
        )
    ].copy()

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
m3_per_gal = 0.003785411784
# mj_per_kg_ch4 = 50.4 # energy content of methane

def m3_per_mg(): 
    return m3_per_gal * 1e6 # Convert MG to m3 


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
    return mgd * 1e6 * m3_per_gal


def g_per_s_to_kg_per_hour(g_per_s):
    """
    Convert grams per second to kg per hour 
    """

    
    return g_per_s *3.6

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
            r2_origin (float): 1 - SS_res / sum(y^2)  (Excel when intercept=0)
            n (int): number of points used
    """
    x, y = _clean_xy(data, x_col, y_col, drop_negative=drop_negative)

    denom = float(np.sum(x**2))
    slope = float(np.sum(x * y) / denom)

    y_pred = slope * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot_origin = float(np.sum(y ** 2))
    r2_origin = (1.0 - ss_res / ss_tot_origin) if ss_tot_origin > 0.0 else float("nan")

    return {
        "slope": slope,
        "r2_origin": r2_origin,
        "n": int(len(x)),
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

def get_chini_r2():
    """Convenience: Excel-style through-origin R² (intercept=0)."""
    return _chini_stats_cached()["r2_origin"]


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


def calc_leak_value(plant_size, leak_rate, leak_fraction_capturable, engine_efficiency, electricity_price_per_kWh):
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
          leak_fraction_capturable * (1/mj_per_kWh()) * engine_efficiency  # Convert to kWh per hour and multiply by engine efficiency
    
    leak_value_usd_per_hour = electricity_generation_potential_kWh_per_hour * electricity_price_per_kWh # Convert to USD per hour
    
    return leak_value_usd_per_hour


def calc_payback_period(plant_size, leak_rate, leak_fraction_capturable, engine_efficiency, electricity_price_per_kWh, ogi_cost=100000): 
    """
    Calculate the payback period (days) for a methane leak OGI survey based on the leak value.
    
    plant_size: size of the plant in m3/day
    biogas_production_rate: biogas production rate as MJ biogas per m3 treated flow 
    leak_rate: leak rate as a fraction of the biogas production rate
    leak_fraction_capturable: fraction of the leak that can be captured
    electricity_price_per_kWh: price of electricity in USD per kWh
    """
    leak_value = calc_leak_value(plant_size, leak_rate, leak_fraction_capturable, engine_efficiency, electricity_price_per_kWh)
    
    payback_period = ogi_cost / leak_value * (1/24) # Payback period in days
    
    return payback_period


def calc_annual_savings(plant_size, leak_rate, leak_fraction_capturable, engine_efficiency, electricity_price_per_kWh, ogi_cost=100000):
    """
    Calculate the annual savings from capturing methane leaks.
    
    plant_size: size of the plant in m3/day
    biogas_production_rate: biogas production rate as MJ biogas per m3 treated flow 
    leak_rate: leak rate as a fraction of the biogas production rate
    leak_fraction_capturable: fraction of the leak that can be captured
    electricity_price_per_kWh: price of electricity in USD per kWh
    ogi_cost: cost of OGI survey in USD
    """
    
    leak_value = calc_leak_value(plant_size, leak_rate, leak_fraction_capturable, engine_efficiency, electricity_price_per_kWh)
    

    annual_savings = leak_value * 24 * 365 - ogi_cost  # Annual savings in USD
    
    return annual_savings

def calc_annual_revenue(plant_size, leak_rate, leak_fraction_capturable, engine_efficiency, electricity_price_per_kWh, ogi_cost=100000):
    """
    Calculate the annual savings from capturing methane leaks.
    
    plant_size: size of the plant in m3/day
    biogas_production_rate: biogas production rate as MJ biogas per m3 treated flow 
    leak_rate: leak rate as a fraction of the biogas production rate
    leak_fraction_capturable: fraction of the leak that can be captured
    electricity_price_per_kWh: price of electricity in USD per kWh
    ogi_cost: cost of OGI survey in USD
    """
    
    leak_value = calc_leak_value(plant_size, leak_rate, leak_fraction_capturable, engine_efficiency, electricity_price_per_kWh)
    

    annual_revenue = leak_value * 24 * 365  # Annual revenue in USD
    
    return annual_revenue



def solve_leak_rate_for_value(target_value_usd_per_year, plant_size, leak_fraction_capturable, engine_efficiency, electricity_price_per_kWh):
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
        mj_per_kg_CH4()
        * leak_fraction_capturable
        * (1 / mj_per_kWh())
        * engine_efficiency
        * electricity_price_per_kWh
    )

    # Step 4: Solve for leak rate
    leak_rate = target_value_usd_per_hour / (biogas_prod_kg_hr * conversion_factor)

    return leak_rate