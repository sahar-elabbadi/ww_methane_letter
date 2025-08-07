import pandas as pd 
import pathlib
import numpy as np


import pandas as pd
import ast

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
mj_per_kg_ch4 = 50.4 # energy content of methane

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
    return mj_ch4 * (1/mj_per_kg_ch4)


def mj_per_kg_CH4(): 
    return mj_per_kg_ch4



def mj_per_kWh(): 
    return 3.6 # 3.6 MJ per kWh


####### Analysis Functions ########################################


def calc_biogas_production_rate(flow_m3_per_day): 
    """
    Calculate biogas production based on an input flow rate, using Tarallo et al process models
    
    Returns biogas production rate mid, low, and high

    Returns kg CH4 produced as biogas / hour
    """

    # Biogas production range for facilites, as reported in Tarallo et al., 2015

    biogas_production_mid = 6434 # Units: MJ / MG
    biogas_production_low = 6343 # Units: MJ / MG
    biogas_production_high =  6874 # Units: MJ / MG

    # Convert to MJ / m3 
    biogas_production_mid = biogas_production_mid * (1/m3_per_gal) * 1e-6 # Units: MJ / m3 treated

    # Convert to kg CH4 / m3 
    kgCH4_production_mid = mj_to_kg_CH4(biogas_production_mid) # Units: kg CH4 produced as biogas / m3 treated
   
    return kgCH4_production_mid * flow_m3_per_day /24 # final units: kg CH4 produced as biogas / hour 




#### For economic analysis #### 


def calc_leak_value(plant_size, biogas_production_rate, leak_rate, leak_fraction_capturable, electricity_price_per_kWh):
    """
    plant_size: size of the plant in m3/day
    biogas_production_rate: biogas production rate as MJ biogas per m3 treated flow 
    leak_rate: leak rate as a fraction of the biogas production rate
    leak_fraction_capturable: fraction of the leak that can be captured

    """

    biogas_production_MJ_per_day = plant_size * biogas_production_rate
    # print(f'Biogas production rate: {biogas_production_MJ_per_day} MJ/day')

    methane_leakage_MJ_per_hr = biogas_production_MJ_per_day * leak_rate * (1/24) # Convert to per hour
    # print(f'Methane leakage: {methane_leakage_MJ_per_hr} MJ/hr')
   
    methane_leakage_kg_per_hr = methane_leakage_MJ_per_hr * (1/mj_per_kg_CH4()) # Convert to kg CH4 per hour
    # print(f'Methane leakage: {methane_leakage_kg_per_hr} kg CH4/hr')
   
    electricity_generation_potential_kWh_per_hour = methane_leakage_MJ_per_hr *\
          leak_fraction_capturable * (1/mj_per_kWh()) # Convert to kWh per hour
    
    leak_value_usd_per_hour = electricity_generation_potential_kWh_per_hour * electricity_price_per_kWh # Convert to USD per hour
    
    return leak_value_usd_per_hour


def calc_payback_period(plant_size, biogas_production_rate, leak_rate, leak_fraction_capturable, electricity_price_per_kWh, ogi_cost=100000): 
    """
    Calculate the payback period (days) for a methane leak OGI survey based on the leak value.
    
    plant_size: size of the plant in m3/day
    biogas_production_rate: biogas production rate as MJ biogas per m3 treated flow 
    leak_rate: leak rate as a fraction of the biogas production rate
    leak_fraction_capturable: fraction of the leak that can be captured
    electricity_price_per_kWh: price of electricity in USD per kWh
    """

    leak_value = calc_leak_value(plant_size, biogas_production_rate, leak_rate, leak_fraction_capturable, electricity_price_per_kWh)
    
    payback_period = ogi_cost / leak_value * (1/24) # Payback period in days
    
    return payback_period


def calc_annual_savings(plant_size, biogas_production_rate, leak_rate, leak_fraction_capturable, electricity_price_per_kWh, ogi_cost=100000):
    """
    Calculate the annual savings from capturing methane leaks.
    
    plant_size: size of the plant in m3/day
    biogas_production_rate: biogas production rate as MJ biogas per m3 treated flow 
    leak_rate: leak rate as a fraction of the biogas production rate
    leak_fraction_capturable: fraction of the leak that can be captured
    electricity_price_per_kWh: price of electricity in USD per kWh
    ogi_cost: cost of OGI survey in USD
    """
    
    leak_value = calc_leak_value(plant_size, biogas_production_rate, leak_rate, leak_fraction_capturable, electricity_price_per_kWh)
    

    annual_savings = leak_value * 24 * 365 - ogi_cost  # Annual savings in USD
    
    return annual_savings