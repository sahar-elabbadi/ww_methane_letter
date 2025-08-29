#%%

from a_my_utilities import M3_PER_GAL, annualized_cost

# Calculate solids production based on flow rate 

def calc_biosolids_primary_mass(q): 
    """
    Calculate primary dry biosolids (m_p) in g/day for low/baseline/high scenarios 
    Parameters
    ----------
    q : float
        Flow in m³/day

    Returns
    -------
    dict
        Total dry solids (kg/day) for each scenario: {'low': ..., 'baseline': ..., 'high': ...}
    """
    # convert flow to liters/day (since mg/L basis)
    q_l_per_day = q * 1000  


    # All values from Seiple et al., 2017
    tss_mg_per_l = {
        'low': 120,
        'baseline': 260,
        'high': 400,
    }

    f_settlable = {
        'low': 0.4,
        'baseline': 0.6,
        'high': 0.7,
    }
    
    results = {}

    for scenario in ['low', 'baseline', 'high']:
        # Primary solids
        m_p = q_l_per_day * tss_mg_per_l[scenario] * f_settlable[scenario] / 1e6 # convert to kg / day instead of g / day
        
        results[scenario] = m_p
    
    return results 



def calc_biosolids_secondary(q):
    """
    Calculate total dry solids (m_t) in grams/day for low, baseline, and high scenarios.

    Parameters
    ----------
    q : float
        Flow in m³/day

    Returns
    -------
    dict
        Total dry solids (grams/day) for each scenario: {'low': ..., 'baseline': ..., 'high': ...}
    """

    # All values from Seiple et al., 2017
    tss_mg_per_l = {
        'low': 120,
        'baseline': 260,
        'high': 400,
    }

    So_mg_per_l = {
        'low': 110,
        'baseline': 230,
        'high': 350,
    }

    f_settlable = {
        'low': 0.4,
        'baseline': 0.6,
        'high': 0.7,
    }

    f_v = {
        'low': 0.8,
        'baseline': 0.85,
        'high': 0.9,
    }

    k = {
        'low': 0.4,
        'baseline': 0.4,
        'high': 0.6,
    }

    results = {}

    # convert flow to liters/day (since mg/L basis)
    q_l_per_day = q * 1000  

    for scenario in ['low', 'baseline', 'high']:
        # Secondary solids
        m_s = q_l_per_day * (
            k[scenario] * So_mg_per_l[scenario]
            + ((1 - f_settlable[scenario]) * tss_mg_per_l[scenario]) * (1 - f_v[scenario])
        )
        results[scenario] = m_s / 1e6 # convert to kg/day instead of g/day

    return results


def reactor_volume(mass_rate_dict, sg_solids, sg_sludge, theta_dict={'low':15, 'baseline':20, 'high': 25}, rho_water=1000):
    """
    Calculate sludge volume per day and reactor volume for different scenarios.

    Parameters
    ----------
    mass_rate_dict : dict
        Dictionary of solids production rates [kg/day] with keys "low", "baseline", "high".
    sg_solids : float
        Specific gravity of solids (dimensionless).
    sg_sludge : float
        Specific gravity of sludge (dimensionless).
    theta_dict : dict
        Dictionary of retention times [days] with keys "low", "baseline", "high".
        Default values are from Rittman & McCarty, page 623; typical theta values for high-rate digesters 
    rho_water : float, optional
        Density of water [kg/m³]. Default = 1000.

    Returns
    -------
    reactor_volume : dict
        Reactor volumes for each scenario [m³].
    sludge_volume_per_day : dict
        Sludge volumes per day for each scenario [m³/day].
    """
    sludge_volume_per_day = {}
    reactor_volume = {}

    # densities
    rho_s = sg_solids * rho_water
    rho_sludge = sg_sludge * rho_water

    for scenario, m_s in mass_rate_dict.items():
        # solids volume per day
        V_s = m_s / rho_s

        # bulk sludge volume per day (using derived formula)
        V_sludge_day = (V_s * rho_water - m_s) / (rho_water - rho_sludge)

        # required reactor volume
        V_reactor = V_sludge_day * theta_dict[scenario]

        sludge_volume_per_day[scenario] = V_sludge_day
        reactor_volume[scenario] = V_reactor

    return reactor_volume, sludge_volume_per_day


def new_digester_cost(volume_dict): 
    """
    volume : dict
        Dictionary with keys "low", "baseline", "high".
        Each value is another dict with:
            - 'sludge_volume_day' : sludge volume produced per day [m³/day]
            - 'reactor_volume'    : required reactor volume [m³]
    """
    cost = {} 

    for scenario, vol in volume_dict.items(): 
        gallons = vol / M3_PER_GAL
        # Constant from EBMUD: 
        multiplier = 1.4 # For fully loaded costs 
        cost[scenario] = 10 * gallons * multiplier # Construction cost is $10/gallon, multiply by 1.4 for consultant, construction management, etc. 
    return cost 



def size_digester_from_flow(q,
                            theta_dict=None,
                            sg_primary_solids=1.4, sg_primary_sludge=1.02,
                            sg_secondary_solids=1.25, sg_secondary_sludge=1.05):
    """
    Wrapper that:
      1) computes primary & secondary solids (kg/day) from flow q using your functions,
      2) computes reactor volumes for each stream using their respective SGs,
      3) sums reactor volumes, and
      4) computes total cost from total reactor volume using your cost function.

    Parameters
    ----------
    q : float
        Flow rate [m³/day]
    theta_dict : dict or None
        Optional override for HRT [days] with keys 'low','baseline','high'.
        If None, uses the default inside your reactor_volume() function.
    sg_primary_solids : float
    sg_primary_sludge : float
    sg_secondary_solids : float
    sg_secondary_sludge : float

    Returns
    -------
    dict
        {
          'primary':   {'mass_kg_day': {...}, 'sludge_m3_day': {...}, 'reactor_m3': {...}},
          'secondary': {'mass_kg_day': {...}, 'sludge_m3_day': {...}, 'reactor_m3': {...}},
          'total':     {'reactor_m3': {...}, 'cost_$': {...}}
        }
    """
    scenarios = ('low', 'baseline', 'high')

    # 1) Mass production (kg/day) from your functions
    m_primary   = calc_biosolids_primary_mass(q)    # expects kg/day (you updated)
    m_secondary = calc_biosolids_secondary(q)       # expects kg/day (you updated)

    # 2) Reactor volumes using your reactor_volume() with different SGs
    if theta_dict is None:
        reactor_vol_primary, sludge_vol_primary = reactor_volume(
            m_primary, sg_primary_solids, sg_primary_sludge
        )
        reactor_vol_secondary, sludge_vol_secondary = reactor_volume(
            m_secondary, sg_secondary_solids, sg_secondary_sludge
        )
    else:
        reactor_vol_primary, sludge_vol_primary = reactor_volume(
            m_primary, sg_primary_solids, sg_primary_sludge, theta_dict=theta_dict
        )
        reactor_vol_secondary, sludge_vol_secondary = reactor_volume(
            m_secondary, sg_secondary_solids, sg_secondary_sludge, theta_dict=theta_dict
        )

    # 3) Totals
    reactor_vol_total = {
        s: reactor_vol_primary[s] + reactor_vol_secondary[s] for s in scenarios
    }

    # 4) Cost from your cost function
    cost_total = new_digester_cost(reactor_vol_total)

    return {
        'primary': {
            'mass_kg_day': m_primary,
            'sludge_m3_day': sludge_vol_primary,
            'reactor_m3': reactor_vol_primary,
        },
        'secondary': {
            'mass_kg_day': m_secondary,
            'sludge_m3_day': sludge_vol_secondary,
            'reactor_m3': reactor_vol_secondary,
        },
        'total': {
            'reactor_m3': reactor_vol_total,
            'cost_$': cost_total,
        }
    }


# q = 70_000_000  # m³/day

# results = size_digester_from_flow(q)  # uses default theta inside reactor_volume()

# for s in ('low','baseline','high'):
#     print(f"\nScenario: {s}")
#     print(f"  Primary mass = {results['primary']['mass_kg_day'][s]:,.1f} kg/day")
#     print(f"  Secondary mass = {results['secondary']['mass_kg_day'][s]:,.1f} kg/day")
#     print(f"  Primary sludge = {results['primary']['sludge_m3_day'][s]:,.2f} m³/day")
#     print(f"  Secondary sludge = {results['secondary']['sludge_m3_day'][s]:,.2f} m³/day")
#     print(f"  Reactor volume (primary) = {results['primary']['reactor_m3'][s]:,.1f} m³")
#     print(f"  Reactor volume (secondary) = {results['secondary']['reactor_m3'][s]:,.1f} m³")
#     print(f"  TOTAL reactor volume = {results['total']['reactor_m3'][s]:,.1f} m³")
#     print(f"  TOTAL cost = ${results['total']['cost_$'][s]:,.0f}")



def annualized_digester_cost(cost_dict, lifetime_years=30, discount_rate=0.07):
    """
    Calculate annualized digester cost for low/baseline/high scenarios.

    Parameters
    ----------
    cost_dict : dict
        Capital costs [$] for each scenario: {'low': ..., 'baseline': ..., 'high': ...}
    lifetime_years : int
        Project lifetime [years], default 30
    discount_rate : float
        Discount/interest rate (decimal, default 0.07 = 7%)

    Returns
    -------
    dict
        Annualized cost [$ per year] for each scenario
    """
    results = {}
    for scenario, capex in cost_dict.items():
        results[scenario] = annualized_cost(capex, lifetime_years, discount_rate)
    return results
q = 100_000  # m³/day

# 1. Run full sizing workflow
results = size_digester_from_flow(q)

# 2. Extract total capital costs
capex = results['total']['cost_$']  # {'low': ..., 'baseline': ..., 'high': ...}

# 3. Annualize
annualized = annualized_digester_cost(capex, lifetime_years=50, discount_rate=0.07)

print(f"Flow rate of facility: {q/1e6:,.1f} Mm3/day")
# 4. Print results
for s in ('low','baseline','high'):
    print(f"{s.capitalize()}:")
    print(f"  Capital cost = ${capex[s]:,.0f}")
    print(f"  Annualized cost = ${annualized[s]:,.0f}/yr")


import numpy as np
import matplotlib.pyplot as plt

# Flow range (1 → 1.6 million m³/day, 50 points)
flows = np.linspace(1, 1_600_000, 50)

# Containers for results
cost_low, cost_base, cost_high = [], [], []

# Loop over flows
for q in flows:
    results = size_digester_from_flow(q)
    capex = results['total']['cost_$']
    cost_low.append(capex['low'])
    cost_base.append(capex['baseline'])
    cost_high.append(capex['high'])

# Convert to numpy arrays
cost_low = np.array(cost_low)
cost_base = np.array(cost_base)
cost_high = np.array(cost_high)

# --- Compute slopes ($ per (m³/day)) using endpoints ---
slope_low = (cost_low[-1] - cost_low[0]) / (flows[-1] - flows[0])
slope_base = (cost_base[-1] - cost_base[0]) / (flows[-1] - flows[0])
slope_high = (cost_high[-1] - cost_high[0]) / (flows[-1] - flows[0])

# --- Plot ---
plt.figure(figsize=(10,6))
plt.plot(flows, cost_low, label=f"Low scenario (slope = {slope_low:,.2f} $/(m³/day))", linestyle="--")
plt.plot(flows, cost_base, label=f"Baseline scenario (slope = {slope_base:,.2f} $/(m³/day))", linestyle="-")
plt.plot(flows, cost_high, label=f"High scenario (slope = {slope_high:,.2f} $/(m³/day))", linestyle="-.")
plt.xlabel("Flow rate (m³/day)")
plt.ylabel("Capital Cost ($)")
plt.title("Digester Capital Cost vs Flow Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
