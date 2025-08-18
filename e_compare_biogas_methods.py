#%% 
### SETUP

# scripts/10_register_chini_dataset.py (or at the top of any notebook/script)
import pandas as pd
import pathlib
import numpy as np
from a_my_utilities import set_chini_dataset, calc_biogas_production_rate

chini_data = pd.read_csv(pathlib.Path("02_clean_data", "chini_cleaned.csv"))  # or csv

set_chini_dataset(
    chini_data,
    x_col="flow_m3_per_day",
    y_col="methane_gen_kgh",
    drop_negative=True
)


#%% 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

def plot_chini_vs_tarallo_over_flow(
    flow_min=0.0,
    flow_max=1_000_000.0,
    n_points=500,
    save_dir=pathlib.Path("03_figures"),
    filename="chini_vs_tarallo_over_flow.png",
    title="Biogas Production vs Flow: Chini vs Tarallo",
    xlabel="Flow (m³/day)",
    ylabel="Methane as Biogas (kg/h)",
    color_chini="#1f77b4",
    color_tarallo="#ff7f0e"
):
    """Overlay Chini and Tarallo predictions as functions of flow."""
    flows = np.linspace(flow_min, flow_max, n_points)

    y_chini   = calc_biogas_production_rate(flows, "chini_data")   # kg CH4/h
    y_tarallo = calc_biogas_production_rate(flows, "tarallo_model")

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    plt.plot(flows, y_chini,   label="Chini (through-origin fit)", linewidth=2, color=color_chini)
    plt.plot(flows, y_tarallo, label="Tarallo (mid)",               linewidth=2, color=color_tarallo)

    plt.xlim(0, flow_max)
    plt.ylim(0, max(y_chini.max(), y_tarallo.max()) * 1.05)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    return out


def plot_parity_chini_vs_tarallo(
    flow_min=0.0,
    flow_max=1_000_000.0,
    n_points=300,
    save_dir=pathlib.Path("03_figures"),
    filename="chini_vs_tarallo_parity.png",
    title="Parity Plot: Chini vs Tarallo",
    xlabel="Tarallo (kg/h)",
    ylabel="Chini (kg/h)",
    point_size=60
):
    """Parity (y=x) comparison of Chini vs Tarallo across a flow range."""
    flows = np.linspace(flow_min, flow_max, n_points)
    y_chini   = calc_biogas_production_rate(flows, "chini_data")
    y_tarallo = calc_biogas_production_rate(flows, "tarallo_model")

    lim = max(y_chini.max(), y_tarallo.max()) * 1.05

    sns.set_style("whitegrid")
    plt.figure(figsize=(6.5, 6))
    plt.scatter(y_tarallo, y_chini, s=point_size, edgecolor="k", alpha=0.7)
    plt.plot([0, lim], [0, lim], linestyle="--", linewidth=1)  # 1:1 line

    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    return out

# %%
# Assumes you've already called set_chini_dataset(...)
plot_chini_vs_tarallo_over_flow(flow_min=0, flow_max=1_000_000)

# Optional: parity plot to see agreement/deviation from y=x
plot_parity_chini_vs_tarallo(flow_min=0, flow_max=1_000_000)

# %%

#%% 
from a_my_utilities import load_ch4_emissions_data

def plot_parity_chini_vs_tarallo_on_measurements(
    save_dir=pathlib.Path("03_figures"),
    filename="chini_vs_tarallo_parity_measurements.png",
    title="Compare Chini and Tarallo Estimates (Measurement Data)",
    xlabel="Tarallo (kg/h)",
    ylabel="Chini (kg/h)",
    point_size=60
):
    """Parity (y=x) comparison of Chini vs Tarallo using facility measurement dataset."""
    # Load the measurement dataset
    measurement_data = load_ch4_emissions_data()

    # Calculate biogas production estimates from both models
    measurement_data["biogas_chini"] = calc_biogas_production_rate(
        measurement_data["flow_m3_per_day"], "chini_data"
    )
    measurement_data["biogas_tarallo"] = calc_biogas_production_rate(
        measurement_data["flow_m3_per_day"], "tarallo_model"
    )

    # Drop missing values if any
    df = measurement_data.dropna(subset=["biogas_chini", "biogas_tarallo"])

    # Axis limits
    lim = max(df["biogas_chini"].max(), df["biogas_tarallo"].max()) * 1.05 if len(df) else 1.0

    sns.set_style("whitegrid")
    plt.figure(figsize=(6.5, 6))
    plt.scatter(
        df["biogas_tarallo"],
        df["biogas_chini"],
        s=point_size,
        edgecolor="k",
        alpha=0.7
    )
    plt.plot([0, lim], [0, lim], linestyle="--", linewidth=1)  # 1:1 line

    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    return out

# Example usage (optional)
plot_parity_chini_vs_tarallo_on_measurements()

