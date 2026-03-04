import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_clean_data(filepath):

    data = pd.read_csv(filepath)

    # Rename columns
    data = data.rename(columns={
        "CO2_emisions": "CO2",
        "renewvable_energy_consumption%": "renewable_energy",
        "electric_power_consumption": "energy_use",
        "GDP_current_US": "gdp"
    })

    # Convert date to year
    data["Year"] = pd.to_datetime(data["date"], errors="coerce").dt.year
    data = data.drop(columns=["date"])

    # Sort properly
    data = data.sort_values(["country", "Year"])

    # Forward + backward fill
    data = data.groupby("country").apply(lambda x: x.ffill().bfill())
    data = data.reset_index(drop=True)

    # Remove countries with more than 30% missing values
    threshold = 0.3
    valid_countries = data.groupby("country").apply(
        lambda x: x.isnull().mean().mean() < threshold
    )
    valid_countries = valid_countries[valid_countries].index
    data = data[data["country"].isin(valid_countries)]

    data = data.dropna()

    # Create per capita features
    data["CO2_per_capita"] = data["CO2"] / data["population"]
    data["GDP_per_capita"] = data["gdp"] / data["population"]

    # Population growth
    data["population_growth"] = data.groupby("country")["population"].pct_change()
    data["population_growth"] = data["population_growth"].fillna(0)

    # Keep required columns
    final_columns = [
        "country",
        "Year",
        "CO2_per_capita",
        "GDP_per_capita",
        "renewable_energy",
        "energy_use",
        "population_growth"
    ]

    data = data[final_columns]

    # Remove extreme outliers
    data = data[data["CO2_per_capita"] < data["CO2_per_capita"].quantile(0.99)]

    # Normalize features
    scaler = MinMaxScaler()

    features = [
        "GDP_per_capita",
        "renewable_energy",
        "energy_use",
        "population_growth"
    ]

    data[features] = scaler.fit_transform(data[features])

    return data