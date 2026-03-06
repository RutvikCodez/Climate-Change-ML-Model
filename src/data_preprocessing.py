import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):

    # Convert column names to lowercase
    df.columns = df.columns.str.lower()

    # Rename columns safely
    df = df.rename(columns={
        "country": "Country",
        "date": "Year",
        "avg_precipitation": "Precipitation",
        "forest_land%": "Forest_Land",
        "agricultural_land%": "Agricultural_Land",
        "renewable_energy_consumption%": "Renewable_Energy",
        "renewable_energy_consumption": "Renewable_Energy",
        "electric_power_consumption": "Electric_Power",
        "co2_emissions": "CO2",
        "co2_emisions": "CO2",
        "other_greenhouse_emissions": "GHG",
        "population_density": "Population_Density",
        "population": "Population",
        "gdp_current_us": "GDP"
    })

    # Keep only columns that exist
    required = [
        "Country",
        "Year",
        "Precipitation",
        "Forest_Land",
        "Agricultural_Land",
        "Renewable_Energy",
        "Electric_Power",
        "CO2",
        "GHG",
        "Population_Density",
        "Population",
        "GDP"
    ]

    df = df[[col for col in required if col in df.columns]]

    df = df.dropna()

    features = [
        "GDP",
        "Renewable_Energy",
        "Electric_Power",
        "Population_Density",
        "Population",
        "Forest_Land",
        "Agricultural_Land",
        "Precipitation"
    ]

    features = [f for f in features if f in df.columns]

    target = "CO2"

    scaler = MinMaxScaler()

    df[features + [target]] = scaler.fit_transform(df[features + [target]])

    return df, features, target