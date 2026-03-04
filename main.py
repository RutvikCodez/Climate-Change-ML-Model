import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

plt.close('all')

st.set_page_config(page_title="Climate Change AI", layout="wide")

st.title("🌍 Climate Change AI: CO₂ Emission Prediction")

# =====================================================
# 1️⃣ LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_csv("data/WDIData.csv")
    return df

data_raw = load_data()

st.header("1️⃣ Raw Dataset Overview")

st.write("Total Rows:", len(data_raw))
st.write("Total Columns:", len(data_raw.columns))

st.subheader("Preview (First 5 Rows)")
st.dataframe(data_raw.head(), width="stretch")

# =====================================================
# 2️⃣ FILTER REQUIRED INDICATORS
# =====================================================

required_indicators = [
    "CO2 emissions (metric tons per capita)",
    "GDP per capita (current US$)",
    "Renewable energy consumption (% of total final energy consumption)",
    "Electric power consumption (kWh per capita)",
    "Population growth (annual %)"
]

data = data_raw[data_raw["Indicator Name"].isin(required_indicators)]

# Convert wide year columns into rows
data = data.melt(
    id_vars=["Country Name", "Indicator Name"],
    var_name="Year",
    value_name="Value"
)

data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
data = data.dropna(subset=["Year"])

# Pivot indicators into columns
data = data.pivot_table(
    index=["Country Name", "Year"],
    columns="Indicator Name",
    values="Value"
).reset_index()

data = data.dropna()

# Rename columns
data.columns = [
    "Country",
    "Year",
    "CO2",
    "GDP",
    "Renewable_Energy",
    "Energy_Use",
    "Population_Growth"
]

st.header("2️⃣ Cleaned Dataset")

st.write("Countries:", data["Country"].nunique())
st.write("Years:", data["Year"].nunique())
st.write("Final Rows:", len(data))

st.subheader("Cleaned Data Preview")
st.dataframe(data.head(10), width="stretch")

# =====================================================
# 3️⃣ NORMALIZATION
# =====================================================

features = ["GDP", "Renewable_Energy", "Energy_Use", "Population_Growth"]
target = "CO2"

scaler = MinMaxScaler()
data[features + [target]] = scaler.fit_transform(data[features + [target]])

# =====================================================
# 4️⃣ MODEL TRAINING
# =====================================================

st.header("3️⃣ Model Training")

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.write("Training Samples:", len(X_train))
st.write("Testing Samples:", len(X_test))

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=200, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

st.success("Models Trained Successfully ✅")

# =====================================================
# 5️⃣ MODEL PERFORMANCE
# =====================================================

st.header("4️⃣ Model Performance Comparison")

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

performance_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "RMSE": [round(lr_rmse, 4), round(rf_rmse, 4)],
    "MAE": [round(lr_mae, 4), round(rf_mae, 4)],
    "R² Score": [round(lr_r2, 4), round(rf_r2, 4)]
})

st.dataframe(performance_df, width="stretch")

# =====================================================
# 6️⃣ VISUALIZATIONS
# =====================================================

st.header("5️⃣ Visualizations")

# 1️⃣ Correlation Heatmap
st.subheader("Correlation Heatmap")

fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.heatmap(
    data[features + [target]].corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax1
)
st.pyplot(fig1)

# 2️⃣ Feature Importance
st.subheader("Feature Importance (Random Forest)")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig2, ax2 = plt.subplots()
sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df,
    hue="Feature",
    legend=False,
    ax=ax2
)
st.pyplot(fig2)

# 3️⃣ Actual vs Predicted
st.subheader("Actual vs Predicted CO₂ (Random Forest)")

fig3, ax3 = plt.subplots()
ax3.scatter(y_test, rf_pred, alpha=0.5)
ax3.set_xlabel("Actual CO₂")
ax3.set_ylabel("Predicted CO₂")
ax3.set_title("Actual vs Predicted CO₂ Emissions")
st.pyplot(fig3)

# 4️⃣ CO₂ Trend (Country-wise)
st.subheader("CO₂ Emission Trend (Country-wise)")

selected_country = st.selectbox(
    "Select Country",
    sorted(data["Country"].unique())
)

country_df = data[data["Country"] == selected_country]

fig4, ax4 = plt.subplots()
ax4.plot(country_df["Year"], country_df["CO2"])
ax4.set_xlabel("Year")
ax4.set_ylabel("CO₂ (Normalized)")
ax4.set_title(f"CO₂ Trend - {selected_country}")
st.pyplot(fig4)