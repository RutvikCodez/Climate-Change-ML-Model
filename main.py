import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_preprocessing import load_data, preprocess_data
from src.model import train_models

st.set_page_config(page_title="Climate Change AI", layout="wide")

st.title("🌍 Climate Change AI: CO₂ Emission Prediction")

# =========================
# Load Dataset
# =========================

data_raw = load_data("data/world_bank_development_indicators.csv")

st.header("1️⃣ Raw Dataset Overview")

st.write("Rows:", data_raw.shape[0])
st.write("Columns:", data_raw.shape[1])

st.dataframe(data_raw.head())

# =========================
# Data Preprocessing
# =========================

data, features, target = preprocess_data(data_raw)
# st.subheader("Features Used for Training")
# st.write(features)

# st.subheader("Target Variable")
# st.write(target)


st.header("2️⃣ Cleaned Dataset")

st.write("Countries:", data["Country"].nunique())
st.write("Years:", data["Year"].nunique())
st.write("Rows:", len(data))

st.dataframe(data.head())

# =========================
# Model Training
# =========================

st.header("3️⃣ Model Training")

lr, rf, performance, X_test, y_test, rf_pred = train_models(data, features, target)
# st.subheader("Train/Test Information")

# st.write("Test Samples:", len(X_test))
# st.write("Target Samples:", len(y_test))

st.success("Models trained successfully!")
# =========================
# Model Prediction Test
# =========================

# st.subheader("Model Prediction Test")

# sample_input = X_test.iloc[0:1]

# prediction = rf.predict(sample_input)

# st.write("Sample Input Features:")
# st.dataframe(sample_input)

# st.write("Predicted CO₂ Emission:")
# st.write(prediction)
# =========================
# Model Performance
# =========================

st.header("4️⃣ Model Performance")

st.dataframe(performance)

# =========================
# Visualizations
# =========================

st.header("5️⃣ Visualizations")

# Correlation Heatmap

st.subheader("Correlation Heatmap")

fig1, ax1 = plt.subplots()

sns.heatmap(
    data[features + [target]].corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax1
)

st.pyplot(fig1)

# Feature Importance

st.subheader("Feature Importance")

importance = rf.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

fig2, ax2 = plt.subplots()

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df,
    ax=ax2
)

st.pyplot(fig2)

# Actual vs Predicted

st.subheader("Actual vs Predicted CO₂")

fig3, ax3 = plt.subplots()

ax3.scatter(y_test, rf_pred, alpha=0.5)
# ax3.scatter(y_test, rf_pred, alpha=0.6)

# ax3.plot([y_test.min(), y_test.max()],
#          [y_test.min(), y_test.max()],
#          color="red")
ax3.set_xlabel("Actual CO2")
ax3.set_ylabel("Predicted CO2")

st.pyplot(fig3)

# CO2 Trend

st.subheader("CO₂ Emission Trend (Country-wise)")

selected_country = st.selectbox(
    "Select Country",
    sorted(data["Country"].unique())
)

country_df = data[data["Country"] == selected_country]

fig4, ax4 = plt.subplots(figsize=(14,6))   # bigger graph

ax4.plot(country_df["Year"], country_df["CO2"], marker="o")
ax4.set_xlabel("Year")
ax4.set_ylabel("CO₂ Emissions (Normalized)")
ax4.set_title(f"CO₂ Emission Trend for {selected_country}")

plt.xticks(rotation=45)

st.pyplot(fig4, use_container_width=True)