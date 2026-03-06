# 🌍 Climate Change AI: Tackling Climate Change with Machine Learning

## 📌 Project Overview

Climate change is one of the most critical global challenges facing the world today. Predicting carbon emissions can help governments and policymakers design better environmental strategies.

This project applies **Machine Learning models** to analyze global climate, population, and economic indicators and **predict CO₂ emissions**.

The system uses a **cleaned dataset derived from World Bank Development Indicators** and compares multiple ML algorithms to identify the most accurate model.

The project demonstrates how **data-driven insights can support climate mitigation strategies**.

---

# 🎯 Objectives

* Analyze global climate and environmental indicators
* Predict **CO₂ emissions** using Machine Learning models
* Compare performance of multiple ML algorithms
* Visualize climate trends across countries
* Provide insights for climate policy and sustainability planning

---

# 📊 Dataset

Dataset Source:
World Bank – World Development Indicators

Dataset used in this project is a **filtered and optimized dataset** containing only the required climate indicators.

You can download the dataset from:

https://www.kaggle.com/datasets/nicolasgonzalezmunoz/world-bank-world-development-indicators/data

Place the dataset inside the project folder:

```
data/world_bank_development_indicators.csv
```

---

# 🗂 Project Structure

```
climate-change-ai
├── data/
│   └── world_bank_development_indicators.csv
├── output/
│   └── Climate Change AI final.pdf
├── src/
│   ├── data_preprocessing.py
│   └── model.py
├── .gitignore
├── co2_model.pkl
├── main.py
├── README.md
└── requirements.txt

```

---

# ⚙️ Technologies Used

* Python
* Machine Learning
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

# 🤖 Machine Learning Models

The following models were implemented:

### Linear Regression

A statistical model used for predicting CO₂ emissions based on independent variables.

### Random Forest Regressor

An ensemble learning model that improves prediction accuracy by combining multiple decision trees.

Random Forest performs better because it captures **non-linear relationships** in climate data.

---

# 📉 Features Used

The machine learning model uses the following indicators:

* Average precipitation
* Forest land percentage
* Agricultural land percentage
* Renewable energy consumption
* Electric power consumption
* Population density
* Population
* GDP (current US$)

These indicators influence CO₂ emission levels across countries.

---

# 📈 Model Evaluation

The models are evaluated using the following metrics:

* **RMSE (Root Mean Squared Error)**
* **MAE (Mean Absolute Error)**
* **R² Score (Coefficient of Determination)**

Example output from the model comparison:

| Model             | RMSE   | MAE    | R² Score  |
| ----------------- | -----  | -----  | --------  |
| Linear Regression | 0.0354 | 0.0125 | 0.8318    |
| Random Forest     | 0.0106 | 0.0019 | 0.9848    |

The **Random Forest model achieved the best performance**.

---

# 📊 Visualizations

The system generates several visualizations:

* Correlation Heatmap
* Feature Importance Graph
* Actual vs Predicted CO₂ Emissions
* CO₂ Emission Trend by Country

These visualizations help understand how different environmental and economic indicators influence carbon emissions.

---

# 🖥 System Workflow

1️⃣ Load climate dataset
2️⃣ Clean and preprocess the data
3️⃣ Handle missing values
4️⃣ Select important climate indicators
5️⃣ Normalize numerical features
6️⃣ Train Machine Learning models
7️⃣ Evaluate model performance
8️⃣ Visualize results and trends

---

# 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```
git clone https://github.com/YOUR_USERNAME/climate-change-ai.git
```

---

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 3️⃣ Add Dataset

Download dataset and place it in:

```
data/world_bank_development_indicators.csv
```

---

### 4️⃣ Run the Streamlit App

```
streamlit run main.py
```

The dashboard will open in your browser.

---

# 📸 Output Screenshots
📄 Full Report: [Download PDF](output/Climate%20Change%20AI%20final.pdf)

---

# 🔮 Future Improvements

* Add more environmental indicators
* Apply **Deep Learning models (LSTM / Neural Networks)**
* Predict emissions for future years
* Deploy the project using **Streamlit Cloud**
* Build a real-time climate monitoring dashboard

---

# 👨‍💻 Contributors

**Project Leader**

Rutvik Darji

**Team Member**

Hepin Suthar

---

# 🌱 Conclusion

This project demonstrates how machine learning can be used to analyze environmental data and predict CO₂ emissions. The results show that **ensemble models like Random Forest provide better accuracy** for complex climate datasets.

Such data-driven approaches can support governments, researchers, and organizations in making informed decisions for **climate change mitigation and sustainability**.

---
