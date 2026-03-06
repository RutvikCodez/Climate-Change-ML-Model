from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd


def train_models(data, features, target):

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=200, random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)

    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)

    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    performance = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "RMSE": [lr_rmse, rf_rmse],
        "MAE": [lr_mae, rf_mae],
        "R2 Score": [lr_r2, rf_r2]
    })

    return lr, rf, performance, X_test, y_test, rf_pred