from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def train_models(data):

    X = data.drop(columns=["CO2_per_capita", "country"])
    y = data["CO2_per_capita"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    results = {
        "LR_R2": r2_score(y_test, y_pred_lr),
        "RF_R2": r2_score(y_test, y_pred_rf),
        "LR_MAE": mean_absolute_error(y_test, y_pred_lr),
        "RF_MAE": mean_absolute_error(y_test, y_pred_rf),
        "LR_RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        "RF_RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    }

    return rf, X_test, y_test, y_pred_rf, results