import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from xgboost import XGBRegressor

def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    return df

def feature_engineering(df):
    df['RoomsPerBedroom'] = (
            df['AveRooms'] / df['AveBedrms']
    )

    df['IncomePerOccupant'] = (
            df['MedInc'] / df['AveOccup']
    )

    df['PopulationDensity'] = (
            df['Population'] / df['AveOccup']
    )

    df["Population_log"] = np.log1p(df["Population"])

    df["AveOccup_log"] = np.log1p(df["AveOccup"])

    return df

def process_data(df):
    X = df.drop(['MedHouseVal'], axis=1)
    y = df['MedHouseVal']

    return X, y

def train_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    joblib.dump(model, 'model/model.pkl')
    print("Model saved")

    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    print("\nModel Performance\n")
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R²  :", r2)

    return y_pred

def show_feature_importance(model,X):
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })
    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )
    print("\nFeature Importance\n")
    print(importance_df)

    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )
    plt.title("XGBoost Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

def shap_analysis(model, X_test):
    print("\nRunning SHAP Analysis...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary Plot
    shap.summary_plot(
        shap_values,
        X_test
    )

    # Feature Importance Plot
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar"
    )

    # Waterfall Plot
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_test.iloc[0],
            feature_names=X_test.columns
        )
    )

def main():
    os.makedirs("model", exist_ok=True)
    print("Loading Dataset...")
    df = load_data()

    print("Performing Feature Engineering...")
    df = feature_engineering(df)

    X, y = process_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    print("Training XGBoost...")
    model = train_model(
        X_train,
        y_train
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )

    show_feature_importance(
        model,
        X
    )

    shap_analysis(
        model,
        X_test
    )

if __name__ == "__main__":
    main()