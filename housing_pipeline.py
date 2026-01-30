from dagster import op, job
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


@op
def load_data():
    data = fetch_california_housing(as_frame=True)
    return data.frame


@op
def eda(df):
    plt.figure(figsize=(6,4))
    sns.histplot(df["MedHouseVal"], bins=30)
    plt.title("House Value Distribution")
    plt.savefig("eda_plot.png")
    plt.close()
    return df


@op
def split_data(df):
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


@op
def decision_tree(data):
    X_train, X_test, y_train, y_test = data
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    print("Decision Tree RMSE:", rmse)


@op
def random_forest(data):
    X_train, X_test, y_train, y_test = data
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    print("Random Forest RMSE:", rmse)


@op
def gradient_boosting(data):
    X_train, X_test, y_train, y_test = data
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    print("Gradient Boosting RMSE:", rmse)


@op
def linear_regression(data):
    X_train, X_test, y_train, y_test = data
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    print("Linear Regression RMSE:", rmse)


@job
def housing_ml_pipeline():
    df = load_data()
    df = eda(df)
    split = split_data(df)
    decision_tree(split)
    random_forest(split)
    gradient_boosting(split)
    linear_regression(split)
