# Reproducible ML Pipeline Using Dagster

## Overview
This project implements a reproducible machine learning pipeline using Dagster.
The pipeline demonstrates dependency-aware execution and selective re-running
of steps when data or preprocessing logic changes.

## Dataset
California Housing Dataset (Scikit-learn)

## Pipeline Steps
- Data Loading
- Exploratory Data Analysis (EDA)
- Train-Test Split
- Model Training and Evaluation

## Models Used
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Linear Regression

## Key Feature
When preprocessing parameters are modified, only downstream model training
steps are re-executed instead of rerunning the entire pipeline.

## How to Run
```bash
pip install -r requirements.txt
dagster dev -f housing_pipeline.py

Open browser:
http://127.0.0.1:3000

Output
DAG visualization in Dagster UI
RMSE values for each model
EDA plot saved as eda_plot.png

License
Academic use only
