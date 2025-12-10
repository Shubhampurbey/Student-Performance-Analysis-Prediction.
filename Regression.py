import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# LOAD DATASET

data = pd.read_csv(r"C:\Users\spurb\OneDrive\Desktop\ML Project\student_performance.csv")

# SET FEATURES AND TARGET
target = "Final Score"
numeric_cols = [
    'Age', 'Attendance (%)', 'Study Hours per Day',
    'Homework Completion (%)', 'Previous Exam Score',
    'Class Participation (%)'
]

categorical_cols = ['Gender', 'Extra Coaching']

X = data[numeric_cols + categorical_cols]
y = data[target]

# PREPROCESSING PIPELINE
numeric_transform = Pipeline(steps=[
    ("scaler", StandardScaler())
])
categorical_transform = Pipeline(steps=[
    ("encoder", OneHotEncoder(drop="first"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transform, numeric_cols),
        ("cat", categorical_transform, categorical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# EVALUATION FUNCTION (ALL METRICS)
def evaluate_and_print(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n===== {model_name} =====")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return mae, mse, rmse, r2

# SIMPLE LINEAR REGRESSION
simple_lr = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LinearRegression())
])

simple_lr.fit(X_train, y_train)
pred1 = simple_lr.predict(X_test)
mae1, mse1, rmse1, r21 = evaluate_and_print("SIMPLE LINEAR REGRESSION", y_test, pred1)

# MULTIPLE LINEAR REGRESSION
multi_lr = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LinearRegression())
])
multi_lr.fit(X_train, y_train)
pred2 = multi_lr.predict(X_test)
mae2, mse2, rmse2, r22 = evaluate_and_print("MULTIPLE LINEAR REGRESSION", y_test, pred2)

# POLYNOMIAL REGRESSION
poly_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", LinearRegression())
])

poly_model.fit(X_train, y_train)
pred3 = poly_model.predict(X_test)
mae3, mse3, rmse3, r23 = evaluate_and_print("POLYNOMIAL REGRESSION (deg=2)", y_test, pred3)

# RANDOM FOREST REGRESSOR
rf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])

rf.fit(X_train, y_train)
pred4 = rf.predict(X_test)
mae4, mse4, rmse4, r24 = evaluate_and_print("RANDOM FOREST REGRESSOR", y_test, pred4)

# 11. MODEL COMPARISON SUMMARY
print("\nMODEL COMPARISON")
print("Model\t\tMAE\t\tMSE\t\tRMSE\t\tR2")
print("-------------------------------------------------------")
print(f"Simple LR\t{mae1:.3f}\t{mse1:.3f}\t{rmse1:.3f}\t{r21:.3f}")
print(f"Multiple LR\t{mae2:.3f}\t{mse2:.3f}\t{rmse2:.3f}\t{r22:.3f}")
print(f"Poly Reg\t{mae3:.3f}\t{mse3:.3f}\t{rmse3:.3f}\t{r23:.3f}")
print(f"RandomForest\t{mae4:.3f}\t{mse4:.3f}\t{rmse4:.3f}\t{r24:.3f}")

best_r2_model = max(
    {"Simple LR": r21, "Multiple LR": r22, "Polynomial": r23, "Random Forest": r24},
    key=lambda x: {"Simple LR": r21, "Multiple LR": r22, "Polynomial": r23, "Random Forest": r24}[x]
)

print(f"\n🔥 Best Model Based on R² = {best_r2_model}")
