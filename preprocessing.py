import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------
# 1. Load Dataset
# ---------------------------------------------
data = pd.read_csv(r"C:\Users\spurb\OneDrive\Desktop\ML Project\student_performance.csv")

print("Dataset Loaded Successfully!")
print("Shape:", data.shape)
print(data.head())
print(data.info())
print("\nMissing Values:\n", data.isnull().sum())

# ---------------------------------------------
# 2. Identify Numeric & Categorical Columns
# ---------------------------------------------
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

print("Numeric Columns:", numeric_cols)
print("Categorical Columns:", categorical_cols)

# ---------------------------------------------
# 3. Handle Missing Values
# ---------------------------------------------
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

data[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])
data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

print("Missing values handled successfully!")

# ---------------------------------------------
# 4. Handle Outliers (IQR Capping)
# ---------------------------------------------
for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    data[col] = data[col].clip(lower, upper)

print("Outliers handled using IQR capping!")

# ---------------------------------------------
# 5. One-Hot Encode Categorical Columns
# ---------------------------------------------
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

encoded_array = ohe.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(categorical_cols))

# Combine numeric + encoded categorical
data = pd.concat([data[numeric_cols].reset_index(drop=True), encoded_df], axis=1)

print("Categorical encoding completed!")

# ---------------------------------------------
# 6. Scale Numeric Columns
# ---------------------------------------------
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

print("Scaling completed!")

# ---------------------------------------------
# 7. Final Output
# ---------------------------------------------
print("\nFinal Preprocessed Dataset:")
print(data.head())
print("\nNew Shape:", data.shape)
