import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# LOAD DATA
data = pd.read_csv(r"C:\Users\spurb\OneDrive\Desktop\ML Project\student_performance.csv")

target = "Final Result"
numeric_cols = [
    'Age', 'Attendance (%)', 'Study Hours per Day',
    'Homework Completion (%)', 'Previous Exam Score',
    'Class Participation (%)', 'Final Score'
]
categorical_cols = ['Gender', 'Extra Coaching']
X = data[numeric_cols + categorical_cols]
y = data[target]

# PREPROCESSING
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

# EVALUATION FUNCTION
def evaluate(model_name, y_true, y_pred):
    print(f"\n======== {model_name} ========")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, pos_label="Pass"))
    print("Recall   :", recall_score(y_true, y_pred, pos_label="Pass"))
    print("F1 Score :", f1_score(y_true, y_pred, pos_label="Pass"))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
# MODELS DICTIONARY
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel="rbf")
}

# TRAIN AND EVALUATE EACH MODEL

for name, model in models.items():
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])
    
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    evaluate(name, y_test, preds)
