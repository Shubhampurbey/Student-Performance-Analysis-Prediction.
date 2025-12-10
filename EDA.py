import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"C:\Users\spurb\OneDrive\Desktop\ML Project\student_performance.csv")
sns.set_style("whitegrid")

numeric_cols = [
    'Age', 'Attendance (%)', 'Study Hours per Day',
    'Homework Completion (%)', 'Previous Exam Score',
    'Class Participation (%)', 'Final Score'
]

categorical_cols = ['Gender', 'Extra Coaching', 'Final Result']

# Histogram (Large Labels)
data[numeric_cols].hist(figsize=(12, 7), color="skyblue", edgecolor="black")
plt.suptitle("Distribution of Numeric Features", fontsize=20)
plt.xlabel("Value", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Countplot (Gender)
plt.figure(figsize=(8, 5))
sns.countplot(x=data['Gender'], hue=data['Gender'], legend=False, palette="viridis")
plt.title("Gender Distribution", fontsize=20)
plt.xlabel("Gender", fontsize=16)
plt.ylabel("Number of Students", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Countplot (Extra Coaching)
plt.figure(figsize=(8, 5))
sns.countplot(x=data['Extra Coaching'], hue=data['Extra Coaching'], legend=False, palette="magma")
plt.title("Extra Coaching Distribution", fontsize=20)
plt.xlabel("Extra Coaching (Yes / No)", fontsize=16)
plt.ylabel("Number of Students", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Boxplot (Numeric Columns)
plt.figure(figsize=(14, 7))
sns.boxplot(data=data[numeric_cols], palette="coolwarm")
plt.title("Outlier Detection in Numeric Features", fontsize=20)
plt.xlabel("Numeric Features", fontsize=16)
plt.ylabel("Values", fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="Spectral", linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features", fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Pairplot (Important Features)
sns.set(rc={'figure.figsize':(12, 12)}) 
g = sns.pairplot(data[numeric_cols[:4]], diag_kind="kde", palette="husl")
g.fig.subplots_adjust(top=0.93)
g.fig.suptitle("Pairplot of Key Numeric Features", fontsize=22)
plt.show()

# Scatter Plot (Study Hours vs Final Score)
plt.figure(figsize=(7, 5))
sns.scatterplot(x=data['Study Hours per Day'], y=data['Final Score'], color="purple")
plt.title("Study Hours vs Final Score", fontsize=20)
plt.xlabel("Study Hours per Day", fontsize=16)
plt.ylabel("Final Score", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Violin Plot (Gender vs Final Score)
plt.figure(figsize=(8, 5))
sns.violinplot(x=data['Gender'], y=data['Final Score'], hue=data['Gender'], legend=False, palette="Set2")
plt.title("Final Score Distribution by Gender", fontsize=20)
plt.xlabel("Gender", fontsize=16)
plt.ylabel("Final Score", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Trend Line (Final Score Across Index)
plt.figure(figsize=(12, 4))
plt.plot(data['Final Score'], color="tomato", linewidth=2)
plt.title("Trend of Final Score Across Students", fontsize=20)
plt.xlabel("Student Index", fontsize=16)
plt.ylabel("Final Score", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
