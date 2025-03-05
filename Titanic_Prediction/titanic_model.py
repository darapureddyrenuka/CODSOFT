import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Ensure the visualizations folder exists
os.makedirs("visualizations", exist_ok=True)

# Load dataset
df = pd.read_csv("titanic-Dataset.csv")

# Display basic info
print("\n📊 Dataset Preview:\n", df.head())

# Drop irrelevant columns
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Convert categorical variables
label_enc = LabelEncoder()
df["Sex"] = label_enc.fit_transform(df["Sex"])
df["Embarked"] = label_enc.fit_transform(df["Embarked"])

# Splitting data
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")

print("\n📌 Classification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# 📊 Data Visualization
# -------------------------------

# Survival Count Plot
plt.figure(figsize=(6, 4))
sns.countplot(x="Survived", data=df, hue="Survived", palette={0: "red", 1: "green"}, legend=False)
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.savefig("visualizations/survival_countplot.png")
plt.show()

# Survival by Gender
plt.figure(figsize=(6, 4))
sns.countplot(x="Sex", hue="Survived", data=df, palette="coolwarm")
plt.xticks([0, 1], ["Female", "Male"])
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.savefig("visualizations/survival_by_gender.png")
plt.show()

# Survival by Passenger Class
plt.figure(figsize=(6, 4))
sns.countplot(x="Pclass", hue="Survived", data=df, palette="Blues")
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.savefig("visualizations/survival_by_pclass.png")
plt.show()

# Age Distribution
plt.figure(figsize=(6, 4))
sns.histplot(df["Age"], bins=20, kde=True, color="purple")
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("visualizations/age_distribution.png")
plt.show()

# Fare Distribution by Survival
plt.figure(figsize=(6, 4))
sns.boxplot(x="Survived", y="Fare", data=df, hue="Survived", palette={0: "red", 1: "green"}, dodge=False)
plt.title("Fare vs Survival")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Fare Amount")
plt.savefig("visualizations/fare_vs_survival.png")
plt.show()

print("\n✅ All visualizations have been saved inside the 'visualizations' folder.")
