import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# Load dataset
file_path = "advertising.csv"  # Ensure the file is in the same directory

df = pd.read_csv(file_path)
print("\nFirst 5 rows of dataset:\n", df.head())
print("\nDataset Info:\n")
print(df.info())
print("\nDataset Statistics:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# Define features and target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nMean Squared Error (MSE):", round(mse, 2))
print("R² Score:", round(r2, 2))

# Create 'visualizations' folder if not exists
if not os.path.exists("visualizations"):
    os.makedirs("visualizations")

# Plot graphs one by one
sns.set_style("whitegrid")

# TV vs Sales
plt.figure()
sns.scatterplot(x=df['TV'], y=df['Sales'])
plt.xlabel("TV Advertisement Budget")
plt.ylabel("Sales")
plt.title("TV Ad Budget vs Sales")
plt.savefig("visualizations/tv_vs_sales.png")
plt.show()

# Radio vs Sales
plt.figure()
sns.scatterplot(x=df['Radio'], y=df['Sales'])
plt.xlabel("Radio Advertisement Budget")
plt.ylabel("Sales")
plt.title("Radio Ad Budget vs Sales")
plt.savefig("visualizations/radio_vs_sales.png")
plt.show()

# Newspaper vs Sales
plt.figure()
sns.scatterplot(x=df['Newspaper'], y=df['Sales'])
plt.xlabel("Newspaper Advertisement Budget")
plt.ylabel("Sales")
plt.title("Newspaper Ad Budget vs Sales")
plt.savefig("visualizations/newspaper_vs_sales.png")
plt.show()

print("✅ All visualizations saved in 'visualizations' folder.")
