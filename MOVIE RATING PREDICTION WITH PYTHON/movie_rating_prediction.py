import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os  # To handle file paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create 'visualizations' folder if it doesn't exist
os.makedirs("visualizations", exist_ok=True)

# Load dataset
df = pd.read_csv("Movie dataset.csv", encoding="latin1")

# Clean 'Year' column
df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Clean 'Duration' column
df['Duration'] = df['Duration'].astype(str).str.extract(r'(\d+)').astype(float)

# Clean 'Votes' column
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

# Fill missing numerical values with median
df['Year'] = df['Year'].fillna(df['Year'].median())
df['Duration'] = df['Duration'].fillna(df['Duration'].median())
df['Votes'] = df['Votes'].fillna(df['Votes'].median())

# Handle categorical data
categorical_cols = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
for col in categorical_cols:
    df[col] = df[col].fillna('Unknown')
    df[col] = LabelEncoder().fit_transform(df[col])

# Define features and target
X = df[['Duration', 'Year', 'Votes', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
y = df['Rating']

# Drop rows where target is missing
df = df.dropna(subset=['Rating'])
X = X.loc[df.index]
y = y.loc[df.index]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", round(mse, 2))
print("R-squared Score:", round(r2, 2))

# Visualizations
sns.set_style("whitegrid")

# Scatter plot: Actual vs Predicted Ratings
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Ratings")
plt.savefig("visualizations/actual_vs_predicted.png")  # Save figure
plt.show()  # Show plot

# Bar plot: Feature Importance
plt.figure(figsize=(8,5))
feature_importance = pd.Series(model.coef_, index=X.columns)
feature_importance.sort_values().plot(kind='barh', color='teal')
plt.xlabel("Coefficient Value")
plt.ylabel("Features")
plt.title("Feature Importance in Rating Prediction")
plt.savefig("visualizations/feature_importance.png")  # Save figure
plt.show()  # Show plot

# Histogram: Distribution of Predicted Ratings
plt.figure(figsize=(8,5))
sns.histplot(y_pred, bins=20, kde=True, color='purple')
plt.xlabel("Predicted Ratings")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Ratings")
plt.savefig("visualizations/predicted_ratings_distribution.png")  # Save figure
plt.show()  # Show plot

print("Visualizations saved in 'visualizations' folder.")
