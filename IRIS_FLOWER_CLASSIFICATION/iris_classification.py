import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
file_path = 'Iris.csv'  # Update with your file path
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Drop the 'Id' column if it exists
df.drop(columns=['Id'], inplace=True, errors='ignore')

# Splitting features and labels
X = df.iloc[:, :-1]  # Features (Sepal and Petal measurements)
y = df.iloc[:, -1]   # Target (Species)

# Encode the target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Create visualizations directory if not exists
os.makedirs('visualizations', exist_ok=True)

# Histogram for feature distributions
features = df.columns[:-1]
colors = ['skyblue', 'lightgreen', 'orange', 'red']
plt.figure(figsize=(12, 8))
for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)
    plt.hist(df[feature], bins=15, color=colors[i], alpha=0.7, edgecolor='black')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
plt.tight_layout()
plt.savefig('visualizations/histograms.png')
plt.show()

# Pairplot for feature relationships
pairplot = sns.pairplot(df, hue='Species', diag_kind='kde', palette='husl')
pairplot.fig.suptitle('Pairplot of Iris Features', y=1.02)
pairplot.savefig('visualizations/pairplot.png')
plt.show()

# Boxplot for feature comparison
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.iloc[:, :-1], orient='h', palette='coolwarm')
plt.title('Boxplot of Feature Distributions')
plt.savefig('visualizations/boxplot.png')
plt.show()

# Violin plot for feature spread
plt.figure(figsize=(12, 6))
sns.violinplot(data=df.iloc[:, :-1], palette='muted')
plt.title('Violin Plot of Feature Distributions')
plt.savefig('visualizations/violin_plot.png')
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('visualizations/confusion_matrix.png')
plt.show()

# Display Conclusion in Terminal
print("\nConclusion:")
print("1. The RandomForestClassifier achieved high accuracy in classifying Iris species.")
print("2. The classification report shows strong precision, recall, and F1-score for all species.")
print("3. Histograms indicate clear separability of features, especially petal length and width.")
print("4. Pairplot visualization shows Setosa is distinct, while Versicolor and Virginica have slight overlap.")
print("5. Boxplots and violin plots provide deeper insights into feature distributions and spread.")
print("6. The confusion matrix confirms minimal misclassifications, proving the model’s effectiveness.")
print("7. Further improvements can be made by testing other models like SVM or KNN.")