# 🚢 Titanic Survival Prediction

## 🎯 Project Overview
This project predicts whether a passenger on the Titanic survived or not using machine learning. The dataset contains various passenger details such as **age, gender, ticket class, fare, and embarkation point**. 

We use **data preprocessing, feature engineering, and classification models** to make predictions based on historical data.

---

## 📂 **Project Structure**
📁 Titanic_Prediction
│── 📄 titanic_model.py        # Main Python script for training and prediction
│── 📄 titanic-Dataset.csv     # Dataset used for the project
│── 📄 README.md               # Documentation for the project
│── 📄 requirements.txt        # Required libraries for running the project
│── 📁 visualizations          # Folder containing generated graphs


---

## ⚙️ **How It Works**
1. **Data Loading**  
   - The dataset **titanic-Dataset.csv** is loaded using `pandas`.

2. **Data Preprocessing**  
   - **Dropping Irrelevant Columns:** `PassengerId`, `Name`, `Ticket`, `Cabin`  
   - **Handling Missing Values:**  
     - `Age` is filled with the median.  
     - `Embarked` is filled with the most frequent (mode) value.  
     - `Fare` is filled with the median.  
   - **Encoding Categorical Features:**  
     - `Sex` is converted to **0 (male)** and **1 (female)**.  
     - `Embarked` is mapped to numerical values.  

3. **Model Training**  
   - We use a **Random Forest Classifier** to train the model.
   - The dataset is split into **80% training and 20% testing**.

4. **Prediction & Evaluation**  
   - The model predicts whether a passenger survived or not.
   - Performance is measured using **accuracy, precision, recall, and F1-score**.

5. **Visualization**  
   - Graphs are generated to understand survival patterns:
     - **Survival count** (bar chart)
     - **Survival by gender**
     - **Survival by passenger class**
     - **Age distribution of survivors vs. non-survivors**

---

## 🖥️ **Installation & Running the Project**
### 🔹 **Step 1: Install Required Libraries**
Ensure you have Python installed. Then, install the required libraries:
```bash
pip install -r requirements.txt

Or install them manually
pip install pandas numpy matplotlib seaborn scikit-learn
🔹 Step 2: Run the Code
Run the Python script:
python titanic_model.py


📊 Model Performance
Metric	Score
Accuracy	81.56%
Precision	0.83 (Non-Survivors), 0.79 (Survivors)
Recall	0.86 (Non-Survivors), 0.76 (Survivors)
F1-Score	0.85 (Non-Survivors), 0.77 (Survivors)
The model provides a good balance between precision and recall, meaning it effectively predicts survival outcomes.

📊 Visualizations
After running the script, these graphs will pop up:

Survival Count: A bar chart showing the number of survivors and non-survivors.
Survival by Gender: How survival rates differ between males and females.
Survival by Passenger Class: Shows survival rates across 1st, 2nd, and 3rd class passengers.
Age Distribution: A histogram comparing the ages of survivors and non-survivors.

🚀 Conclusion
This project demonstrates how machine learning can be used to analyze historical events and make predictions. The model provides useful insights into survival patterns based on passenger characteristics.

🔹 Future Improvements:

Try different models like Logistic Regression, SVM, or XGBoost for better accuracy.
Perform hyperparameter tuning for model optimization.
Add feature engineering to improve predictions.

📝 Author
Darapureddy Renuka
Intern at Codesoft
Project: Titanic Survival Prediction
