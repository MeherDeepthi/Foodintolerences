# ml_training_notebook.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_excel("ML_Ready_Ingredient_Symptom_Data.xlsx")

# Define features and target
X = df.drop(columns=["Date", "Symptom_Severity"])
y = df["Symptom_Severity"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
with open("severity_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Feature importance visualization
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.coef_[0] if len(model.coef_.shape) == 2 else model.coef_
})
coefficients = coefficients.sort_values(by="Importance", ascending=False)

print("Feature columns used for training:")
print(X.columns.tolist())

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=coefficients)
plt.title("Feature Importance for Symptom Severity")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

