import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 🔹 Load the accident dataset
df = pd.read_csv("road_accidents_cleaned.csv")

# 🔹 Check dataset columns
print("\n✅ Available columns in dataset:")
print(df.columns)

# 🔹 Select only existing features for training (removes missing ones)
feature_columns = ["Severity - 2015", "Total Accidents - 2015", "Killed - 2015", "Injured - 2015"]
df = df[feature_columns].dropna()  # Remove missing values

# 🔹 Shuffle dataset for variation
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 🔹 Define labels (risk level) based on severity
df["Risk Level"] = np.where(df["Severity - 2015"] > df["Severity - 2015"].median(), 1, 0)  # 1 = High Risk, 0 = Low Risk

# 🔹 Check class balance
print("\n🔹 Risk Level Distribution:\n", df["Risk Level"].value_counts())

# 🔹 Split dataset for training with stratification
X = df.drop("Risk Level", axis=1)  # Features
y = df["Risk Level"]  # Labels

print("\n✅ Model expects the following features:")
print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)

# 🔹 Train the Random Forest Classifier with improved generalization
model = RandomForestClassifier(n_estimators=30, max_depth=3, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)

# 🔹 Model Evaluation
y_pred = model.predict(X_test)
print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

# 🔹 Use Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf)
print("\n✅ Stratified Cross-Validation Accuracy:", cv_scores.mean())

# 🔹 Save the trained model for Streamlit integration
joblib.dump(model, "safety_model.pkl")
print("\n✅ Model saved as 'safety_model.pkl'!")

# 🔹 Feature Importance Visualization
plt.figure(figsize=(8, 6))
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("🔍 Feature Importance in Route Safety Prediction")
plt.show()
