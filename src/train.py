# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1️⃣ Load processed data (relative to project root)
df = pd.read_csv("data/processed/diabetes_processed.csv")  # <-- remove ../

# 2️⃣ Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 3️⃣ Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5️⃣ Predict & evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6️⃣ Save model
joblib.dump(model, "models/diabetes_model.pkl")  # <-- remove ../
print("Model saved to models/diabetes_model.pkl")