import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

# Generate synthetic dataset for children malnutrition
# Features: age_months, weight_kg, height_cm, muac_cm (mid-upper arm circumference)
# Target: 0 = normal, 1 = malnourished

np.random.seed(42)
n_samples = 5000

# Realistic ranges
age_months = np.random.randint(6, 60, n_samples)
weight_kg = np.random.uniform(5, 25, n_samples)
height_cm = np.random.uniform(60, 120, n_samples)
muac_cm = np.random.uniform(10, 20, n_samples)

# Create synthetic target based on WHO standards (simplified)
# Malnutrition if weight-for-age or height-for-age or MUAC is low
malnourished = []
for i in range(n_samples):
    # Simulate risk factors
    low_weight = weight_kg[i] < (age_months[i] * 0.3 + 3)  # rough threshold
    low_height = height_cm[i] < (age_months[i] * 0.6 + 50)
    low_muac = muac_cm[i] < 12.5
    if low_weight or low_height or low_muac:
        malnourished.append(1)
    else:
        malnourished.append(0)

X = pd.DataFrame({
    'age_months': age_months,
    'weight_kg': weight_kg,
    'height_cm': height_cm,
    'muac_cm': muac_cm
})
y = np.array(malnourished)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Model Evaluation:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")

# Save the model and evaluation metrics for the web app
joblib.dump(model, 'model.pkl')
with open('metrics.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")

print("Model saved as model.pkl and metrics saved to metrics.txt")