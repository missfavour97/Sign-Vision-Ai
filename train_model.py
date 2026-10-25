import os
import glob
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

dataset_path = 'dataset'
all_files = glob.glob(os.path.join(dataset_path, "*.csv"))

print(f"Found {len(all_files)} dataset files. Loading...")

X_list = []
y_list = []

for filename in all_files:
    # Extract the digit/label directly from the filename (e.g., "Sign 0.csv" -> "0")
    base_name = os.path.basename(filename)
    match = re.search(r'Sign\s+(\d+)', base_name, re.IGNORECASE)
    if not match:
        continue
    label = match.group(1)

    # Read CSV without assuming header structure
    df = pd.read_csv(filename, header=None)

    # Convert everything possible to numbers, turning non-numeric strings (like image paths) into NaN
    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    
    # Drop rows that couldn't be converted or contain NaNs
    numeric_df = numeric_df.dropna()

    if numeric_df.empty:
        continue

    # Add each valid row as a feature vector
    for row_values in numeric_df.values:
        X_list.append(row_values)
        y_list.append(label)

X = np.array(X_list)
y = np.array(y_list)

print(f"Cleaned dataset shape: {X.shape}")

if len(X) == 0:
    print("Error: No valid numeric data found. Please check your CSV contents.")
    exit()

# 3. Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train a Machine Learning Classifier
print("Training the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Training Complete! Accuracy: {accuracy * 100:.2f}%")

# 6. Save the trained model to disk
joblib.dump(model, 'sign_language_model.pkl')
print("Model saved successfully as 'sign_language_model.pkl'!")