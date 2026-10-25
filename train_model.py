import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Initialize lists to hold dataset rows
X_list = []

# Path to dataset directory
dataset_path = "dataset"

if os.path.exists(dataset_path):
    files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    print(f"Found {len(files)} dataset files. Loading...")
    
    for file in files:
        file_path = os.path.join(dataset_path, file)
        df = pd.read_csv(file_path, header=None)
        
        for _, row in df.iterrows():
            try:
                # Convert row to numeric, filtering out image paths and invalid text
                numeric_row = pd.to_numeric(row, errors='coerce')
                if not numeric_row.isna().any():
                    X_list.append(numeric_row.values)
            except Exception:
                continue

dataset_array = np.array(X_list)

print(f"Cleaned dataset shape: {dataset_array.shape}")

if dataset_array.size == 0:
    print("Error: No valid numeric data found. Please check your CSV contents.")
    exit()

# Separate labels (column 0) and the 63 landmark features (columns 1 to 64)
y = dataset_array[:, 0]
X = dataset_array[:, 1:64]

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

# Save the trained model
joblib.dump(model, 'sign_language_model.pkl')
print("Model saved successfully as 'sign_language_model.pkl'!")