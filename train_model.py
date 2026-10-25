import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

X_list = []
y_list = []
dataset_path = "dataset"

if os.path.exists(dataset_path):
    files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    print(f"Found {len(files)} dataset files. Loading...")
    
    for file in files:
        file_path = os.path.join(dataset_path, file)
        df = pd.read_csv(file_path, header=None)
        
        for _, row in df.iterrows():
            try:
                # Column 0 is the label/digit class
                label = row.iloc[0]
                # Columns 1 to 63 (or up to 64) are the landmark coordinates
                features = pd.to_numeric(row.iloc[1:64], errors='coerce')
                
                if not features.isna().any() and len(features) == 63:
                    y_list.append(label)
                    X_list.append(features.values)
            except Exception:
                continue

X = np.array(X_list, dtype=float)
y = np.array(y_list)

print(f"Cleaned features shape (X): {X.shape}")
print(f"Cleaned labels shape (y): {y.shape}")

if X.size == 0 or y.size == 0:
    print("Error: No valid data found. Please check your CSV contents.")
    exit()

# 3. Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
print("Training the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate and save
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Training Complete! Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, 'sign_language_model.pkl')
print("Model saved successfully as 'sign_language_model.pkl'!")