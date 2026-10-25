import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load and combine all CSV files from the dataset folder
dataset_path = 'dataset'
all_files = glob.glob(os.path.join(dataset_path, "*.csv"))

print(f"Found {len(all_files)} dataset files. Loading...")

df_list = []
for filename in all_files:
    # header=None assumes there are no column names in the CSV
    df = pd.read_csv(filename, header=None)
    df_list.append(df)

# Combine into a single dataframe
data = pd.concat(df_list, ignore_index=True)

# Drop any rows that might contain text/headers accidentally
data = data.dropna()

# 2. Separate features (landmark coordinates) and labels (signs)
# Features = all columns except the last one, converted to numeric values
X = data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
# Labels = the last column
y = data.iloc[:, -1]

# Drop any rows that failed conversion
combined = pd.concat([X, y], axis=1).dropna()
X = combined.iloc[:, :-1]
y = combined.iloc[:, -1]

print(f"Cleaned dataset shape: {X.shape}")

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