import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
file_path = "Student-Employability-Datasets.xlsx"  
df = pd.read_excel(file_path)
print(df.head())

selected_features = [
    "MANNER OF SPEAKING", 
    "SELF-CONFIDENCE", 
    "ABILITY TO PRESENT IDEAS", 
    "COMMUNICATION SKILLS", 
    "MENTAL ALERTNESS"
]
target_column = "CLASS" 

label_encoder = LabelEncoder()
df[target_column] = label_encoder.fit_transform(df[target_column])

X = df[selected_features]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "employability_model_selected.joblib")
joblib.dump(label_encoder, "label_encoder_fixed.joblib")

print("Model and encoder saved successfully!")
