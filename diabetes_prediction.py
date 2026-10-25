import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("diabetes.csv")

print(data.head())

print("\nMissing values:")
print(data.isnull().sum())

print("\nChecking for invalid values:")

print("Glucose <= 0:", (data["Glucose"] <= 0).sum())
print("BloodPressure <= 0:", (data["BloodPressure"] <= 0).sum())
print("BMI <= 0:", (data["BMI"] <= 0).sum())
print("Age <= 0:", (data["Age"] <= 0).sum())

# Replace impossible zero values with NaN
columns_with_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

data[columns_with_zero] = data[columns_with_zero].replace(0, np.nan)

# Fill missing values with the median
for column in columns_with_zero:
    data[column] = data[column].fillna(data[column].median())

print("\nData cleaned successfully!")

# Separate features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model trained successfully!")
print("Accuracy:", accuracy)

# Make prediction for a new patient

print("\nEnter patient details for prediction:")

pregnancies = float(input("Pregnancies: "))
glucose = float(input("Glucose: "))
blood_pressure = float(input("Blood Pressure: "))
skin_thickness = float(input("Skin Thickness: "))
insulin = float(input("Insulin: "))
bmi = float(input("BMI: "))
diabetes_pedigree = float(input("Diabetes Pedigree Function: "))
age = float(input("Age: "))

# Create patient data
patient = pd.DataFrame([[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
]], columns=X.columns)

# Scale patient data
patient_scaled = scaler.transform(patient)

# Make prediction
prediction = model.predict(patient_scaled)

if prediction[0] == 1:
    print("\nPrediction: Diabetes")
else:
    print("\nPrediction: No Diabetes")