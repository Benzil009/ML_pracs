import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("final_dataset.csv")

# Step 1: Convert AQI into categorical classes
def categorize_aqi(aqi):
    if aqi <= 50: return 0  # Good
    elif aqi <= 100: return 1  # Moderate
    elif aqi <= 200: return 2  # Unhealthy (Sensitive)
    elif aqi <= 300: return 3  # Unhealthy
    elif aqi <= 400: return 4  # Very Unhealthy
    else: return 5  # Hazardous

df["AQI_Class"] = df["AQI"].apply(categorize_aqi)

# Step 2: Features & Target
X = df[["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]]
y = df["AQI_Class"]

# Step 3: Train-Test Split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 4: Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train Logistic Regression (no warning now)
model = LogisticRegression(max_iter=3)
model.fit(X_train_scaled, y_train)

# Step 6: Evaluate Model
y_pred = model.predict(X_test_scaled)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Step 7: Manual Testing
# Example input (replace with your own values)
manual_input = pd.DataFrame([{
    "PM2.5": 120.0,
    "PM10": 180.0,
    "NO2": 90.0,
    "SO2": 15.0,
    "CO": 1.2,
    "Ozone": 40.0
}])

manual_input_scaled = scaler.transform(manual_input)
manual_pred = model.predict(manual_input_scaled)[0]

classes = {
    0: "Good",
    1: "Moderate",
    2: "Unhealthy (Sensitive)",
    3: "Unhealthy",
    4: "Very Unhealthy",
    5: "Hazardous"
}

print("\nManual Test Input Prediction:")
print(f"Predicted AQI Category: {classes[manual_pred]}")

from sklearn.metrics import accuracy_score

# After you already have predictions:
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
