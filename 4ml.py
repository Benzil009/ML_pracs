import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

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

# Step 4: Scale Features (needed for LR & SVM, not strictly for Tree)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------
# Decision Tree
# -----------------------
dt_model = DecisionTreeClassifier(random_state=42, max_depth=8)
dt_model.fit(X_train, y_train)  # Trees don't need scaling
y_pred_dt = dt_model.predict(X_test)

print("\n=== Decision Tree ===")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# -----------------------
# Support Vector Machine
# -----------------------
svm_model = SVC(kernel="rbf", C=10, gamma="scale")
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

print("\n=== Support Vector Machine (SVM) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
