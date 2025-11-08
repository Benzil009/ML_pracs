Experiment 1: Exploratory Data Analysis

from google.colab import files import pandas as pd
uploaded = files.upload() file_name = list(uploaded.keys())[0]
# Load the dataset
df = pd.read_csv(file_name)
# Display the first 5 rows
print("First 5 rows of the dataframe:") print(df.head())
# Display the summary of the dataframe print("\nSummary of the dataframe:") df.info()
# Drop the 'id' column df = df.drop('id', axis=1)
# Display the updated columns
print("Columns after dropping 'id' column:", df.columns)
# Impute numerical columns with the median for col in ['trestbps', 'chol', 'thalch', 'oldpeak']:
df[col] = df[col].fillna(df[col].median())
# Impute categorical columns with 'unknown' and convert to category type for col in ['restecg', 'slope', 'ca', 'thal']:
if col == 'ca':
df[col] = df[col].astype(str)
df[col] = df[col].replace('nan', 'unknown') else:
df[col] = df[col].fillna('unknown')
# Impute boolean-like columns with the mode for col in ['fbs', 'exang']:
df[col] = df[col].fillna(df[col].mode()[0])
# Convert data types
df['ca'] = df['ca'].astype('category')
df['fbs'] = df['fbs'].astype(bool)
df['exang'] = df['exang'].astype(bool)
# Check the data types and missing values again print("\nUpdated summary of the dataframe:") df.info()
import matplotlib.pyplot as plt import seaborn as sns
# Create a new binary target variable 'target' df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
# Plot the distribution of the new binary target variable plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df)
plt.title('Distribution of the New Target Variable (Binary)') plt.xlabel('Heart Disease Presence (0: No, 1: Yes)') plt.ylabel('Count')
plt.show()
# Print the value counts
print("Value counts for the new 'target' variable:") print(df['target'].value_counts())
from google.colab import files import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# This will prompt you to upload the file uploaded = files.upload()
file_name = list(uploaded.keys())[0]
# Load the dataset
df = pd.read_csv(file_name)
print("Data loaded successfully. Performing preprocessing...") #
# Drop the 'id' column as it is not useful for analysis df = df.drop('id', axis=1)
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0) df = df.drop('num', axis=1)
for col in ['trestbps', 'chol', 'thalch', 'oldpeak', 'age']:
df[col] = df[col].fillna(df[col].median())
 ['restecg', 'slope', 'ca', 'thal']:
if col == 'ca':
df[col] = df[col].astype(str).replace('nan', 'unknown') else:
df[col] = df[col].fillna('unknown')
for col in ['fbs', 'exang']:
df[col] = df[col].fillna(df[col].mode()[0])
df['ca'] = df['ca'].astype('category')
df['fbs'] = df['fbs'].astype(bool)
df['exang'] = df['exang'].astype(bool)
# Separate numerical and categorical columns numerical_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
categorical_cols = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Normalize numerical features scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# One-hot encode categorical features
df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Save the processed DataFrame to a new CSV file output_file = 'preprocessed.csv' df_processed.to_csv(output_file, index=False)

print("\nAll preprocessing steps are complete.")
print(f"The final preprocessed data has been saved to '{output_file}'.")

# Display the first 5 rows of the final dataframe
print("\nFirst 5 rows of the final preprocessed dataframe:") print(df_processed.head())
