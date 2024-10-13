import pandas as pd
import numpy as np  
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r'C:\CINS 490 Project\diabetes.csv')

# Replace 0 with np.nan in the columns where 0 is invalid
invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[invalid_cols] = df[invalid_cols].replace(0, np.nan)

# Use Mean or Median Imputation for Glucose, BloodPressure, and BMI
df['Glucose'] = df['Glucose'].fillna(df['Glucose'].median())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
df['BMI'] = df['BMI'].fillna(df['BMI'].median())

# Use KNN Imputer for Insulin and SkinThickness
imputer = KNNImputer(n_neighbors=5)
df[['Insulin', 'SkinThickness']] = imputer.fit_transform(df[['Insulin', 'SkinThickness']])

# Separate features and labels
X = df.drop('Outcome', axis=1)  # Features (all columns except 'Outcome')
y = df['Outcome']  # Labels (the target variable)

# Split the dataset into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Save the cleaned and balanced data to CSV files
X_train_smote.to_csv('X_train_smote.csv', index=False)
y_train_smote.to_csv('y_train_smote.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print(f"Before SMOTE: {sum(y_train == 1)} diabetic, {sum(y_train == 0)} non-diabetic")
print(f"After SMOTE: {sum(y_train_smote == 1)} diabetic, {sum(y_train_smote == 0)} non-diabetic")

# Check for missing values in the cleaned dataset
print("\nMissing values in each column after cleaning:")
print(df.isnull().sum())

# Check dataset balance after cleaning
print("\nDataset balance (Outcome counts):")
print(df['Outcome'].value_counts())
