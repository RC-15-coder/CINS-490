import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os  

# Load the cleaned and balanced data
X_train = pd.read_csv('X_train_smote.csv')
y_train = pd.read_csv('y_train_smote.csv')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use
scaler_path = os.path.join('predictor', 'ml_model', 'scaler.pkl')
joblib.dump(scaler, scaler_path)

# Create a Random Forest classifier 
rf_model = RandomForestClassifier(
    n_estimators=200,   # Number of trees
    max_depth=20,       # Maximum depth of the trees
    min_samples_split=5,  # Minimum number of samples required to split a node
    min_samples_leaf=2,   # Minimum number of samples required to be a leaf node
    max_features='sqrt',  # Number of features to consider when looking for the best split
    bootstrap=True,       # Whether bootstrap samples are used when building trees
    random_state=42       # Ensures reproducibility
)

# Fit Randomized Search on training data
rf_model.fit(X_train_scaled, y_train.values.ravel())

# Make predictions on test data
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Save the Random Forest model inside ml folder
model_path = os.path.join('predictor', 'ml_model', 'best_rf_model.pkl')
joblib.dump(rf_model, model_path)
