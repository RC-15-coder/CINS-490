Overview
The Diabetes Prediction Web Application is a data-driven tool designed to predict the likelihood of diabetes based on user inputs such as glucose levels, blood pressure, BMI, and other health indicators. This project leverages machine learning and data preprocessing techniques to provide accurate predictions while offering an interactive and user-friendly interface.
The goal of this project was to build a complete end-to-end solution for predicting diabetes, including a robust backend for data processing and storage, a trained ML model, and a frontend for user interaction.

Why This Project?
Diabetes is a growing health concern worldwide, and early detection plays a crucial role in its management. This project:

Provides an easy-to-use platform for individuals to assess their risk of diabetes.
Showcases the power of machine learning in solving real-world problems.

Workflow of the Diabetes Prediction Web Application:

1. User Interaction
User Registration/Login:

Users start by registering an account or logging in if they already have one.
Once logged in, users access the dashboard and the prediction form.

Input Health Metrics:

On the prediction page, users provide their health metrics, such as:
Gender, Glucose levels, Blood Pressure, Skin Thickness, Insulin levels, BMI (Body Mass Index).
Diabetes Pedigree Function (a measure of genetic influence), Age.
These inputs are collected via an HTML form and sent to the backend for processing.

2. Data Preprocessing
Standardization:

User inputs are preprocessed using the scaler.pkl file saved during training.
The scaler ensures all input values are scaled to match the data format used during model training.
This is crucial because the model was trained on scaled data, and raw inputs may lead to inaccurate predictions.

Pregnancy Adjustment:

If the user is male, the Pregnancies feature is automatically set to 0, as it’s not applicable.

3. Model Prediction
Loading the Model:

The pre-trained LightGBM Classifier (best_lgb_model.pkl) is loaded into memory.

Prediction Process:

The preprocessed inputs are fed into the model.
The model predicts a probability of the user being diabetic.

Threshold Application:

The predicted probability is compared against the optimal threshold (e.g., 0.16).
If the probability exceeds the threshold, the user is classified as "Diabetic", otherwise "Non-Diabetic".
This threshold ensures a balance between sensitivity (identifying diabetics correctly) and specificity (avoiding false positives).

4. Prediction Storage
Storing Results in SQLite:
Each prediction result is stored in the backend SQLite database.
The logged-in user's ID is associated with the prediction, ensuring personal tracking of history.

5. Prediction Dashboard
Viewing Prediction History:
Users can view their past predictions on the dashboard page.
The dashboard displays:
Prediction Result: Indicates whether the user was classified as "Diabetic" or "Non-Diabetic and The exact date of the prediction.

7. Error Handling
If inputs are invalid (e.g., missing values, non-numeric inputs), an error message is displayed.

Backend Workflow Summary
The user inputs data through the frontend form.
The data is sent to the backend via a POST request.
The backend:
Preprocesses the data (scaling and feature adjustment).
Uses the LightGBM model to predict diabetes probability.
Applies the optimal threshold to determine the final result.
Saves the result to the SQLite database for logged-in users.
The prediction result is displayed on the result page and stored for future reference.

Technologies Used
Languages and Libraries
Backend: Python, Django, SQLite
Frontend: HTML, CSS, Bootstrap
Machine Learning: LightGBM, Scikit-learn, Pandas, NumPy
Data Preprocessing: Local Outlier Factor, StandardScaler
Tools: PythonAnywhere (for hosting)

Key Files
views.py: Handles backend logic, including data preprocessing and prediction.
model_training.py: Trains the LightGBM model and saves it as a .pkl file.
data_cleaning.py: Cleans and preprocesses the dataset (diabetes.csv).


Steps to Set Up and Run the Backend
1. Clone the Repository:
git clone https://github.com/RC-15-coder/CINS-490.git
cd CINS-490

2. Create a Virtual Environment
To avoid dependency conflicts, create a virtual environment:
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows

3. Install Dependencies
Ensure all required packages are installed. This is achieved using the requirements.txt file
pip install -r requirements.txt

4. Run the website from the backend if pythonanywhere.com is down or not working properly:
python manage.py runserver
This will output a URL on the console like:
Starting development server at http://127.0.0.1:8000/

If you want to see the output of the data cleaning and model training script run (optional):
Data Cleaning:
python data_cleaning.py

Model Training:
python model_training.py

These scripts will generate the required scaler.pkl and best_lgb_model.pkl files in the appropriate directories.


To see the Website go to this link:
https://raghavchandna.pythonanywhere.com/ 


For testing the results on the website:
Already tested user:
Username: Jay
Password: jay@123456

Username: Rachel 
Password: rachel@123456

Health metrics from the actual dataset to test the results:
1 - pregnancies
89 - glucose
66 - blood pressure 
23 - skin thickness
94 - insulin 
28.1 - BMI
0.167 -  Diabetes pedigree function (Family History)
21 - Age
0 - Should be non-diabetic


This result can also be tested for males because of 0 pregnancies.
0 
137
40
35
168
43.1
2.288
33 
1 - should be diabetic 


1
103
30
38
83
43.3
0.183
33
0 - Non-diabetic


3
78
50
32
88
31
0.248
26
1 - Diabetic
