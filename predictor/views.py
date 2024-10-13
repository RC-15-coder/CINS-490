from django.shortcuts import render
from django import forms
import joblib
import os
import numpy as np
import pandas as pd 

# a form class for user inputs
class DiabetesForm(forms.Form):
    pregnancies = forms.FloatField(label="Pregnancies")
    glucose = forms.FloatField(label="Glucose Level")
    blood_pressure = forms.FloatField(label="Blood Pressure")
    skin_thickness = forms.FloatField(label="Skin Thickness")
    insulin = forms.FloatField(label="Insulin Level")
    bmi = forms.FloatField(label="BMI")
    diabetes_pedigree_function = forms.FloatField(label="Diabetes Pedigree Function")
    age = forms.FloatField(label="Age")

# View to handle the form and make prediction
def predict_diabetes(request):
    if request.method == 'POST':
        form = DiabetesForm(request.POST)
        
        if form.is_valid():
            # Get the cleaned data from the form
            input_data = pd.DataFrame([[
                form.cleaned_data['pregnancies'],
                form.cleaned_data['glucose'],
                form.cleaned_data['blood_pressure'],
                form.cleaned_data['skin_thickness'],
                form.cleaned_data['insulin'],
                form.cleaned_data['bmi'],
                form.cleaned_data['diabetes_pedigree_function'],
                form.cleaned_data['age']
            ]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

            # Load the saved Random Forest model
            model_path = os.path.join('predictor', 'ml_model', 'best_rf_model.pkl')
            scaler_path = os.path.join('predictor', 'ml_model', 'scaler.pkl')

            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
            except FileNotFoundError:
                return render(request, "predictor/error.html", {"error": "Model or scaler file not found!"})

            # Applied the same scaler to the input data
            input_data_scaled = scaler.transform(input_data)

            # Make prediction using the trained model
            prediction = model.predict(input_data_scaled)
            
            # Interpret the prediction result
            if prediction[0] == 1:
                result = "Diabetic"
            else:
                result = "Non-Diabetic"
            
            # Redirect to result page with the prediction
            return render(request, "predictor/result.html", {"result": result})
    
    else:
        form = DiabetesForm()

    return render(request, "predictor/predict.html", {"form": form})

# View to display the result
def show_result(request):
    return render(request, "predictor/result.html")
