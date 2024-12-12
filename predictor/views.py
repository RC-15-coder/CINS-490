from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from django.contrib import messages
from django.http import JsonResponse
import joblib
import os
import pandas as pd
from .models import Prediction


def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()          # Save the new user
            login(request, user)        # Log in the user
            messages.success(request, "Registration successful! You are now logged in.")
            return redirect('dashboard')  # Redirect to the dashboard
    else:
        form = UserCreationForm()  # Initialize an empty form for GET requests

    # Render the registration template with the form
    return render(request, 'predictor/register.html', {'form': form})


@login_required
def predict_diabetes(request):
    if request.method == 'POST':
        try:
            # Extract input values from POST data
            gender = request.POST.get('gender', '').lower()
            pregnancies = float(request.POST.get('pregnancies', 0)) if gender != 'male' else 0.0
            glucose = float(request.POST.get('glucose', 0))
            blood_pressure = float(request.POST.get('blood_pressure', 0))
            skin_thickness = float(request.POST.get('skin_thickness', 0))
            insulin = float(request.POST.get('insulin', 0))
            bmi = float(request.POST.get('bmi', 0))
            diabetes_pedigree_function = float(request.POST.get('diabetes_pedigree_function', 0))
            age = float(request.POST.get('age', 0))
        except ValueError:
            # Handle invalid input conversions
            messages.error(request, "Invalid input. Please enter valid numbers.")
            return render(request, "predictor/predict.html")

        # Create Input DataFrame
        feature_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        input_values = [
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, diabetes_pedigree_function, age
        ]
        input_data = pd.DataFrame([input_values], columns=feature_columns)

        # Load the Trained Model and Scaler
        model_path = os.path.join('predictor', 'ml_model', 'best_lgb_model.pkl')
        scaler_path = os.path.join('predictor', 'ml_model', 'scaler.pkl')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return render(request, "predictor/error.html", {"error": "Model or scaler file not found!"})

        # Load the model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Scale the Input Data
        input_data_scaled = scaler.transform(input_data)

        # Predict Probabilities
        y_pred_proba = model.predict_proba(input_data_scaled)[:, 1]

        # Apply the Optimal Threshold for Classification
        optimal_threshold = 0.16 
        prediction = (y_pred_proba >= optimal_threshold).astype(int)

        # Interpret the Prediction
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

        # Save the Prediction to the Database
        Prediction.objects.create(
            user=request.user,              # Associate the prediction with the logged-in user
            prediction_result=result        # Save the prediction result
        )

        # Render the Result Page 
        return render(request, "predictor/result.html", {"result": result})

    else:
        # For GET requests, render the prediction form
        return render(request, "predictor/predict.html")


@login_required
def user_dashboard(request):
    user_predictions = Prediction.objects.filter(user=request.user).order_by('-timestamp')

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        predictions = [
            {
                "timestamp": prediction.timestamp.isoformat(),
                "prediction_result": prediction.prediction_result
            }
            for prediction in user_predictions
        ]
        return JsonResponse({"predictions": predictions})
    else:
        # Handle Standard Requests
        return render(request, "predictor/dashboard.html", {"predictions": user_predictions})
