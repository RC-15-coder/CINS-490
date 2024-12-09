from django.db import models
from django.contrib.auth.models import User

class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Links each prediction to a specific user
    prediction_result = models.CharField(max_length=20)  # Store the result as "Diabetic" or "Non-Diabetic"
    timestamp = models.DateTimeField(auto_now_add=True)  # Automatically adds the time of prediction

    def __str__(self):
        return f"{self.user.username} - {self.prediction_result} - {self.timestamp}"
