from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_diabetes, name='predict_diabetes'),  # Route for the form
    path('result/', views.show_result, name='show_result'),     # Route for displaying result
]
