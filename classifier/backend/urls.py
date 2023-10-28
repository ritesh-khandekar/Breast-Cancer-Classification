from django.urls import path
from .views import *

urlpatterns = [
 path('predict/', PredictBreastCancerTypeView.as_view(), name='predict_cancer_type'),
 path('', breast_cancer_prediction_view, name='predict_cancer'),
]
