from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views import View
from django.shortcuts import render
from sklearn.datasets import load_breast_cancer
import pandas as pd
import json
import numpy as np
import os
import joblib

class PredictBreastCancerTypeView(View):

    @csrf_exempt
    def post(self, request):
        body = request.body
        # Ensure you read and parse the request body only once
        try:
            request_data = json.loads(body.decode('utf-8'))
            features = request_data.get('features', [])

            breast_cancer_data = load_breast_cancer()
            feature_names = breast_cancer_data.feature_names
            # Ensure 'features' is a list with the same number of features as your model
            if len(features) != len(feature_names):
                return JsonResponse({'error': 'Invalid number of features'})

            # Convert features to a numpy array and make predictions
            features = [float(features[feature]) for feature in feature_names]  # Convert feature values to float
            features = np.array(features).reshape(1, -1)
            model_file_path = os.path.join('model', 'model.pkl')
            model = joblib.load(model_file_path)  # Load your trained model here
            prediction = model.predict(features)

            # Return the prediction as a JSON response
            return JsonResponse({'prediction': int(prediction[0])})

        except Exception as e:
            return JsonResponse({'error': str(e)})

def breast_cancer_prediction_view(request):
    # Load the breast cancer dataset to get column names
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)

    # Find the most common values for each feature
    common_values = df.mode().iloc[0].to_dict()

    context = {"common_values": common_values}
    return render(request, "backend/index.html", context)
