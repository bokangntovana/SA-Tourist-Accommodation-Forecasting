# app/app.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- 1. Load the Model ---
# Load the model once when the application starts
try:
    # Path is relative to where the 'app.py' script is run from (which is the project root)
    MODEL = joblib.load('models/xgboost_occupancy_model.joblib')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file not found. Ensure 'xgboost_occupancy_model.joblib' is in the 'models/' directory.")
    MODEL = None

# --- 2. Define Required Features ---
# These features MUST match the exact features used to train the XGBoost model
REQUIRED_FEATURES = [
    'year', 'quarter', 'month_sin', 'month_cos', 
    'lag_1', 'lag_12', 'rolling_mean_3', 'rolling_std_12', 
    'covid_flag'
]

@app.route('/predict_occupancy', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error': 'Model not available.'}), 500

    try:
        data = request.get_json()
        
        # --- 3. Input Validation ---
        # Check if all necessary features are in the JSON request
        if not data or not all(key in data for key in REQUIRED_FEATURES):
            return jsonify({
                'error': 'Missing required features in the request body.', 
                'required': REQUIRED_FEATURES
            }), 400

        # --- 4. Create DataFrame for Prediction ---
        # Ensure the input DataFrame has the features in the exact training order
        input_data = {feature: [data[feature]] for feature in REQUIRED_FEATURES}
        input_df = pd.DataFrame(input_data)
        
        # --- 5. Predict ---
        prediction = MODEL.predict(input_df)
        
        # --- 6. Return JSON Response ---
        return jsonify({
            'predicted_occupancy_rate': round(float(prediction[0]), 2),
            'unit': '%'
        })

    except Exception as e:
        # Catch any runtime errors during processing
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Run the server. host='0.0.0.0' makes it accessible externally (if deployed), 
    # but locally you access it via 127.0.0.1:5000.
    app.run(host='0.0.0.0', port=5000, debug=True)