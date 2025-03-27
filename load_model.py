import os
import json
import logging
from flask import Flask, request, jsonify
import numpy as np
from setup_env import setup_environment

# Set up the environment
config = setup_environment()
model_path = config['model_path']
port = config['port']

# Initialize Flask app
app = Flask(__name__)

# Load the model
try:
    # Load model based on file type
    import joblib
    model = joblib.load(model_path)
    logging.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.json
        
        # Process input data - convert to appropriate format for your model
        # This is a simplified example - adjust based on your model's requirements
        if isinstance(data, dict) and 'features' in data:
            features = np.array(data['features'])
            
            # Make prediction
            prediction = model.predict(features)
            
            # Convert numpy types to Python native types for JSON serialization
            if isinstance(prediction, np.ndarray):
                prediction = prediction.tolist()
            
            return jsonify({
                'status': 'success',
                'prediction': prediction
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Invalid input format. Expected {features: [...]}'
            }), 400
            
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'ML Model API is running. Use POST /predict for predictions.'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
