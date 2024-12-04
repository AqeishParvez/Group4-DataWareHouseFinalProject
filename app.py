from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from pymongo import MongoClient
from datetime import datetime
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the saved Random Forest model
model = joblib.load('tuned_random_forest_model.pkl')

# Initialize MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['prediction_db']
collection = db['predictions']

# Define the expected features
expected_features = [
    'MANOEUVER_Going Ahead', 'DRIVACT_Driving Properly', 'IMPACTYPE_Pedestrian Collisions',
    'INVTYPE_Driver', 'DRIVCOND_Unknown', 'VEHTYPE_Automobile, Station Wagon',
    'HOUR', 'YEAR', 'INVTYPE_Pedestrian', 'INVTYPE_Passenger', 'VEHTYPE_Other',
    'INVAGE_unknown', 'INITDIR_West', 'DRIVCOND_Normal', 'INITDIR_South',
    'ROAD_CLASS_Major Arterial', 'INITDIR_North'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("\nReceived Data:", data)

    # Mapping of frontend keys to backend expected keys
    key_mapping = {
        'MANOEUVER_Going_Ahead': 'MANOEUVER_Going Ahead',
        'DRIVACT_Driving_Properly': 'DRIVACT_Driving Properly',
        'IMPACTYPE_Pedestrian_Collisions': 'IMPACTYPE_Pedestrian Collisions',
        'INVTYPE_Driver': 'INVTYPE_Driver',
        'DRIVCOND_Unknown': 'DRIVCOND_Unknown',
        'VEHTYPE_Automobile_Station_Wagon': 'VEHTYPE_Automobile, Station Wagon',
        'HOUR': 'HOUR',
        'YEAR': 'YEAR',
        'INVTYPE_Pedestrian': 'INVTYPE_Pedestrian',
        'INVTYPE_Passenger': 'INVTYPE_Passenger',
        'VEHTYPE_Other': 'VEHTYPE_Other',
        'INVAGE_unknown': 'INVAGE_unknown',
        'INITDIR_West': 'INITDIR_West',
        'DRIVCOND_Normal': 'DRIVCOND_Normal',
        'INITDIR_South': 'INITDIR_South',
        'ROAD_CLASS_Major_Arterial': 'ROAD_CLASS_Major Arterial',
        'INITDIR_North': 'INITDIR_North'
    }

    # Transform received data to match the expected format
    transformed_data = {key_mapping[key]: value for key, value in data.items() if key in key_mapping}
    print("\nTransformed Data:", transformed_data)

    # Ensure all expected features are present
    if not all(feature in transformed_data for feature in expected_features):
        return jsonify({'error': 'Missing required features!'}), 400

    try:
        # Extract features and reshape for model input
        input_data = np.array([transformed_data[feature] for feature in expected_features]).reshape(1, -1)
        print("\nInput Data for Model:", input_data)  # Log input data

        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data).max()

        # Save to MongoDB
        prediction_entry = {
            "input": data,
            "prediction": int(prediction),
            "probability": float(probability),
            "timestamp": datetime.now().isoformat()
        }

        # Save prediction to MongoDB
        collection.insert_one(prediction_entry)
        print("\nPrediction saved to MongoDB:", prediction_entry)


        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability)
        })
    except Exception as e:
        print("\nError processing request:", str(e))
        return jsonify({'error': 'Error processing request', 'details': str(e)}), 500

@app.route('/predictions', methods=['GET'])
def get_predictions():
    try:
        predictions = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB `_id`
        return jsonify(predictions)
    except Exception as e:
        print("\nError fetching predictions:", str(e))
        return jsonify({'error': 'Error fetching predictions', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)