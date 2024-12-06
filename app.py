from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# Load the model, scaler, and label encoders
with open('machine_pred.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('label_encoder_product_id.pkl', 'rb') as le_product_id_file:
    label_encoder_product_id = pickle.load(le_product_id_file)

with open('label_encoder_type.pkl', 'rb') as le_type_file:
    label_encoder_type = pickle.load(le_type_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Validate input keys
        required_keys = [
            "Product ID", "Type", "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque", "Tool wear [min]",
            "TWF", "HDF", "PWF", "OSF", "RNF"
        ]
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing required key: {key}"}), 400

        # Extract and process features
        product_id = data["Product ID"]
        type_ = data["Type"]
        air_temp = float(data["Air temperature [K]"])
        process_temp = float(data["Process temperature [K]"])
        rotational_speed = float(data["Rotational speed [rpm]"])
        torque = float(data["Torque"])
        tool_wear = float(data["Tool wear [min]"])
        twf = int(data["TWF"])
        hdf = int(data["HDF"])
        pwf = int(data["PWF"])
        osf = int(data["OSF"])
        rnf = int(data["RNF"])

        # Encode categorical data
        try:
            product_id_encoded = label_encoder_product_id.transform([product_id])[0]
            type_encoded = label_encoder_type.transform([type_])[0]
        except ValueError as e:
            return jsonify({"error": f"Encoding error: {str(e)}"}), 400

        # Prepare features in the correct order
        features = np.array([[product_id_encoded, type_encoded, air_temp, process_temp,
                              rotational_speed, torque, tool_wear, twf, hdf, pwf, osf, rnf]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        # Recommendation logic based on the prediction
        if prediction == 1:  # Tool Wear Failure
            if tool_wear > 15:
                if rotational_speed > 1600:
                    maintenance_action = (
                        "Severe Tool Wear Failure detected with high rotational speed. "
                        "Replace tools immediately and reduce rotational speed to prevent further wear."
                    )
                elif torque > 45:
                    maintenance_action = (
                        "Severe Tool Wear Failure detected with high torque. Replace tools immediately. "
                        "Inspect for mechanical misalignment or overload issues."
                    )
                else:
                    maintenance_action = (
                        "Severe Tool Wear Failure detected. Replace tools immediately and inspect machining parameters."
                    )
            else:
                if rotational_speed > 1600:
                    maintenance_action = (
                        "Moderate Tool Wear Failure detected. Adjust rotational speed to prevent excessive wear. "
                        "Monitor wear trends and plan tool replacement soon."
                    )
                elif torque < 30:
                    maintenance_action = (
                        "Moderate Tool Wear Failure detected. Torque is lower than usual. Inspect for underutilization or parameter misalignment."
                    )
                else:
                    maintenance_action = (
                        "Moderate Tool Wear Failure detected. Inspect tools and lubrication. "
                        "Perform preventive maintenance and monitor wear trends."
                    )
        elif prediction == 2:  # Heat Dissipation Failure
            if air_temp > 310 or process_temp > 310:
                maintenance_action = (
                    "Heat Dissipation Failure detected. High air or process temperature. Inspect cooling systems and clean heat exchangers."
                )
            else:
                maintenance_action = (
                    "Heat Dissipation Failure detected. Temperatures are within range. Check for blockages or reduced efficiency in the cooling system."
                )
        elif prediction == 3:  # Power Failure
            maintenance_action = (
                "Power Failure detected. Inspect electrical systems and check voltage stability. "
                "Replace worn-out components and perform system calibration."
            )
        elif prediction == 4:  # Overstrain Failure
            maintenance_action = (
                "Overstrain Failure detected. High torque or rotational speed may be the cause. "
                "Inspect mechanical components for wear or misalignment."
            )
        elif prediction == 5:  # Random Failure
            maintenance_action = (
                "Random Failure detected. Perform a comprehensive diagnostic. "
                "Inspect all sensors and recalibrate as necessary."
            )
        else:
            maintenance_action = "No maintenance needed; system is operating normally."

        # Return the prediction and recommendation
        return jsonify({
            "prediction": int(prediction),
            "recommendation": maintenance_action
        })

    except Exception as e:
        # Log the error
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
