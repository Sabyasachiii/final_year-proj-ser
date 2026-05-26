from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load model and scaler
model = tf.keras.models.load_model("lstm_fraud_model")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return jsonify({"status": "LSTM Fraud Detection API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = data.get("features")

    # ✅ Validate feature count
    if len(features) != scaler.n_features_in_:
        return jsonify({
            "error": f"Expected {scaler.n_features_in_} features, got {len(features)}"
        }), 400

    # Convert to numpy
    features = np.array(features).reshape(1, -1)

    # Scale
    features_scaled = scaler.transform(features)

    # ✅ Correct LSTM shape: (batch, timesteps, features)
    features_scaled = features_scaled.reshape(
        (features_scaled.shape[0], 1, features_scaled.shape[1])
    )

    # Predict
    prediction = model.predict(features_scaled, verbose=0)[0][0]

    return jsonify({
        "prediction": float(prediction),
        "result": "Fraud" if prediction > 0.5 else "Not Fraud"
    })

if __name__ == "__main__":
    app.run(debug=True)
