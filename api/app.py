# ============================================================
# BankGuard - Flask API
# Step 2: Serve the ML model via REST API
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS  # allows our frontend (Step 3) to talk to this API
import joblib
import numpy as np
import os
from datetime import datetime
import random

# ============================================================
# STEP 1: INITIALIZE THE APP
# ============================================================
# Flask is a lightweight web framework
# CORS lets browsers call our API from HTML files

app = Flask(__name__)
CORS(app)

# ============================================================
# STEP 2: LOAD THE SAVED MODEL
# ============================================================
# We load the model ONCE when the server starts
# (Not on every request — that would be very slow)

print("Loading fraud detection model...")

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH):
    print("model.pkl not found! Run train_model.py first.")
    model = None
    scaler = None
else:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model loaded successfully!\n")

# ============================================================
# In-memory transaction log (stores last 50 predictions)
# In a real app, this would be a database
# ============================================================
transaction_log = []

# ============================================================
# ROUTE 1: Health Check
# ============================================================
# Visit http://localhost:5000/ to confirm the server is running

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "BankGuard API is running",
        "model_loaded": model is not None,
        "endpoints": ["/predict", "/transactions", "/stats"]
    })

# ============================================================
# ROUTE 2: /predict  — The main endpoint
# ============================================================
# Accepts a JSON body with transaction features
# Returns fraud probability and a decision

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 500

    try:
        # Get JSON data sent from the frontend
        data = request.get_json()

        # Extract the "Amount" field (user provides this)
        amount = float(data.get("amount", 0))

        # Scale the amount the same way we did during training
        amount_scaled = scaler.transform([[amount]])[0][0]

        # The other 27 features (V1-V27) are auto-generated here
        # In a real banking system, these come from the transaction itself
        # For our demo, we simulate them based on the amount
        np.random.seed(int(amount * 100) % 10000)
        v_features = np.random.randn(28)

        # Combine into one feature array (28 features total)
        features = np.array([[amount_scaled] + list(v_features)])

        # Ask the model: fraud or not?
        prediction = model.predict(features)[0]           # 0 or 1
        probability = model.predict_proba(features)[0]    # [prob_legit, prob_fraud]

        fraud_probability = round(float(probability[1]) * 100, 2)
        is_fraud = bool(prediction == 1)

        # Build result object
        result = {
            "transaction_id": f"TXN{random.randint(100000, 999999)}",
            "amount": amount,
            "is_fraud": is_fraud,
            "fraud_probability": fraud_probability,
            "risk_level": (
                "HIGH"   if fraud_probability > 70 else
                "MEDIUM" if fraud_probability > 30 else
                "LOW"
            ),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "merchant": data.get("merchant", "Unknown"),
            "category": data.get("category", "General")
        }

        # Save to our in-memory log (keep last 50)
        transaction_log.append(result)
        if len(transaction_log) > 50:
            transaction_log.pop(0)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ============================================================
# ROUTE 3: /transactions — View transaction history
# ============================================================

@app.route("/transactions", methods=["GET"])
def get_transactions():
    # Return transactions newest first
    return jsonify(list(reversed(transaction_log)))


# ============================================================
# ROUTE 4: /stats — Summary statistics for the dashboard
# ============================================================

@app.route("/stats", methods=["GET"])
def get_stats():
    if not transaction_log:
        return jsonify({
            "total": 0,
            "fraud_count": 0,
            "legit_count": 0,
            "fraud_rate": 0,
            "total_amount": 0
        })

    total = len(transaction_log)
    fraud_count = sum(1 for t in transaction_log if t["is_fraud"])
    legit_count = total - fraud_count
    total_amount = sum(t["amount"] for t in transaction_log)
    fraud_rate = round((fraud_count / total) * 100, 2)

    return jsonify({
        "total": total,
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "fraud_rate": fraud_rate,
        "total_amount": round(total_amount, 2)
    })


# ============================================================
# START THE SERVER
# ============================================================

if __name__ == "__main__":
    print("Starting BankGuard API on http://localhost:5000")
    print("Press CTRL+C to stop\n")
    app.run(debug=True, port=5000)