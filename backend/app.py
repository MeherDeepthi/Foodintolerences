from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load your saved model
model_path = os.path.join(os.path.dirname(__file__), "severity_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    inputs = np.array(data["inputs"]).reshape(1, -1)
    prediction = model.predict(inputs)[0]  # severity from 0 to 3
    return jsonify({"symptom_severity": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
