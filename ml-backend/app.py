import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the dataset with full feature set
df = pd.read_csv("features.csv")  # This must include all 13 model features + Store Number

# Define the feature columns
model_features = [
    'Lag_1', 'Lag_2', 'Lag_3', 'Lag_12',
    'Month_sin', 'Month_cos',
    'store_mean_sales', 'store_std_sales',
    'rolling_mean_3', 'rolling_std_3',
    'rolling_mean_6', 'rolling_trend', 'sales_to_avg_ratio'
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    store = int(data["store"])

    # Load features
    df = pd.read_csv("features.csv")

    # Filter for store
    store_df = df[df['Store Number'] == store].sort_values(by=['Year', 'Month'])

    if store_df.empty:
        return jsonify({"error": f"Store {store} not found in the dataset."}), 404

    row = store_df.iloc[-1]  # Last known row

    features = row[[
        'Lag_1', 'Lag_2', 'Lag_3', 'Lag_12',
        'Month_sin', 'Month_cos',
        'store_mean_sales', 'store_std_sales',
        'rolling_mean_3', 'rolling_std_3',
        'rolling_mean_6', 'rolling_trend', 'sales_to_avg_ratio'
    ]].values.reshape(1, -1)

    prediction = model.predict(features)
    return jsonify({"prediction": prediction.tolist()})


if __name__ == "__main__":
    app.run(debug=True)
