import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 1) Load your model and data
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("features.csv")

# 2) Define your model features
model_features = [
    'Lag_1', 'Lag_2', 'Lag_3', 'Lag_12',
    'Month_sin', 'Month_cos',
    'store_mean_sales', 'store_std_sales',
    'rolling_mean_3', 'rolling_std_3',
    'rolling_mean_6', 'rolling_trend', 'sales_to_avg_ratio'
]

@app.route("/stores", methods=["GET"])
def get_stores():
    # Return a list of unique store IDs
    store_ids = sorted(df["Store Number"].unique().astype(int).tolist())
    return jsonify({"stores": store_ids})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    store = int(data["store"])
    weeks = int(data["weeks"])

    # Filter the data for this store
    store_df = df[df["Store Number"] == store].sort_values(by=["Year", "Month"])
    if store_df.empty:
        return jsonify({"error": f"Store {store} not found."}), 404

    # Take the last known row
    current_row = store_df.iloc[-1].copy()

    predictions = []

    # Use the monthly stepping approach
    current_month = int(current_row["Month"])

    for _ in range(weeks):
        X = current_row[model_features].values.reshape(1, -1)
        pred_sales = float(model.predict(X)[0])
        predictions.append(pred_sales)

        # Update lags
        current_row["Lag_3"] = current_row["Lag_2"]
        current_row["Lag_2"] = current_row["Lag_1"]
        current_row["Lag_1"] = pred_sales

        # (Optional) handle Lag_12 after 12 steps

        # Advance month
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_row["Year"] = float(current_row["Year"]) + 1

        current_row["Month"] = float(current_month)
        current_row["Month_sin"] = np.sin(2 * np.pi * current_month / 12)
        current_row["Month_cos"] = np.cos(2 * np.pi * current_month / 12)

        # Update rolling means
        lags = [current_row["Lag_1"], current_row["Lag_2"], current_row["Lag_3"]]
        current_row["rolling_mean_3"] = np.mean(lags)
        current_row["rolling_std_3"] = np.std(lags)

        rolling_6_data = lags + [current_row["rolling_mean_3"]] * 3
        current_row["rolling_mean_6"] = np.mean(rolling_6_data)
        current_row["rolling_trend"] = current_row["rolling_mean_3"] - current_row["rolling_mean_6"]
        current_row["sales_to_avg_ratio"] = pred_sales / (current_row["rolling_mean_3"] + 1e-6)

    return jsonify({
        "prediction": [round(p, 2) for p in predictions],
        "total": round(float(np.sum(predictions)), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
