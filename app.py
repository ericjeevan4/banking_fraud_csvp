from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------

MODEL_PATH = "models/lightgbm_fraud_pipeline.joblib"

best_model = joblib.load(MODEL_PATH)

# ---------------------------------------------------------
# FEATURES USED DURING TRAINING
# ---------------------------------------------------------

trained_features = [
    "TransactionAmount",
    "TransactionType",
    "Location",
    "Channel",
    "CustomerAge",
    "CustomerOccupation",
    "AccountBalance",
    "AnnualIncome",
    "CurrentAddressMonthCount",
    "PreviousAddressMonthCount"
]

categorical_features = [
    "TransactionType",
    "Location",
    "Channel",
    "CustomerOccupation"
]

# ---------------------------------------------------------
# HTML
# ---------------------------------------------------------

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Bank Account Fraud Detection</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            margin: 0;
            padding: 40px;
        }

        .container {
            max-width: 1100px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        h1 {
            text-align: center;
            margin-bottom: 10px;
        }

        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }

        .upload-box {
            border: 2px dashed #888;
            padding: 30px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 25px;
        }

        input[type="file"] {
            margin: 15px;
        }

        button {
            padding: 12px 25px;
            border: none;
            border-radius: 6px;
            background: #222;
            color: white;
            cursor: pointer;
            font-size: 16px;
        }

        button:hover {
            background: #444;
        }

        .error {
            background: #ffe5e5;
            color: #b00020;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .success {
            background: #e8f5e9;
            color: #1b5e20;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .summary {
            display: flex;
            gap: 20px;
            margin: 25px 0;
        }

        .card {
            flex: 1;
            padding: 20px;
            background: #f1f3f5;
            border-radius: 10px;
            text-align: center;
        }

        .card h2 {
            margin: 5px 0;
        }

        .account {
            margin-top: 25px;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
        }

        .account h2 {
            margin-top: 0;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        th, td {
            padding: 10px;
            border: 1px solid #ddd;
            text-align: left;
        }

        th {
            background: #f0f0f0;
        }

        .fraud {
            color: #c62828;
            font-weight: bold;
        }

        .no-fraud {
            color: #2e7d32;
            font-weight: bold;
        }
    </style>
</head>

<body>

<div class="container">

    <h1>Bank Account Fraud Detection</h1>

    <div class="subtitle">
        Upload a CSV file to detect fraudulent transactions
    </div>

    <div class="upload-box">

        <form method="POST" enctype="multipart/form-data">

            <input
                type="file"
                name="file"
                accept=".csv"
                required
            >

            <br>

            <button type="submit">
                Detect Fraud
            </button>

        </form>

    </div>

    {% if error %}

        <div class="error">
            {{ error }}
        </div>

    {% endif %}


    {% if result %}

        <div class="summary">

            <div class="card">
                <div>Total Transactions</div>
                <h2>{{ total_transactions }}</h2>
            </div>

            <div class="card">
                <div>Predicted Fraud</div>
                <h2>{{ fraud_count }}</h2>
            </div>

            <div class="card">
                <div>Predicted NonFraud</div>
                <h2>{{ nonfraud_count }}</h2>
            </div>

        </div>


        {% if fraud_count == 0 %}

            <div class="success">
                NO FRAUD TRANSACTIONS DETECTED.
            </div>

        {% else %}

            <h2>Detected Fraud Accounts</h2>

            {% for account in accounts %}

                <div class="account">

                    <h2>
                        Account ID: {{ account.account_id }}
                    </h2>

                    <p>
                        <strong>Total Transactions:</strong>
                        {{ account.total_transactions }}
                    </p>

                    <p>
                        <strong>User Name:</strong>
                        {{ account.username }}
                    </p>

                    <p>
                        <strong>Email ID:</strong>
                        {{ account.email }}
                    </p>

                    <p>
                        <strong>Account Balance:</strong>
                        {{ account.balance }}
                    </p>

                    <h3>Fraud Transactions</h3>

                    <table>

                        <tr>
                            <th>Transaction ID</th>
                            <th>Transaction Date</th>
                            <th>Fraud Probability</th>
                        </tr>

                        {% for transaction in account.transactions %}

                        <tr>

                            <td>
                                {{ transaction.transaction_id }}
                            </td>

                            <td>
                                {{ transaction.transaction_date }}
                            </td>

                            <td class="fraud">
                                {{ transaction.probability }}
                            </td>

                        </tr>

                        {% endfor %}

                    </table>

                </div>

            {% endfor %}

        {% endif %}

    {% endif %}

</div>

</body>
</html>
"""


# ---------------------------------------------------------
# HOME / CSV UPLOAD
# ---------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "GET":
        return render_template_string(HTML)

    # -----------------------------------------------------
    # CHECK FILE
    # -----------------------------------------------------

    if "file" not in request.files:
        return render_template_string(
            HTML,
            error="Please select a CSV file."
        )

    file = request.files["file"]

    if file.filename == "":
        return render_template_string(
            HTML,
            error="Please select a CSV file."
        )

    if not file.filename.lower().endswith(".csv"):
        return render_template_string(
            HTML,
            error="Only CSV files are allowed."
        )

    # -----------------------------------------------------
    # READ CSV
    # -----------------------------------------------------

    try:

        test_csv = pd.read_csv(file)

    except Exception as e:

        return render_template_string(
            HTML,
            error=f"Unable to read CSV file: {str(e)}"
        )

    # -----------------------------------------------------
    # CHECK REQUIRED TRAINED FEATURES
    # -----------------------------------------------------

    missing_features = [
        col
        for col in trained_features
        if col not in test_csv.columns
    ]

    if missing_features:

        missing_text = ", ".join(missing_features)

        return render_template_string(
            HTML,
            error=(
                "Required trained features are missing: "
                + missing_text
            )
        )

    # -----------------------------------------------------
    # CHECK OUTPUT COLUMNS
    # -----------------------------------------------------

    required_output_columns = [
        "AccountID",
        "TransactionID",
        "TransactionDate",
        "UserName",
        "Email",
        "AccountBalance"
    ]

    missing_output_columns = [
        col
        for col in required_output_columns
        if col not in test_csv.columns
    ]

    if missing_output_columns:

        missing_text = ", ".join(missing_output_columns)

        return render_template_string(
            HTML,
            error=(
                "Required output columns are missing: "
                + missing_text
            )
        )

    # -----------------------------------------------------
    # PREPARE ONLY TRAINED FEATURES
    # -----------------------------------------------------

    X_prediction = test_csv[trained_features].copy()

    # Convert categorical features to string
    for col in categorical_features:
        X_prediction[col] = X_prediction[col].astype(str)

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    try:

        predictions = best_model.predict(X_prediction)

        probabilities = best_model.predict_proba(
            X_prediction
        )[:, 1]

    except Exception as e:

        return render_template_string(
            HTML,
            error=f"Prediction error: {str(e)}"
        )

    # -----------------------------------------------------
    # CONVERT PREDICTION TO LABEL
    # -----------------------------------------------------

    prediction_labels = np.where(
        predictions == 1,
        "Fraud",
        "NonFraud"
    )

    test_csv["PredictedFraud"] = prediction_labels
    test_csv["FraudProbability"] = probabilities

    # -----------------------------------------------------
    # FRAUD TRANSACTIONS
    # -----------------------------------------------------

    fraud_transactions = test_csv[
        test_csv["PredictedFraud"] == "Fraud"
    ].copy()

    # -----------------------------------------------------
    # ACCOUNT TRANSACTION COUNTS
    # -----------------------------------------------------

    transaction_counts = (
        test_csv
        .groupby("AccountID")
        .size()
        .to_dict()
    )

    # -----------------------------------------------------
    # BUILD ACCOUNT OUTPUT
    # -----------------------------------------------------

    accounts = []

    for account_id in fraud_transactions["AccountID"].unique():

        account_data = test_csv[
            test_csv["AccountID"] == account_id
        ]

        account_fraud = fraud_transactions[
            fraud_transactions["AccountID"] == account_id
        ]

        transactions = []

        for _, row in account_fraud.iterrows():

            transactions.append({
                "transaction_id": row["TransactionID"],
                "transaction_date": row["TransactionDate"],
                "probability": f"{row['FraudProbability']:.4f}"
            })

        accounts.append({
            "account_id": account_id,
            "total_transactions": transaction_counts[account_id],
            "username": account_data["UserName"].iloc[0],
            "email": account_data["Email"].iloc[0],
            "balance": account_data["AccountBalance"].iloc[0],
            "transactions": transactions
        })

    # -----------------------------------------------------
    # RENDER RESULT
    # -----------------------------------------------------

    return render_template_string(
        HTML,
        result=True,
        total_transactions=len(test_csv),
        fraud_count=len(fraud_transactions),
        nonfraud_count=len(test_csv) - len(fraud_transactions),
        accounts=accounts
    )

# ---------------------------------------------------------
# REST API - CSV FRAUD PREDICTION
# ---------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():

    # -----------------------------------------------------
    # CHECK FILE
    # -----------------------------------------------------

    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "Please upload a CSV file using the 'file' field."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "No CSV file selected."
        }), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({
            "success": False,
            "error": "Only CSV files are allowed."
        }), 400

    # -----------------------------------------------------
    # READ CSV
    # -----------------------------------------------------

    try:

        test_csv = pd.read_csv(file)

    except Exception as e:

        return jsonify({
            "success": False,
            "error": f"Unable to read CSV file: {str(e)}"
        }), 400

    # -----------------------------------------------------
    # CHECK TRAINED FEATURES
    # -----------------------------------------------------

    missing_features = [
        col
        for col in trained_features
        if col not in test_csv.columns
    ]

    if missing_features:

        return jsonify({
            "success": False,
            "error": "Required trained features are missing.",
            "missing_features": missing_features
        }), 400

    # -----------------------------------------------------
    # CHECK OUTPUT COLUMNS
    # -----------------------------------------------------

    required_output_columns = [
        "AccountID",
        "TransactionID",
        "TransactionDate",
        "UserName",
        "Email",
        "AccountBalance"
    ]

    missing_output_columns = [
        col
        for col in required_output_columns
        if col not in test_csv.columns
    ]

    if missing_output_columns:

        return jsonify({
            "success": False,
            "error": "Required output columns are missing.",
            "missing_columns": missing_output_columns
        }), 400

    # -----------------------------------------------------
    # PREPARE ONLY TRAINED FEATURES
    # -----------------------------------------------------

    X_prediction = test_csv[trained_features].copy()

    for col in categorical_features:
        X_prediction[col] = X_prediction[col].astype(str)

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    try:

        predictions = best_model.predict(X_prediction)

        probabilities = best_model.predict_proba(
            X_prediction
        )[:, 1]

    except Exception as e:

        return jsonify({
            "success": False,
            "error": f"Prediction error: {str(e)}"
        }), 500

    # -----------------------------------------------------
    # CONVERT PREDICTION
    # -----------------------------------------------------

    prediction_labels = np.where(
        predictions == 1,
        "Fraud",
        "NonFraud"
    )

    test_csv["PredictedFraud"] = prediction_labels
    test_csv["FraudProbability"] = probabilities

    # -----------------------------------------------------
    # FRAUD TRANSACTIONS
    # -----------------------------------------------------

    fraud_transactions = test_csv[
        test_csv["PredictedFraud"] == "Fraud"
    ].copy()

    # -----------------------------------------------------
    # TRANSACTION COUNTS
    # -----------------------------------------------------

    transaction_counts = (
        test_csv
        .groupby("AccountID")
        .size()
        .to_dict()
    )

    # -----------------------------------------------------
    # BUILD FRAUD ACCOUNT JSON
    # -----------------------------------------------------

    fraud_accounts = []

    for account_id in fraud_transactions["AccountID"].unique():

        account_data = test_csv[
            test_csv["AccountID"] == account_id
        ]

        account_fraud = fraud_transactions[
            fraud_transactions["AccountID"] == account_id
        ]

        transactions = []

        for _, row in account_fraud.iterrows():

            transactions.append({
                "transaction_id": str(row["TransactionID"]),
                "transaction_date": str(row["TransactionDate"]),
                "fraud_probability": float(
                    row["FraudProbability"]
                )
            })

        fraud_accounts.append({
            "account_id": str(account_id),
            "total_transactions": int(
                transaction_counts[account_id]
            ),
            "user_name": str(
                account_data["UserName"].iloc[0]
            ),
            "email": str(
                account_data["Email"].iloc[0]
            ),
            "account_balance": float(
                account_data["AccountBalance"].iloc[0]
            ),
            "fraud_transactions": transactions
        })

    # -----------------------------------------------------
    # API RESPONSE
    # -----------------------------------------------------

    return jsonify({
        "success": True,
        "total_transactions": int(len(test_csv)),
        "predicted_fraud": int(len(fraud_transactions)),
        "predicted_nonfraud": int(
            len(test_csv) - len(fraud_transactions)
        ),
        "fraud_accounts": fraud_accounts
    })

# ---------------------------------------------------------
# RENDER SERVER
# ---------------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
