from flask import Flask, request, render_template_string
import pandas as pd
import joblib

app = Flask(__name__)

# Load the saved Random Forest model
bundle = joblib.load("healthletics_random_forest_churn_model.joblib")

model = bundle["model"]
features = bundle["features"]

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Healthletics Churn Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
        }
        h1 {
            text-align: center;
        }
        label {
            display: block;
            margin-top: 12px;
            font-weight: bold;
        }
        input {
            width: 100%;
            padding: 8px;
            margin-top: 4px;
            box-sizing: border-box;
        }
        button {
            margin-top: 20px;
            padding: 12px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .result {
            margin-top: 25px;
            padding: 20px;
            border: 1px solid #ccc;
        }
    </style>
</head>

<body>

<h1>Healthletics Member Churn Prediction</h1>

<p>
Enter member and activity information to estimate churn risk.
</p>

<form method="POST">

<label>Gender</label>
<input type="number" name="gender" min="0" max="1" required>

<label>Near Location</label>
<input type="number" name="Near_Location" min="0" max="1" required>

<label>Partner</label>
<input type="number" name="Partner" min="0" max="1" required>

<label>Promo Friends</label>
<input type="number" name="Promo_friends" min="0" max="1" required>

<label>Phone</label>
<input type="number" name="Phone" min="0" max="1" required>

<label>Contract Period (months)</label>
<input type="number" name="Contract_period" required>

<label>Group Visits</label>
<input type="number" name="Group_visits" min="0" max="1" required>

<label>Age</label>
<input type="number" name="Age" required>

<label>Average Additional Charges Total</label>
<input type="number" step="any" name="Avg_additional_charges_total" required>

<label>Months to End Contract</label>
<input type="number" name="Month_to_end_contract" required>

<label>Lifetime</label>
<input type="number" name="Lifetime" required>

<label>Average Class Frequency Total</label>
<input type="number" step="any" name="Avg_class_frequency_total" required>

<label>Average Class Frequency Current Month</label>
<input type="number" step="any" name="Avg_class_frequency_current_month" required>

<button type="submit">Predict Churn Risk</button>

</form>

{% if probability is not none %}
<div class="result">

<h2>Prediction Result</h2>

<p>
<strong>Predicted Churn Probability:</strong>
{{ probability }}%
</p>

{% if prediction == 1 %}
<h3>HIGHER CHURN RISK</h3>
<p>
This profile is predicted to have a higher likelihood of churn.
</p>
{% else %}
<h3>LOWER CHURN RISK</h3>
<p>
This profile is predicted to have a lower likelihood of churn.
</p>
{% endif %}

</div>
{% endif %}

<p>
<small>
Proof-of-concept decision-support tool. Predictions should not be used
as the sole basis for decisions affecting individual members.
</small>
</p>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None

    if request.method == "POST":

        data = {
            "gender": int(request.form["gender"]),
            "Near_Location": int(request.form["Near_Location"]),
            "Partner": int(request.form["Partner"]),
            "Promo_friends": int(request.form["Promo_friends"]),
            "Phone": int(request.form["Phone"]),
            "Contract_period": int(request.form["Contract_period"]),
            "Group_visits": int(request.form["Group_visits"]),
            "Age": int(request.form["Age"]),
            "Avg_additional_charges_total": float(
                request.form["Avg_additional_charges_total"]
            ),
            "Month_to_end_contract": int(
                request.form["Month_to_end_contract"]
            ),
            "Lifetime": int(request.form["Lifetime"]),
            "Avg_class_frequency_total": float(
                request.form["Avg_class_frequency_total"]
            ),
            "Avg_class_frequency_current_month": float(
                request.form["Avg_class_frequency_current_month"]
            )
        }

        # Engineered feature
        data["frequency_change"] = (
            data["Avg_class_frequency_current_month"]
            - data["Avg_class_frequency_total"]
        )

        X = pd.DataFrame([data])[features]

        prediction = int(model.predict(X)[0])

        probability = round(
            float(model.predict_proba(X)[0][1]) * 100,
            2
        )

    return render_template_string(
        HTML,
        prediction=prediction,
        probability=probability
    )


if __name__ == "__main__":
    app.run(debug=True)
