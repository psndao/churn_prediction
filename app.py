import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template
import joblib   

app = Flask(__name__)

# Charger le modèle et les encodeurs sauvegardés
#model = pickle.load(open("models/Model.sav", "rb"))
#encoders = pickle.load(open("models/encoders.pkl", "rb"))

model = joblib.load("models/Model.sav")
encoders = joblib.load("models/encoders.pkl")

@app.route("/")
def home_page():
    return render_template("home.html")

@app.route("/", methods=["POST"])
def predict():
    """Selected features: Dependents, tenure, OnlineSecurity,
       OnlineBackup, DeviceProtection, TechSupport, Contract,
       PaperlessBilling, MonthlyCharges, TotalCharges"""

    # Récupérer les données du formulaire
    data = [[
        request.form["Dependents"],
        float(request.form["tenure"]),
        request.form["OnlineSecurity"],
        request.form["OnlineBackup"],
        request.form["DeviceProtection"],
        request.form["TechSupport"],
        request.form["Contract"],
        request.form["PaperlessBilling"],
        float(request.form["MonthlyCharges"]),
        float(request.form["TotalCharges"])
    ]]

    df = pd.DataFrame(data, columns=[
        "Dependents", "tenure", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "Contract",
        "PaperlessBilling", "MonthlyCharges", "TotalCharges"
    ])

    # Encodage des variables catégorielles
    for feature, encoder in encoders.items():
        df[feature] = encoder.transform(df[feature])

    # Prédiction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[:, 1][0] * 100

    if prediction == 1:
        op1 = "This Customer is likely to Churn!"
    else:
        op1 = "This Customer is likely to Stay!"

    op2 = f"Confidence level: {np.round(probability, 2)}%"

    return render_template("home.html", op1=op1, op2=op2, **request.form)

if __name__ == "__main__":
    app.run(debug=True)
