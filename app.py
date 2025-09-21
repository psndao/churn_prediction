from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

# Initialisation de l'app Flask
app = Flask(__name__)

# Répertoire de base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chargement du modèle et du scaler
model_path = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "standard_scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Colonnes numériques
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

# Mapping des variables catégorielles (comme dans ton encodage)
mapping = {
    "gender": {"Male": 1, "Female": 0},
    "Partner": {"Yes": 1, "No": 0},
    "Dependents": {"Yes": 1, "No": 0},
    "PhoneService": {"Yes": 1, "No": 0},
    "MultipleLines": {"No phone service": 0, "No": 1, "Yes": 2},
    "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2},
    "OnlineSecurity": {"No": 0, "Yes": 1, "No internet service": 2},
    "OnlineBackup": {"No": 0, "Yes": 1, "No internet service": 2},
    "DeviceProtection": {"No": 0, "Yes": 1, "No internet service": 2},
    "TechSupport": {"No": 0, "Yes": 1, "No internet service": 2},
    "StreamingTV": {"No": 0, "Yes": 1, "No internet service": 2},
    "StreamingMovies": {"No": 0, "Yes": 1, "No internet service": 2},
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "PaperlessBilling": {"Yes": 1, "No": 0},
    "PaymentMethod": {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    }
}

@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupération des données du formulaire
        data = {
            "gender": request.form["gender"],
            "SeniorCitizen": int(request.form["SeniorCitizen"]),
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "tenure": float(request.form["tenure"]),
            "PhoneService": request.form["PhoneService"],
            "MultipleLines": request.form["MultipleLines"],
            "InternetService": request.form["InternetService"],
            "OnlineSecurity": request.form["OnlineSecurity"],
            "OnlineBackup": request.form["OnlineBackup"],
            "DeviceProtection": request.form["DeviceProtection"],
            "TechSupport": request.form["TechSupport"],
            "StreamingTV": request.form["StreamingTV"],
            "StreamingMovies": request.form["StreamingMovies"],
            "Contract": request.form["Contract"],
            "PaperlessBilling": request.form["PaperlessBilling"],
            "PaymentMethod": request.form["PaymentMethod"],
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "TotalCharges": float(request.form["TotalCharges"])
        }

        # Transformation en DataFrame
        input_df = pd.DataFrame([data])
# Supprimer les colonnes inutiles (pas vues pendant l'entraînement)
        input_df = input_df.drop(columns=["gender", "PhoneService"])

        # Application du mapping pour transformer en valeurs numériques
        for col, map_dict in mapping.items():
            if col in input_df.columns:
                input_df[col] = input_df[col].map(map_dict)

        # Normalisation des colonnes numériques
        #input_df[num_cols] = scaler.transform(input_df[num_cols])

        # Prédiction
        prediction = model.predict(input_df)[0]

        # Résultat lisible
        result = "⚠️ Ce client est susceptible de résilier !" if prediction == 1 else "✅ Ce client est fidèle et restera probablement."

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"❌ Erreur : {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
