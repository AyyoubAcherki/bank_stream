import streamlit as st
import joblib
import numpy as np
import os
import gdown

# ✅ Cette ligne doit être AVANT tout autre appel Streamlit
st.set_page_config(page_title="BankDeposit - Prédiction", layout="centered")

# === 1. Chargement du modèle ===
def charger_modele():
    model_path = "model.joblib"
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists(model_path):
        try:
            st.sidebar.warning("⚠️ Téléchargement du modèle...")
            url = "https://drive.google.com/uc?id=1nGyl2AfgxNxPtVtileso-VTJCeJud2Xa"
            gdown.download(url, model_path, quiet=False)
            st.sidebar.success("✅ Modèle téléchargé !")
        except Exception as e:
            st.sidebar.error(f"❌ Échec du téléchargement : {str(e)}")
            st.stop()
    
    try:
        model = joblib.load(model_path)
        st.sidebar.success("✅ Modèle chargé avec succès")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Erreur de chargement : {str(e)}")
        st.stop()

# Charger le modèle
modele = charger_modele()

st.title("BankDeposit - Prédiction")

# Formulaire
with st.form("prediction_form"):
    age = st.number_input("Âge", min_value=18, max_value=100, step=1)
    balance = st.number_input("Balance")
    duration = st.number_input("Durée")
    campaign = st.number_input("Campagne")
    previous = st.number_input("Précédent")

    default = st.radio("Default", ["Oui", "Non"])
    housing = st.radio("Housing", ["Oui", "Non"])
    loan = st.radio("Loan", ["Oui", "Non"])

    education = st.radio("Niveau d'éducation", ["Primaire", "Secondaire", "Tertiaire", "Inconnu"])
    marital = st.radio("Statut marital", ["Marié(e)", "Célibataire", "Divorcé(e)"])

    job = st.radio("Profession", [
        "Admin", "Blue-collar", "Entrepreneur", "Housemaid", "Management", "Retraité",
        "Auto-entrepreneur", "Services", "Étudiant", "Technicien", "Sans emploi", "Inconnu"
    ])

    submit = st.form_submit_button("Lancer la prédiction")

if submit:
    features = np.array([[age, balance, duration, campaign, previous,
                          int(default == "Oui"), int(housing == "Oui"), int(loan == "Oui"),
                          int(education == "Primaire"), int(education == "Secondaire"),
                          int(education == "Tertiaire"), int(education == "Inconnu"),
                          int(marital == "Marié(e)"), int(marital == "Célibataire"),
                          int(marital == "Divorcé(e)"),
                          int(job == "Admin"), int(job == "Blue-collar"), int(job == "Entrepreneur"),
                          int(job == "Housemaid"), int(job == "Management"), int(job == "Retraité"),
                          int(job == "Auto-entrepreneur"), int(job == "Services"),
                          int(job == "Étudiant"), int(job == "Technicien"),
                          int(job == "Sans emploi"), int(job == "Inconnu")]])

    prediction = modele.predict(features)[0]
    result = "Client à risque de départ" if prediction == 1 else "Client fidèle"
    st.success(result)
