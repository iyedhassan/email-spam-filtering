import pickle
from pathlib import Path

from preprocessing import clean_text


# Chemin du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# Chemin du modèle
MODEL_PATH = BASE_DIR / "models" / "naive_bayes_model.pkl"


# Charger le modèle
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# Demander un SMS
message = input("Enter your SMS: ")


# Nettoyer le message
cleaned_message = clean_text(message)


# Faire la prédiction
prediction = model.predict_one(cleaned_message)


# Afficher le résultat
if prediction == 1:
    print("Prediction: SPAM")
else:
    print("Prediction: HAM")