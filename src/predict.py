import pickle
from pathlib import Path

from src.preprocessing import clean_text


# ============================================================
# 1. Chemins du projet
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "naive_bayes_model.pkl"


# ============================================================
# 2. Charger le modèle
# ============================================================

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# ============================================================
# 3. Afficher le titre
# ============================================================

print("=" * 40)
print("       SMS SPAM CLASSIFIER")
print("=" * 40)


# ============================================================
# 4. Demander le SMS
# ============================================================

message = input("\nEnter your SMS: ")


# ============================================================
# 5. Nettoyer le message
# ============================================================

cleaned_message = clean_text(message)


# ============================================================
# 6. Faire la prédiction
# ============================================================

prediction = model.predict_one(cleaned_message)


# ============================================================
# 7. Afficher le résultat
# ============================================================

print("\nPrediction:")

if prediction == 1:
    print("SPAM")
else:
    print("HAM")