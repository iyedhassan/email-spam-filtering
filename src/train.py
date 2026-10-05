import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import clean_text
from model import NaiveBayesSpamClassifier


# ============================================================
# 1. Chemins du projet
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "SMSSpamCollection"
MODEL_PATH = BASE_DIR / "models" / "naive_bayes_model.pkl"


# ============================================================
# 2. Charger le dataset
# ============================================================

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("Dataset chargé :", len(df), "messages")


# ============================================================
# 3. Supprimer les doublons
# ============================================================

df = df.drop_duplicates()

print("Après suppression des doublons :", len(df))


# ============================================================
# 4. Nettoyer les messages
# ============================================================

df["clean_message"] = df["message"].apply(clean_text)

# Supprimer les messages devenus vides
df = df[df["clean_message"].str.strip() != ""].copy()

print("Après nettoyage :", len(df))


# ============================================================
# 5. Préparer X et y
# ============================================================

X = df["clean_message"]

y = df["label"].astype(str).str.strip().str.lower()

y = y.map({
    "ham": 0,
    "spam": 1
})


# ============================================================
# 6. Train / Test
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train :", len(X_train))
print("X_test  :", len(X_test))


# ============================================================
# 7. Entraîner Naive Bayes
# ============================================================

model = NaiveBayesSpamClassifier()

model.fit(X_train, y_train)

print("Modèle entraîné avec succès !")


# ============================================================
# 8. Faire les prédictions
# ============================================================

y_pred = model.predict(X_test)

print("Prédictions terminées !")


# ============================================================
# 9. Accuracy from scratch
# ============================================================

def accuracy_from_scratch(y_true, y_pred):

    correct = 0

    for true, pred in zip(y_true, y_pred):

        if true == pred:
            correct += 1

    return correct / len(y_true)


accuracy = accuracy_from_scratch(y_test, y_pred)

print("Accuracy :", accuracy)
print("Accuracy en % :", accuracy * 100, "%")


# ============================================================
# 10. Sauvegarder le modèle
# ============================================================

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print("Modèle sauvegardé dans :", MODEL_PATH)