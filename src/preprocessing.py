import re


def clean_text(text):
    """
    Nettoie un message texte pour le traitement NLP.
    """

    # 1. Convertir en minuscules
    text = text.lower()

    # 2. Supprimer les URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 3. Supprimer les caractères non alphabétiques
    text = re.sub(r"[^a-z\s]", "", text)

    # 4. Supprimer les espaces multiples
    text = re.sub(r"\s+", " ", text).strip()

    return text