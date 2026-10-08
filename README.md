# 📱 SMS Spam Filtering

Projet de classification automatique de SMS utilisant le **Natural Language Processing (NLP)** et l'algorithme **Naive Bayes**.

L'objectif est de déterminer automatiquement si un SMS est :

* **HAM** : message normal
* **SPAM** : message indésirable

---

## 🎯 Objectif

Construire un système capable de :

1. charger un dataset de SMS ;
2. nettoyer les messages ;
3. entraîner un classifieur Naive Bayes ;
4. prédire si un nouveau SMS est HAM ou SPAM ;
5. évaluer les performances du modèle.

Le projet contient une implémentation **from scratch** de Naive Bayes et des principales métriques d'évaluation.

---

## 📊 Dataset

Le projet utilise le dataset **SMSSpamCollection**.

Chaque ligne contient :

```text
label    message
```

Les deux classes sont :

```text
ham
spam
```

Le dataset contient initialement **5572 messages**.

Après suppression des doublons et des messages devenus vides après nettoyage, le dataset utilisé contient **5166 messages**.

---

## 🧹 Preprocessing

Les messages sont nettoyés avec une fonction personnalisée.

Les étapes sont :

* conversion en minuscules ;
* suppression des URLs ;
* suppression des caractères spéciaux ;
* suppression des espaces inutiles.

### Exemple

Message original :

```text
"Congratulations!!! Visit http://example.com NOW"
```

Après preprocessing :

```text
"congratulations visit now"
```

---

## 🧠 Naive Bayes

Le classifieur Naive Bayes a été développé **from scratch en Python**.

Le modèle utilise :

* les probabilités a priori des classes ;
* le comptage des mots ;
* le **Laplace smoothing** ;
* les logarithmes pour éviter les problèmes numériques.

Le projet n'utilise pas :

```python
MultinomialNB()
```

de Scikit-learn.

---

## 📈 Evaluation

Les métriques ont également été implémentées **from scratch** :

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Aucune fonction de métrique de `sklearn.metrics` n'est utilisée.

---

## 🧪 Tests

Le projet utilise **pytest** pour vérifier automatiquement le fonctionnement du code.

Les tests couvrent notamment :

* le preprocessing ;
* l'entraînement du modèle ;
* la prédiction ;
* Accuracy ;
* Precision ;
* Recall ;
* F1-score ;
* Confusion Matrix.

Pour lancer les tests :

```bash
python -m pytest
```

---

## 📁 Structure du projet

```text
email-spam-filtering/
│
├── data/
│   └── raw/
│       └── SMSSpamCollection
│
├── models/
│   └── naive_bayes_model.pkl
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluation.py
│   ├── train.py
│   └── predict.py
│
├── tests/
│   ├── test_model.py
│   ├── test_preprocessing.py
│   └── test_evaluation.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Cloner le projet

```bash
git clone https://github.com/iyedhassan/email-spam-filtering.git
```

Entrer dans le projet :

```bash
cd email-spam-filtering
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
```

### 3. Activer l'environnement virtuel sous Windows

```bash
.venv\Scripts\activate
```

### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 🚂 Entraîner le modèle

Pour entraîner le modèle :

```bash
python -m src.train
```

Cette commande :

1. charge le dataset ;
2. supprime les doublons ;
3. nettoie les messages ;
4. sépare les données en train/test ;
5. entraîne le modèle Naive Bayes ;
6. effectue les prédictions ;
7. calcule les métriques ;
8. sauvegarde le modèle.

Le modèle est sauvegardé dans :

```text
models/naive_bayes_model.pkl
```

---

## 📱 Prédire un SMS

Lancer :

```bash
python -m src.predict
```

Le programme demande ensuite un SMS.

### Exemple SPAM

```text
========================================
       SMS SPAM CLASSIFIER
========================================

Enter your SMS: Congratulations! You won a free prize

Prediction:
SPAM
```

### Exemple HAM

```text
Enter your SMS: Hey, are we still meeting tonight?

Prediction:
HAM
```

---

## 🔬 Pipeline du projet

Le fonctionnement général du système est :

```text
SMS
 │
 ▼
Preprocessing
 │
 ├── lowercase
 ├── remove URLs
 ├── remove special characters
 └── remove extra spaces
 │
 ▼
Naive Bayes
 │
 ├── Prior probability
 ├── Word probability
 ├── Laplace smoothing
 └── Log probabilities
 │
 ▼
Prediction
 │
 ├── HAM
 └── SPAM
```

---

## 🛠️ Technologies utilisées

* **Python**
* **Pandas**
* **Scikit-learn**
* **Natural Language Processing (NLP)**
* **Naive Bayes**
* **Pytest**
* **Git**
* **GitHub**

---

## 📚 Concepts utilisés

Ce projet permet de mettre en pratique plusieurs concepts de Machine Learning et de NLP :

* Text preprocessing
* Classification supervisée
* Train/Test Split
* Naive Bayes
* Probability
* Laplace Smoothing
* Logarithmic probabilities
* Model evaluation
* Unit testing
* Git / GitHub

---

## 👨‍💻 Auteur

Projet réalisé dans le cadre d'un projet de **Natural Language Processing / Machine Learning**.
