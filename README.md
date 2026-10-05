# Email Spam Filtering

Projet de classification de SMS en utilisant le Natural Language Processing (NLP) et l'algorithme Naive Bayes.

## Objectif

L'objectif de ce projet est de construire un modèle capable de classifier automatiquement un SMS comme :

- HAM : message normal
- SPAM : message indésirable

## Dataset

Le projet utilise le dataset `SMSSpamCollection`.

Le dataset contient des messages SMS associés à deux classes :

- `ham`
- `spam`

Avant l'entraînement, les données sont nettoyées et les doublons sont supprimés.

## Preprocessing

Les messages passent par plusieurs étapes de nettoyage :

- conversion en minuscules
- suppression des URLs
- suppression des caractères spéciaux
- suppression des espaces inutiles

## Train / Test

Les données sont séparées en :

- 80% pour l'entraînement
- 20% pour le test

Le découpage utilise `stratify` afin de conserver la proportion HAM/SPAM.

## Naive Bayes

Le classifieur Naive Bayes a été implémenté from scratch en Python.

Le modèle utilise :

- les probabilités a priori des classes
- le comptage des mots
- le Laplace smoothing
- les logarithmes pour éviter les problèmes numériques

Le classifieur n'utilise pas `MultinomialNB` de Scikit-learn.

## Evaluation

Les métriques suivantes ont également été implémentées from scratch :

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Structure du projet

```text
email-spam-filtering/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── naive_bayes_model.pkl
│
├── tests/
│
├── .gitignore
├── requirements.txt
└── README.md