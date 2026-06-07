# 🚢 Port Intelligence AI — Doraleh Multi-Purpose Port

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Gradio](https://img.shields.io/badge/Gradio-App-green)
![R2](https://img.shields.io/badge/R²-0.667-brightgreen)

## 📋 Description

Projet de Machine Learning réalisé durant un stage de Master 1 
en Intelligence Artificielle et Modélisation de Données au sein 
du département **Business Performance Management** du 
**Doraleh Multi-Purpose Port (DMP)** de Djibouti.

**Objectif** : Prédire la surestarie (durée d'attente des camions 
dans le terminal portuaire) à partir des données opérationnelles 
réelles du port.

## 🌐 Application déployée

👉 **[Tester l'application ici](https://huggingface.co/spaces/HassanML2026/port-intelligence-ai)**

## 📊 Résultats des modèles

| Modèle | MAE (min) | RMSE (min) | R² |
|--------|-----------|------------|-----|
| Régression Linéaire | 561.17 | 1332.67 | 0.070 |
| KNN | 326.03 | 663.26 | 0.391 |
| Arbre de Décision | 308.66 | 856.01 | 0.616 |
| **Forêt Aléatoire ✅** | **260.03** | **551.98** | **0.667** |

## 🗂️ Structure du projet
ort-intelligence-ai/
│
├── chargement.ipynb                          # Chargement des données
├── Exploration_de_donnes_.ipynb              # EDA et visualisation
├── prediction_Surestarie_regression_lineaire.ipynb
├── prediction_Surestarie_KNN.ipynb
├── prediction_Surestarie_arbre.ipynb
├── prediction_Surestarie_foret.ipynb         # ✅ Modèle retenu
├── app.py                                    # Interface Gradio
└── requirements.txt                          # Dépendances

## 🔬 Méthodologie

- **62 329** enregistrements opérationnels réels (Oct 2023 – Juin 2024)
- **4 fichiers** : Gate_GC, Gate_Out, DISCH Conteneurs, LOAD Conteneurs
- **3 stratégies d'imputation** : Simple, KNN, Itérative
- **Pipeline sklearn** : ColumnTransformer + OneHotEncoding
- **Analyse SHAP** pour l'interprétabilité du modèle

## 🛠️ Technologies utilisées

- Python 3.10
- Scikit-learn
- Pandas / NumPy
- SHAP
- Gradio
- Power BI
- Hugging Face Spaces

## 👤 Auteur

**Hassan Mahamoud Hassan**
Master 1 — Intelligence Artificielle & Modélisation de Données
Université de Djibouti — Stage Mars/Juin 2026
Tuteur de stage : Samatar Houssein — DMP/PDSA
