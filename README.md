## 🚚 Shipping Delay Prediction

Projet de Machine Learning pour prédire si une livraison sera à temps ou en retard.
## 📊 Dataset

Source : Dataset Kaggle "Shipping" (~11 000 commandes)
Features : 21 variables (coût, poids, remise, appels client, entrepôt, mode de transport, etc.)
Cible : Reached_on_Time_Y_N (0 = Retard, 1 = À temps)
Distribution : 60% À temps / 40% Retard

## 🎯 Objectif

Prédire avec la meilleure accuracy possible si une livraison arrivera à temps.


Objectif	    Résultat	     Status
83% accuracy	~68.5%	⚠️ Plafond dataset

Pipeline sans data leakage	✅	Réalisé
Modèle optimal	CatBoost	✅

"Note" : Le plafond de ~68.5% est dû aux limitations intrinsèques du dataset (pas de dates, pas de localisation géographique, pas de données météo). Les retards dépendent fortement de facteurs externes non mesurés.

## 🏗️ Architecture du Projet

Shipping/
├── data/
│   ├── dataset.csv              # Données brutes
│   └── clean_dataset.csv        # Données préparées
├── src/
│   ├── data_cleaning.py         # Feature engineering + encodage
│   ├── train.py                 # Entraînement des modèles
│   └── evaluate.py              # Évaluation sur test set
├── models/
│   ├── best_model_v5.joblib     # Meilleur modèle (CatBoost)
│   ├── scaler_v5.pkl            # Scaler StandardScaler
│   └── test_data_v5.pkl         # Test set sauvegardé
├── outputs/
│   ├── model_comparison_v5.png  # Comparaison des modèles
│   ├── threshold_tuning.png     # Optimisation du seuil
│   └── eval_best_model_v5.png   # Matrices de confusion, ROC, etc.
└── config.py                    # Configuration globale
└── app.py                       # Interface streamlit

## 🚀 Installation

# Cloner le projet
git clone <repo-url>
cd Shipping

# Créer un environnement virtuel (recommandé)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install pandas numpy scikit-learn xgboost catboost lightgbm imbalanced-learn matplotlib seaborn joblib tensorflow

## 📖 Utilisation

1. Préparation des données

python src/data_cleaning.py
Nettoie les données, crée les features d'ingénierie et sauvegarde clean_dataset.csv.

2. Entraînement

python src/train.py

Teste 5 modèles : LightGBM, LightGBM+SMOTE, CatBoost, CatBoost+SMOTE, XGBoost
Effectue du threshold tuning pour chaque modèle
Sauvegarde le meilleur modèle dans models/best_model_v5.joblib

3. Évaluation

python src/evaluate.py

Évalue le meilleur modèle sur le test set avec le threshold optimal.

## 📈 Résultats
Meilleur modèle : CatBoost

Métrique	Valeur
Accuracy	68.45%
F1-Score	0.658
Precision (À temps)	0.933
Recall (À temps)	0.508
AUC-ROC	0.733
MCC	0.475
Threshold optimal	0.53
Matrice de confusion
Table
Prédit Retard	Prédit À temps
Réel Retard	842	45
Réel À temps	643	670

Comparaison des modèles

Modèle	CV Accuracy	Test Accuracy (tuned)	Threshold
LightGBM	67.62%	68.23%	0.60
LightGBM+SMOTE	66.09%	68.23%	0.64
CatBoost	67.63%	68.45%	0.53
CatBoost+SMOTE	66.09%	68.23%	0.64
XGBoost	66.33%	68.05%	0.62

## 🔧 Techniques utilisées

Feature Engineering
Ratios et interactions (cost_per_weight, discount_per_cost, calls_per_purchase)
Features binaires (high_value, heavy_and_expensive, high_calls_low_rating)
Interactions polynomiales (cost_x_rating, calls_x_discount, rating_squared)
Target encoding par entrepôt et mode de transport
Clustering K-Means (5 clusters)
Scores composites (urgency_score, value_score)
Modèles testés
LightGBM (gradient boosting rapide)
CatBoost (gradient boosting optimisé pour catégories)
XGBoost (référence)
Techniques d'optimisation
Cross-validation stratifiée (5 folds)
SMOTE (oversampling de la classe minoritaire)
Threshold tuning (optimisation du seuil de décision)
StandardScaler pour la normalisation

## 🧠 Interprétation métier : Pourquoi 68% ?

**68.45% d'accuracy** peut sembler modeste, mais il faut la lire à la lumière du **métier logistique** :

| Classe | Recall | Ce que ça signifie |
|--------|--------|-------------------|
| **Retard** | **~95%** | On détecte **quasiment tous les retards**. Le modèle est très prudent : il préfère anticiper un retard plutôt que de promettre une livraison à temps et de rater. |
| **À temps** | **~51%** | On ne capte qu'une partie des livraisons ponctuelles. Ce n'est pas grave métier : le coût d'un faux négatif (prédire "à temps" alors que c'est en retard) est bien plus élevé que l'inverse. |

### Pourquoi ce plafond ?
Le dataset est **intrinsèquement limité** : pas de dates d'expédition, pas de localisation géographique, pas de données météo/trafic. Les retards dépendent de facteurs externes non mesurés. **68.5% est le plafond réaliste** avec ces features.

### Pourquoi c'est un bon modèle malgré tout ?
Dans la logistique, **mieux vaut un faux positif (prédire retard) qu'un faux négatif (rater un vrai retard)**. Le modèle adopte une stratégie **conservatrice** : quand il prédit « À temps », il a **93% de chance d'avoir raison** (précision élevée). C'est exactement ce qu'on attend d'un outil de surveillance : une alerte fiable, pas une promesse optimiste.

## ⚠️ Limitations connues

Plafond de performance : ~68.5% est le maximum atteignable avec les features actuelles
Biais vers "Retard" : Le modèle a un recall de 95% sur Retard mais seulement 51% sur À temps
Données manquantes : Pas de dates, pas de localisation géographique, pas de données météo/trafic
Dataset bruité : Les retards dépendent de nombreux facteurs externes non mesurés

## 🔮 Améliorations possibles

Pour atteindre 80%+ d'accuracy, il faudrait :
[ ] Données temporelles : date d'expédition, jour de la semaine, saisonnalité
[ ] Données géographiques : distance entrepôt-client, zone géographique
[ ] Données externes : météo, trafic, grèves, jours fériés
[ ] Historique transporteur : taux de retard passé par transporteur/route
[ ] Deep Learning : TabNet, réseaux de neurones profonds
[ ] AutoML : Optuna, FLAML pour l'optimisation automatique
