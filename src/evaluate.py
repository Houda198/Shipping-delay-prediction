import pandas as pd
import numpy as np
import joblib
import os
import sys
import logging
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score,
    recall_score, matthews_corrcoef, precision_recall_curve
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Même feature engineering que train_v5.py."""
    X = df.copy()

    X['cost_per_weight'] = X['Cost_of_the_Product'] / (X['Weight_in_gms'] + 1)
    X['discount_per_cost'] = X['Discount_offered'] / (X['Cost_of_the_Product'] + 1)
    X['calls_per_purchase'] = X['Customer_care_calls'] / (X['Prior_purchases'] + 1)
    X['weight_per_call'] = X['Weight_in_gms'] / (X['Customer_care_calls'] + 1)
    X['cost_per_call'] = X['Cost_of_the_Product'] / (X['Customer_care_calls'] + 1)
    X['discount_per_weight'] = X['Discount_offered'] / (X['Weight_in_gms'] + 1)

    X['high_value'] = (X['Cost_of_the_Product'] > X['Cost_of_the_Product'].median()).astype(int)
    X['high_discount'] = (X['Discount_offered'] > X['Discount_offered'].median()).astype(int)
    X['heavy_and_expensive'] = ((X['is_heavy'] == 1) & (X['high_value'] == 1)).astype(int)
    X['light_and_cheap'] = ((X['is_light'] == 1) & (X['high_value'] == 0)).astype(int)
    X['high_calls_low_rating'] = ((X['Customer_care_calls'] > 3) & (X['Customer_rating'] < 3)).astype(int)

    warehouse_cols = ['Warehouse_block_B', 'Warehouse_block_C', 'Warehouse_block_D', 'Warehouse_block_F']
    X['warehouse_count'] = X[warehouse_cols].sum(axis=1)

    mode_cols = ['Mode_of_Shipment_Road', 'Mode_of_Shipment_Ship']
    X['mode_count'] = X[mode_cols].sum(axis=1)

    X['urgency_score'] = (
        X['Customer_care_calls'] * 0.3 +
        X['Discount_offered'] * 0.01 +
        X['Prior_purchases'] * 0.1
    )

    X['value_score'] = (
        X['Cost_of_the_Product'] * 0.001 +
        X['Customer_rating'] * 0.2 -
        X['Discount_offered'] * 0.01
    )

    X['cost_x_rating'] = X['Cost_of_the_Product'] * X['Customer_rating']
    X['calls_x_discount'] = X['Customer_care_calls'] * X['Discount_offered']
    X['weight_x_cost'] = X['Weight_in_gms'] * X['Cost_of_the_Product']
    X['rating_squared'] = X['Customer_rating'] ** 2
    X['cost_squared'] = X['Cost_of_the_Product'] ** 2

    return X


def add_target_encoding(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame):
    """Target encoding par groupe."""
    X_train_te = X_train.copy()
    X_test_te = X_test.copy()

    warehouse_cols = ['Warehouse_block_B', 'Warehouse_block_C', 'Warehouse_block_D', 'Warehouse_block_F']
    for col in warehouse_cols:
        if col in X_train.columns:
            mask = X_train[col] == 1
            mean_delay = y_train[mask].mean() if mask.sum() > 0 else y_train.mean()
            X_train_te[f'{col}_te'] = mean_delay
            X_test_te[f'{col}_te'] = mean_delay

    mode_cols = ['Mode_of_Shipment_Road', 'Mode_of_Shipment_Ship']
    for col in mode_cols:
        if col in X_train.columns:
            mask = X_train[col] == 1
            mean_delay = y_train[mask].mean() if mask.sum() > 0 else y_train.mean()
            X_train_te[f'{col}_te'] = mean_delay
            X_test_te[f'{col}_te'] = mean_delay

    return X_train_te, X_test_te


def add_clustering(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Clustering K-Means comme feature."""
    from sklearn.cluster import KMeans

    numeric_cols = ['Cost_of_the_Product', 'Weight_in_gms', 'Customer_rating',
                    'Customer_care_calls', 'Prior_purchases', 'Discount_offered']
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]

    if len(numeric_cols) < 3:
        return X_train, X_test

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    X_train['cluster'] = kmeans.fit_predict(X_train[numeric_cols])
    X_test['cluster'] = kmeans.predict(X_test[numeric_cols])

    return X_train, X_test


def load_test_data():
    """Charge le test set sauvegardé par train_v5.py."""
    logger.info(" Chargement des données...")

    test_path = os.path.join(config.MODELS_DIR, "test_data_v5.pkl")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ {test_path} introuvable. Lancez train_v5.py d'abord.")

    data = joblib.load(test_path)
    logger.info(f" Test set chargé: {len(data['y_test'])} échantillons")
    logger.info(f"   Features: {data['feature_names']}")

    return data['X_test'], data['X_test_scaled'], data['y_test']


def evaluate_best_model(X_test, X_test_scaled, y_test):
    """Évalue le meilleur modèle sauvegardé avec threshold tuning."""
    logger.info(" Chargement du meilleur modèle...")

    model_path = os.path.join(config.MODELS_DIR, "best_model_v5.joblib")
    threshold_path = os.path.join(config.MODELS_DIR, "best_threshold_v5.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f" {model_path} introuvable.")

    model = joblib.load(model_path)
    threshold_info = joblib.load(threshold_path) if os.path.exists(threshold_path) else {'threshold': 0.5}
    best_thresh = threshold_info['threshold']

    # Déterminer si le modèle a besoin de features brutes ou scalées
    try:
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        X_eval = X_test_scaled
        logger.info("   Utilisation des features scalées")
    except Exception:
        y_proba = model.predict_proba(X_test)[:, 1]
        X_eval = X_test
        logger.info("   Utilisation des features brutes")

    # Prédictions avec threshold tuned
    y_pred = (y_proba >= best_thresh).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_proba),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'threshold': best_thresh
    }

    logger.info("\n" + "="*50)
    logger.info(" MEILLEUR MODÈLE - Évaluation sur TEST SET")
    logger.info("="*50)
    logger.info(f"Threshold : {metrics['threshold']:.2f}")
    logger.info(f"Accuracy  : {metrics['accuracy']:.4f}")
    logger.info(f"F1-Score  : {metrics['f1']:.4f}")
    logger.info(f"Precision : {metrics['precision']:.4f}")
    logger.info(f"Recall    : {metrics['recall']:.4f}")
    logger.info(f"AUC-ROC   : {metrics['auc_roc']:.4f}")
    logger.info(f"MCC       : {metrics['mcc']:.4f}")

    logger.info("\nRapport détaillé:")
    print(classification_report(y_test, y_pred, target_names=['Retard', 'À temps']))

    # Visualisations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Matrice de Confusion')
    axes[0, 0].set_xlabel('Prédit')
    axes[0, 0].set_ylabel('Réel')

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = metrics['auc_roc']
    axes[0, 1].plot(fpr, tpr, label=f'AUC = {auc_val:.3f}', linewidth=2)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 1].set_title('Courbe ROC')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    if hasattr(model, 'feature_importances_'):
        n_features = len(model.feature_importances_)
        importance = pd.DataFrame({
            'feature': X_eval.columns[:n_features],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True).tail(15)
        axes[1, 0].barh(importance['feature'], importance['importance'], color='steelblue')
        axes[1, 0].set_title('Top 15 Features Importantes')
        axes[1, 0].set_xlabel('Importance')

    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    axes[1, 1].plot(rec, prec, color='green', linewidth=2)
    axes[1, 1].set_title('Precision-Recall Curve')
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUTS_DIR, 'eval_best_model_v5.png'), dpi=300)
    plt.close()

    # Rapport texte
    report_path = os.path.join(config.OUTPUTS_DIR, 'evaluation_report_v5.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("RAPPORT D'ÉVALUATION V5 - TEST SET\n")
        f.write("="*60 + "\n\n")
        f.write(f"Threshold : {metrics['threshold']:.2f}\n")
        for k, v in metrics.items():
            if k != 'threshold':
                f.write(f"{k:12s}: {v:.4f}\n")

    logger.info(f"\n Rapport sauvegardé: {report_path}")
    return metrics


def run_evaluation():
    logger.info("="*60)
    logger.info(" ÉVALUATION V5 - TEST SET")
    logger.info("="*60)

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    X_test, X_test_scaled, y_test = load_test_data()
    evaluate_best_model(X_test, X_test_scaled, y_test)

    logger.info("\n" + "="*60)
    logger.info(" ÉVALUATION TERMINÉE")
    logger.info("="*60)


if __name__ == "__main__":
    run_evaluation()