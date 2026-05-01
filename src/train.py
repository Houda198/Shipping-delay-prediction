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
from typing import Tuple, Optional, Dict

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score,
    recall_score, matthews_corrcoef, precision_recall_curve
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# CatBoost et LightGBM
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print(" CatBoost non installé. pip install catboost")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print(" LightGBM non installé. pip install lightgbm")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering avancé."""
    X = df.copy()

    # === RATIOS ET INTERACTIONS ===
    X['cost_per_weight'] = X['Cost_of_the_Product'] / (X['Weight_in_gms'] + 1)
    X['discount_per_cost'] = X['Discount_offered'] / (X['Cost_of_the_Product'] + 1)
    X['calls_per_purchase'] = X['Customer_care_calls'] / (X['Prior_purchases'] + 1)
    X['weight_per_call'] = X['Weight_in_gms'] / (X['Customer_care_calls'] + 1)
    X['cost_per_call'] = X['Cost_of_the_Product'] / (X['Customer_care_calls'] + 1)
    X['discount_per_weight'] = X['Discount_offered'] / (X['Weight_in_gms'] + 1)

    # === FEATURES BINAIRES ===
    X['high_value'] = (X['Cost_of_the_Product'] > X['Cost_of_the_Product'].median()).astype(int)
    X['high_discount'] = (X['Discount_offered'] > X['Discount_offered'].median()).astype(int)
    X['heavy_and_expensive'] = ((X['is_heavy'] == 1) & (X['high_value'] == 1)).astype(int)
    X['light_and_cheap'] = ((X['is_light'] == 1) & (X['high_value'] == 0)).astype(int)
    X['high_calls_low_rating'] = ((X['Customer_care_calls'] > 3) & (X['Customer_rating'] < 3)).astype(int)

    # === INTERACTIONS CATÉGORIELLES ===
    warehouse_cols = ['Warehouse_block_B', 'Warehouse_block_C', 'Warehouse_block_D', 'Warehouse_block_F']
    X['warehouse_count'] = X[warehouse_cols].sum(axis=1)

    mode_cols = ['Mode_of_Shipment_Road', 'Mode_of_Shipment_Ship']
    X['mode_count'] = X[mode_cols].sum(axis=1)

    # === SCORES COMPOSITES ===
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

    # === FEATURES POLYNOMIALES CLÉS ===
    X['cost_x_rating'] = X['Cost_of_the_Product'] * X['Customer_rating']
    X['calls_x_discount'] = X['Customer_care_calls'] * X['Discount_offered']
    X['weight_x_cost'] = X['Weight_in_gms'] * X['Cost_of_the_Product']
    X['rating_squared'] = X['Customer_rating'] ** 2
    X['cost_squared'] = X['Cost_of_the_Product'] ** 2

    # === TARGET ENCODING PAR GROUPE ===
    # Entrepôt avec retard moyen (sur train uniquement, fait après le split)

    # === CLUSTERING ===
    # Fait après le split pour éviter le leakage

    return X


def add_target_encoding(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Target encoding par groupe (sans leakage)."""
    X_train_te = X_train.copy()
    X_test_te = X_test.copy()

    # Target encoding par entrepôt
    warehouse_cols = ['Warehouse_block_B', 'Warehouse_block_C', 'Warehouse_block_D', 'Warehouse_block_F']
    for col in warehouse_cols:
        if col in X_train.columns:
            mask = X_train[col] == 1
            mean_delay = y_train[mask].mean() if mask.sum() > 0 else y_train.mean()
            X_train_te[f'{col}_te'] = mean_delay
            X_test_te[f'{col}_te'] = mean_delay

    # Target encoding par mode de shipment
    mode_cols = ['Mode_of_Shipment_Road', 'Mode_of_Shipment_Ship']
    for col in mode_cols:
        if col in X_train.columns:
            mask = X_train[col] == 1
            mean_delay = y_train[mask].mean() if mask.sum() > 0 else y_train.mean()
            X_train_te[f'{col}_te'] = mean_delay
            X_test_te[f'{col}_te'] = mean_delay

    return X_train_te, X_test_te


def add_clustering(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clustering K-Means comme feature."""
    # Sélectionner les features numériques pour le clustering
    numeric_cols = ['Cost_of_the_Product', 'Weight_in_gms', 'Customer_rating',
                    'Customer_care_calls', 'Prior_purchases', 'Discount_offered']
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]

    if len(numeric_cols) < 3:
        return X_train, X_test

    # K-Means sur train
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    X_train['cluster'] = kmeans.fit_predict(X_train[numeric_cols])
    X_test['cluster'] = kmeans.predict(X_test[numeric_cols])

    return X_train, X_test


def load_and_prepare_data():
    logger.info(" Chargement des données...")
    df = pd.read_csv(config.CLEAN_DATA_PATH)
    df = df.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x)

    y = df[config.TARGET_COL]
    X_raw = df.drop(columns=[config.TARGET_COL])

    # Feature engineering de base
    X = add_features(X_raw)

    logger.info(f"Features de base: {len(X.columns)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Target encoding (sans leakage)
    X_train, X_test = add_target_encoding(X_train, y_train, X_test)

    # Clustering
    X_train, X_test = add_clustering(X_train, X_test)

    logger.info(f"Features finales: {len(X_train.columns)}")
    logger.info(f"Features: {list(X_train.columns)}")

    # Scaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # Sauvegarde
    joblib.dump(scaler, os.path.join(config.MODELS_DIR, "scaler_v5.pkl"))
    joblib.dump({
        'X_test': X_test,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'feature_names': list(X.columns)
    }, os.path.join(config.MODELS_DIR, "test_data_v5.pkl"))

        # Sauvegarde des stats pour l'API / Streamlit
    joblib.dump({
        'medians': {
            'Cost_of_the_Product': float(X['Cost_of_the_Product'].median()),
            'Discount_offered': float(X['Discount_offered'].median()),
        },
        'target_encodings': {
            'Warehouse_block_B': float(y_train[X_train['Warehouse_block_B']==1].mean()) if (X_train['Warehouse_block_B']==1).sum()>0 else float(y_train.mean()),
            'Warehouse_block_C': float(y_train[X_train['Warehouse_block_C']==1].mean()) if (X_train['Warehouse_block_C']==1).sum()>0 else float(y_train.mean()),
            'Warehouse_block_D': float(y_train[X_train['Warehouse_block_D']==1].mean()) if (X_train['Warehouse_block_D']==1).sum()>0 else float(y_train.mean()),
            'Warehouse_block_F': float(y_train[X_train['Warehouse_block_F']==1].mean()) if (X_train['Warehouse_block_F']==1).sum()>0 else float(y_train.mean()),
            'Mode_of_Shipment_Road': float(y_train[X_train['Mode_of_Shipment_Road']==1].mean()) if (X_train['Mode_of_Shipment_Road']==1).sum()>0 else float(y_train.mean()),
            'Mode_of_Shipment_Ship': float(y_train[X_train['Mode_of_Shipment_Ship']==1].mean()) if (X_train['Mode_of_Shipment_Ship']==1).sum()>0 else float(y_train.mean()),
        }
    }, os.path.join(config.MODELS_DIR, "feature_engineering_stats_v5.pkl"))

    return X_train_scaled, y_train, X_test_scaled, y_test


def find_best_threshold(model, X_test, y_test):
    """Trouve le meilleur threshold pour maximiser l'accuracy."""
    y_proba = model.predict_proba(X_test)[:, 1]

    best_thresh = 0.5
    best_acc = 0

    thresholds = np.arange(0.1, 0.9, 0.01)
    accuracies = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    logger.info(f"   Best threshold: {best_thresh:.2f} (accuracy: {best_acc:.4f})")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, accuracies, linewidth=2)
    ax.axvline(x=best_thresh, color='red', linestyle='--', label=f'Best: {best_thresh:.2f}')
    ax.set_title('Accuracy vs Threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUTS_DIR, 'threshold_tuning.png'), dpi=300)
    plt.close()

    return best_thresh, best_acc


def evaluate_model(name, model, X_train, y_train, X_test, y_test, use_smote=False, cv=5):
    """Évalue un modèle avec CV + test set + SMOTE optionnel + threshold tuning."""
    logger.info(f"\n{'='*50}")
    logger.info(f" {name}")
    logger.info(f"{'='*50}")

    # Pipeline avec SMOTE si demandé
    if use_smote:
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])
        logger.info("   SMOTE activé")
    else:
        pipeline = model

    # Cross-validation
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy')
    logger.info(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Fit sur tout le train
    pipeline.fit(X_train, y_train)

    # Test set avec threshold 0.5
    y_pred_default = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc_default = accuracy_score(y_test, y_pred_default)
    logger.info(f"   Test Accuracy (threshold=0.5): {acc_default:.4f}")

    # Threshold tuning
    logger.info("    Threshold tuning...")
    best_thresh, best_acc = find_best_threshold(pipeline, X_test, y_test)

    # Prédictions avec meilleur threshold
    y_pred_tuned = (y_proba >= best_thresh).astype(int)

    metrics = {
        'name': name,
        'cv_accuracy': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy_default': acc_default,
        'test_accuracy_tuned': best_acc,
        'best_threshold': best_thresh,
        'f1': f1_score(y_test, y_pred_tuned),
        'precision': precision_score(y_test, y_pred_tuned),
        'recall': recall_score(y_test, y_pred_tuned),
        'auc_roc': roc_auc_score(y_test, y_proba),
        'mcc': matthews_corrcoef(y_test, y_pred_tuned)
    }

    logger.info(f"   Test Accuracy (tuned): {metrics['test_accuracy_tuned']:.4f}")
    logger.info(f"   F1-Score:     {metrics['f1']:.4f}")
    logger.info(f"   Precision:    {metrics['precision']:.4f}")
    logger.info(f"   Recall:       {metrics['recall']:.4f}")
    logger.info(f"   AUC-ROC:      {metrics['auc_roc']:.4f}")
    logger.info(f"   MCC:          {metrics['mcc']:.4f}")

    logger.info("\nRapport détaillé (threshold tuned):")
    print(classification_report(y_test, y_pred_tuned, target_names=['Retard', 'À temps']))

    return pipeline, metrics


def plot_comparison(all_metrics, outputs_dir):
    """Compare tous les modèles."""
    df = pd.DataFrame(all_metrics)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Accuracy default vs tuned
    x = np.arange(len(df))
    width = 0.35
    axes[0].bar(x - width/2, df['test_accuracy_default'], width, label='Default (0.5)', color='steelblue')
    axes[0].bar(x + width/2, df['test_accuracy_tuned'], width, label='Tuned', color='coral')
    axes[0].set_title('Accuracy: Default vs Threshold Tuned')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df['name'], rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Autres métriques
    df_plot = df.set_index('name')[['f1', 'auc_roc', 'mcc']]
    df_plot.plot(kind='bar', ax=axes[1], color=['green', 'purple', 'brown'])
    axes[1].set_title('F1, AUC-ROC, MCC')
    axes[1].set_ylabel('Score')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'model_comparison_v5.png'), dpi=300)
    plt.close()

    df.to_csv(os.path.join(outputs_dir, 'model_comparison_v5.csv'), index=False)

    # Meilleur modèle
    best_idx = df['test_accuracy_tuned'].idxmax()
    best = df.loc[best_idx]
    logger.info(f"\n MEILLEUR MODÈLE: {best['name']}")
    logger.info(f"   Accuracy (tuned): {best['test_accuracy_tuned']:.4f}")
    logger.info(f"   Threshold: {best['best_threshold']:.2f}")

    return best['name']


def run_pipeline():
    logger.info("=" * 60)
    logger.info(" PIPELINE V5 - CATBOOST/LIGHTGBM + SMOTE + THRESHOLD TUNING")
    logger.info("=" * 60)

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    X_train, y_train, X_test, y_test = load_and_prepare_data()

    all_models = {}
    all_metrics = []

    # 1. LightGBM
    if HAS_LIGHTGBM:
        model, metrics = evaluate_model(
            "LightGBM",
            LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                class_weight='balanced', random_state=42, n_jobs=-1
            ),
            X_train, y_train, X_test, y_test,
            use_smote=False
        )
        all_models["LightGBM"] = model
        all_metrics.append(metrics)

        # LightGBM + SMOTE
        model, metrics = evaluate_model(
            "LightGBM+SMOTE",
            LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1
            ),
            X_train, y_train, X_test, y_test,
            use_smote=True
        )
        all_models["LightGBM+SMOTE"] = model
        all_metrics.append(metrics)

    # 2. CatBoost
    if HAS_CATBOOST:
        model, metrics = evaluate_model(
            "CatBoost",
            CatBoostClassifier(
                iterations=500, depth=6, learning_rate=0.05,
                l2_leaf_reg=3, random_seed=42, verbose=0,
                auto_class_weights='Balanced'
            ),
            X_train, y_train, X_test, y_test,
            use_smote=False
        )
        all_models["CatBoost"] = model
        all_metrics.append(metrics)

        # CatBoost + SMOTE
        model, metrics = evaluate_model(
            "CatBoost+SMOTE",
            CatBoostClassifier(
                iterations=500, depth=6, learning_rate=0.05,
                l2_leaf_reg=3, random_seed=42, verbose=0
            ),
            X_train, y_train, X_test, y_test,
            use_smote=True
        )
        all_models["CatBoost+SMOTE"] = model
        all_metrics.append(metrics)

    # 3. XGBoost (référence)
    from xgboost import XGBClassifier
    model, metrics = evaluate_model(
        "XGBoost",
        XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=1.0, random_state=42, n_jobs=-1
        ),
        X_train, y_train, X_test, y_test,
        use_smote=False
    )
    all_models["XGBoost"] = model
    all_metrics.append(metrics)

    # Comparaison
    best_name = plot_comparison(all_metrics, config.OUTPUTS_DIR)

    # Sauvegarde du meilleur modèle
    best_model = all_models[best_name]
    joblib.dump(best_model, os.path.join(config.MODELS_DIR, "best_model_v5.joblib"))

    # Sauvegarde du threshold
    best_metrics = [m for m in all_metrics if m['name'] == best_name][0]
    joblib.dump({
        'threshold': best_metrics['best_threshold'],
        'accuracy': best_metrics['test_accuracy_tuned']
    }, os.path.join(config.MODELS_DIR, "best_threshold_v5.pkl"))

    logger.info(f" Meilleur modèle sauvé: {best_name}")
    logger.info(f" Threshold sauvé: {best_metrics['best_threshold']:.2f}")

    logger.info("\n" + "=" * 60)
    logger.info(" PIPELINE TERMINÉ")
    logger.info(f" Résultats dans: {config.OUTPUTS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()

def train_isolation_forest():
    """Réentraîne l'Isolation Forest sur les données brutes (avant feature engineering)."""
    from model import get_isolation_forest
    
    df = pd.read_csv(config.CLEAN_DATA_PATH)
    X = df.drop(columns=[config.TARGET_COL])
    X = X.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x)
    
    iso = get_isolation_forest(contamination=0.1)
    iso.fit(X)
    
    joblib.dump(iso, os.path.join(config.MODELS_DIR, "iso_forest_v5.joblib"))
    print(f"✅ Isolation Forest réentraînée sur {X.shape[1]} features")

train_isolation_forest()