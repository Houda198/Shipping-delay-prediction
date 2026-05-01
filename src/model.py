import joblib
from typing import Optional, Dict, Any
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

RANDOM_STATE = 42

# TensorFlow optionnel
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.regularizers import l2
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print(" TensorFlow non installé. L'Autoencoder sera désactivé.")


def get_isolation_forest(contamination: str = 'auto'):
    """Détecte les anomalies sans labels (unsupervised)."""
    return IsolationForest(
        n_estimators=200,
        max_samples='auto',
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def get_baseline_model():
    """Baseline simple pour comparaison."""
    return LogisticRegression(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=1000,
        n_jobs=-1
    )


def get_random_forest():
    """Random Forest pour feature importance et comparaison."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def get_xgboost_model(best_params: Optional[Dict[str, Any]] = None):
    """XGBoost optimisé. Defaults conservateurs pour grid search."""
    if best_params:
        params = dict(best_params)
        params['random_state'] = RANDOM_STATE
        params['n_jobs'] = -1
        return XGBClassifier(**params)

    return XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        min_child_weight=5,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def get_autoencoder(input_dim: int):
    """Autoencoder avec régularisation L2, dropout et batch norm."""
    if not HAS_TENSORFLOW:
        raise ImportError("TensorFlow requis pour l'Autoencoder")

    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu', name="bottleneck", kernel_regularizer=l2(0.001)),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(input_dim, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


if __name__ == "__main__":
    print(" Test des modèles...")
    iso = get_isolation_forest()
    print(" Isolation Forest")
    baseline = get_baseline_model()
    print(" Baseline (Logistic Regression)")
    rf = get_random_forest()
    print(" Random Forest")
    xgb = get_xgboost_model()
    print(" XGBoost")
    if HAS_TENSORFLOW:
        auto = get_autoencoder(10)
        print(" Autoencoder")
    print("\n Tous les modèles sont valides.")