import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# ─── PATH CONFIG ───
# Si app.py est à la racine, config.py doit être accessible directement.
# Si app.py est dans src/, décommente les 2 lignes suivantes :
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# TensorFlow optionnel (même logique que model.py)
try:
    from tensorflow.keras.models import load_model
    HAS_TF = True
except ImportError:
    HAS_TF = False

# ─── CHEMINS ───
MODELS_DIR = getattr(config, 'MODELS_DIR', 'models')
OUTPUTS_DIR = getattr(config, 'OUTPUTS_DIR', 'outputs')

# ─── CHARGEMENT ARTEFACTS ───
@st.cache_resource
def load_artifacts():
    """Charge tous les modèles et utilitaires sauvegardés par train.py / evaluate.py."""
    art = {}

    # 1. Modèle principal (CatBoost/XGBoost/LightGBM)
    art["model"] = joblib.load(os.path.join(MODELS_DIR, "best_model_v5.joblib"))

    # 2. Scaler & threshold
    art["scaler"] = joblib.load(os.path.join(MODELS_DIR, "scaler_v5.pkl"))
    art["threshold"] = joblib.load(os.path.join(MODELS_DIR, "best_threshold_v5.pkl"))

    # 3. Colonnes attendues (depuis le test set sauvegardé)
    test_data = joblib.load(os.path.join(MODELS_DIR, "test_data_v5.pkl"))
    art["expected_cols"] = test_data["X_test_scaled"].columns.tolist()

    # 4. Isolation Forest (optionnel)
    iso_path = os.path.join(MODELS_DIR, "iso_forest_v5.joblib")
    art["iso"] = joblib.load(iso_path) if os.path.exists(iso_path) else None

    # 5. Autoencoder (optionnel)
    auto_path = os.path.join(MODELS_DIR, "autoencoder.h5")
    if HAS_TF and os.path.exists(auto_path):
        art["auto"] = load_model(auto_path)
    else:
        art["auto"] = None

    return art


# ─── FEATURE ENGINEERING (reproduit train.py à l'identique) ───
def add_features_app(df: pd.DataFrame, medians: dict, te_stats: dict) -> pd.DataFrame:
    """
    Reproduit le feature engineering de train.py.
    Utilise des valeurs pré-calculées (médianes & target encodings) car on n'a
    pas le y_train disponible en inférence.
    """
    X = df.copy()

    # ── Ratios & interactions ──
    X["cost_per_weight"] = X["Cost_of_the_Product"] / (X["Weight_in_gms"] + 1)
    X["discount_per_cost"] = X["Discount_offered"] / (X["Cost_of_the_Product"] + 1)
    X["calls_per_purchase"] = X["Customer_care_calls"] / (X["Prior_purchases"] + 1)
    X["weight_per_call"] = X["Weight_in_gms"] / (X["Customer_care_calls"] + 1)
    X["cost_per_call"] = X["Cost_of_the_Product"] / (X["Customer_care_calls"] + 1)
    X["discount_per_weight"] = X["Discount_offered"] / (X["Weight_in_gms"] + 1)

    # ── Features binaires (avec médianes du train) ──
    cost_med = medians.get("Cost_of_the_Product", 200)
    discount_med = medians.get("Discount_offered", 10)

    X["high_value"] = (X["Cost_of_the_Product"] > cost_med).astype(int)
    X["high_discount"] = (X["Discount_offered"] > discount_med).astype(int)

    is_heavy = X.get("is_heavy", 0)
    is_light = X.get("is_light", 0)
    X["heavy_and_expensive"] = ((is_heavy == 1) & (X["high_value"] == 1)).astype(int)
    X["light_and_cheap"] = ((is_light == 1) & (X["high_value"] == 0)).astype(int)
    X["high_calls_low_rating"] = ((X["Customer_care_calls"] > 3) & (X["Customer_rating"] < 3)).astype(int)

    # ── Counts catégoriels ──
    wh_cols = ["Warehouse_block_B", "Warehouse_block_C", "Warehouse_block_D", "Warehouse_block_F"]
    X["warehouse_count"] = X[[c for c in wh_cols if c in X.columns]].sum(axis=1)

    mode_cols = ["Mode_of_Shipment_Road", "Mode_of_Shipment_Ship"]
    X["mode_count"] = X[[c for c in mode_cols if c in X.columns]].sum(axis=1)

    # ── Scores composites ──
    X["urgency_score"] = (
        X["Customer_care_calls"] * 0.3
        + X["Discount_offered"] * 0.01
        + X["Prior_purchases"] * 0.1
    )
    X["value_score"] = (
        X["Cost_of_the_Product"] * 0.001
        + X["Customer_rating"] * 0.2
        - X["Discount_offered"] * 0.01
    )

    # ── Polynomiales ──
    X["cost_x_rating"] = X["Cost_of_the_Product"] * X["Customer_rating"]
    X["calls_x_discount"] = X["Customer_care_calls"] * X["Discount_offered"]
    X["weight_x_cost"] = X["Weight_in_gms"] * X["Cost_of_the_Product"]
    X["rating_squared"] = X["Customer_rating"] ** 2
    X["cost_squared"] = X["Cost_of_the_Product"] ** 2

    # ── Target encoding (fallback depuis stats du train) ──
    for col in wh_cols + mode_cols:
        if col in X.columns:
            te_val = te_stats.get(col, 0.60)  # 0.60 ≈ moyenne globale
            X[f"{col}_te"] = te_val

    # ── Clustering (fallback) ──
    # Le KMeans n'est pas exporté dans ton train.py actuel.
    # Si tu veux le vrai cluster, sauvegarde-le dans train.py (voir note plus bas).
    X["cluster"] = 0

    return X


# ─── UI PRINCIPALE ───
def main():
    st.set_page_config(page_title="Shipping Intelligence", layout="wide")
    st.title("🚢 Système de Surveillance Logistique")
    st.caption("CatBoost + Isolation Forest + Autoencoder | Pipeline V5")

    # ── Chargement ──
    try:
        art = load_artifacts()
    except Exception as e:
        st.error(f"Erreur de chargement des modèles : {e}")
        st.info("Lance d'abord `python src/train.py` puis `python src/evaluate.py` pour générer les artefacts.")
        return

    stats_path = os.path.join(MODELS_DIR, "feature_engineering_stats_v5.pkl")
    if os.path.exists(stats_path):
        stats = joblib.load(stats_path)
        MEDIAN_FALLBACK = stats["medians"]
        TE_FALLBACK = stats["target_encodings"]
    else:
        MEDIAN_FALLBACK = {
            "Cost_of_the_Product": 200,
            "Discount_offered": 10,
        }
        TE_FALLBACK = {
            "Warehouse_block_B": 0.60,
            "Warehouse_block_C": 0.60,
            "Warehouse_block_D": 0.60,
            "Warehouse_block_F": 0.60,
            "Mode_of_Shipment_Road": 0.60,
            "Mode_of_Shipment_Ship": 0.60,
        }

    # ── Sidebar : saisie ──
    with st.sidebar:
        st.header("📦 Caractéristiques du colis")

        cost = st.number_input("Coût du produit ($)", min_value=1, value=200, step=10)
        weight = st.number_input("Poids (g)", min_value=1, value=3000, step=100)
        discount = st.slider("Remise (%)", 0, 60, 10)
        calls = st.slider("Appels support", 1, 7, 3)
        rating = st.slider("Note client", 1, 5, 3)
        prior = st.number_input("Achats antérieurs", min_value=0, value=3)

        st.divider()
        st.header("🏭 Logistique")

        warehouse = st.selectbox("Bloc entrepôt", ["A", "B", "C", "D", "F"])
        mode = st.selectbox("Mode de transport", ["Flight", "Road", "Ship"])
        importance = st.selectbox("Importance produit", ["low", "medium", "high"])
        gender = st.selectbox("Genre client", ["M", "F"])

    # ── Construction DataFrame brut (doit matcher clean_dataset.csv) ──
    
        input_raw = pd.DataFrame([{
        "Customer_care_calls": calls,
        "Customer_rating": rating,
        "Cost_of_the_Product": cost,
        "Prior_purchases": prior,
        "Discount_offered": discount,
        "Weight_in_gms": weight,
        "Warehouse_block_A": 1 if warehouse == "A" else 0,      
        "Warehouse_block_B": 1 if warehouse == "B" else 0,
        "Warehouse_block_C": 1 if warehouse == "C" else 0,
        "Warehouse_block_D": 1 if warehouse == "D" else 0,
        "Warehouse_block_F": 1 if warehouse == "F" else 0,
        "Mode_of_Shipment_Flight": 1 if mode == "Flight" else 0, 
        "Mode_of_Shipment_Road": 1 if mode == "Road" else 0,
        "Mode_of_Shipment_Ship": 1 if mode == "Ship" else 0,
        "Product_importance_high": 1 if importance == "high" else 0, 
        "Product_importance_low": 1 if importance == "low" else 0,
        "Product_importance_medium": 1 if importance == "medium" else 0,
        "Gender_F": 1 if gender == "F" else 0,                   
        "Gender_M": 1 if gender == "M" else 0,
        "is_heavy": 1 if weight > 4000 else 0,
        "is_light": 1 if weight < 1500 else 0,
    }])

    # ── Feature engineering + alignement colonnes ──
    df_eng = add_features_app(input_raw, MEDIAN_FALLBACK, TE_FALLBACK)

    # On s'assure d'avoir EXACTEMENT les colonnes attendues par le scaler
    expected = art["expected_cols"]
    for col in expected:
        if col not in df_eng.columns:
            df_eng[col] = 0.0
    df_eng = df_eng[expected]  # même ordre, même nombre

    # ── Scaling (obligatoire, le modèle est entraîné sur du scaled) ──
    X_scaled = art["scaler"].transform(df_eng)

    # ─── PANEL PRINCIPAL : 3 colonnes ───
    c1, c2, c3 = st.columns(3)

    # 1. PRÉDICTION (modèle principal)
    with c1:
        st.subheader("🎯 Prédiction")

        model = art["model"]
        thresh_info = art["threshold"]
        best_thresh = thresh_info.get("threshold", 0.5)

        proba = model.predict_proba(X_scaled)[0]
        # D'après evaluate.py : classe 0 = Retard, classe 1 = À temps
        prob_on_time = proba[1]
        pred = 1 if prob_on_time >= best_thresh else 0

        st.metric("Probabilité « À temps »", f"{prob_on_time:.1%}")

        if pred == 1:
            st.success("✅ Livraison à temps probable")
        else:
            st.error("⚠️ Risque de retard élevé")

        st.caption(f"Seuil optimal : {best_thresh:.2f}")

    # 2. ISOLATION FOREST (anomalie)
    with c2:
        st.subheader("🔍 Anomalie")

        if art["iso"]:
            try:
                # Alignement automatique des colonnes
                if hasattr(art["iso"], "feature_names_in_"):
                    iso_cols = list(art["iso"].feature_names_in_)
                    # Crée un DataFrame avec les 22 colonnes attendues, rempli de 0
                    iso_input = pd.DataFrame(0, index=[0], columns=iso_cols)
                    # Remplit avec les valeurs disponibles dans input_raw
                    for col in iso_cols:
                        if col in input_raw.columns:
                            iso_input[col] = input_raw[col].values
                else:
                    iso_input = input_raw

                iso_pred = art["iso"].predict(iso_input)[0]
                if iso_pred == -1:
                    st.warning("🚨 Profil colis atypique")
                    st.caption("Ce colis ressemble à aucun profil connu.")
                else:
                    st.info("🟢 Profil standard")
                    st.caption("Le colis respecte les patterns habituels.")

            except Exception as e:
                st.error("🚨 Erreur Isolation Forest")
                st.caption(f"Détail : {str(e)[:60]}")
        else:
            st.info("Isolation Forest non chargé")
            st.caption("Vérifie `models/iso_forest.joblib`")

        # 3. AUTOENCODER (indice de confiance)
    with c3:
        st.subheader("🧠 Indice de confiance")

        if art["auto"]:
            try:
                # L'autoencoder est entraîné sur les données BRUTES (22 features)
                # avec un scaling manuel, pas sur les features engineered
                auto_input = input_raw.copy().astype(float)
                
                # Scaling manuel (même logique que ton entraînement original)
                if 'Cost_of_the_Product' in auto_input.columns:
                    auto_input['Cost_of_the_Product'] /= 350
                if 'Weight_in_gms' in auto_input.columns:
                    auto_input['Weight_in_gms'] /= 8000
                if 'Discount_offered' in auto_input.columns:
                    auto_input['Discount_offered'] /= 65
                
                # Vérification du shape
                expected_dim = art["auto"].input_shape[1]
                if auto_input.shape[1] != expected_dim:
                    st.warning(f"Autoencoder attend {expected_dim} features, input en a {auto_input.shape[1]}")
                    st.info("🟡 Autoencoder temporairement indisponible")
                else:
                    recon = art["auto"].predict(auto_input, verbose=0)
                    mse = float(np.mean(np.power(auto_input - recon, 2)))

                    AE_THRESH = 0.034435
                    confiance = max(0.0, min(1.0, 1.0 - (mse / (AE_THRESH * 1.25))))

                    st.progress(confiance)
                    st.write(f"MSE reconstruction : **{mse:.4f}**")

                    if mse > AE_THRESH:
                        st.error("🚨 Données inhabituelles")
                    elif mse > AE_THRESH * 0.7:
                        st.warning("⚠️ Profil peu commun")
                    else:
                        st.success("✅ Profil standard")

            except Exception as e:
                st.error("🚨 Erreur Autoencoder")
                st.caption(f"Détail : {str(e)[:80]}")
        else:
            st.info("Autoencoder non chargé")
            st.caption("Vérifie `models/autoencoder.h5`")

    # ─── DÉTAILS TECHNIQUES ───
    with st.expander("🔧 Voir les features utilisées"):
        st.dataframe(df_eng.T.rename(columns={0: "Valeur"}), use_container_width=True)

    # ─── RÉSULTATS DU TEST SET (lecture seule) ───
    st.divider()
    st.subheader("📊 Performance sur le Test Set")

    report_path = os.path.join(OUTPUTS_DIR, "evaluation_report_v5.txt")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            st.text(f.read())
    else:
        st.caption("Rapport non trouvé — lance `evaluate.py`.")

    img_path = os.path.join(OUTPUTS_DIR, "eval_best_model_v5.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="Évaluation du meilleur modèle (Confusion, ROC, Features, PR)")


if __name__ == "__main__":
    main()