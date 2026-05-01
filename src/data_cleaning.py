import os
import sys
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

def clean_data():
    print(" Lancement du Feature Engineering...")
    
    raw_path = os.path.join(os.path.dirname(config.RAW_DATA_PATH), "dataset.csv")
    if not os.path.exists(raw_path):
        print(f" Erreur : Fichier {raw_path} introuvable.")
        return

    df = pd.read_csv(raw_path)
    df.columns = [col.strip() for col in df.columns]
    target = "Reached_on_Time_Y_N"

    # 1. Nettoyage de base
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # 2. FEATURE ENGINEERING (SANS LEAKAGE)
    print(" Calcul des variables...")
    
    # Variables de poids
    df['is_heavy'] = (df['Weight_in_gms'] > 4000).astype(int)
    df['is_light'] = (df['Weight_in_gms'] < 2000).astype(int)
    df['price_weight_ratio'] = df['Cost_of_the_Product'] / (df['Weight_in_gms'] + 1)
    
    # Variables de remise et pression
    df['is_high_discount'] = (df['Discount_offered'] > 10).astype(int)
    
    importance_map = {'low': 1, 'medium': 2, 'high': 3}
    df['imp_score'] = df['Product_importance'].str.lower().map(importance_map).fillna(1)
    df['pressure_index'] = df['Customer_care_calls'] * df['imp_score']
    df['discount_per_call'] = df['Discount_offered'] / (df['Customer_care_calls'] + 1)

    # 3. ENCODAGE
    cat_cols = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']
    df_clean = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], drop_first=True)
    
    # Nettoyage des noms pour XGBoost
    df_clean.columns = [c.replace(' ', '_').replace('.', '_').replace('<', 'lt').replace('>', 'gt') for c in df_clean.columns]

    if 'imp_score' in df_clean.columns:
        df_clean = df_clean.drop(columns=['imp_score'])
        
    df_clean = df_clean.fillna(0)
    df_clean.to_csv(config.CLEAN_DATA_PATH, index=False)

    print(f" Terminé ! Shape : {df_clean.shape}")
    print(f"   Features: {list(df_clean.columns)}")

if __name__ == "__main__":
    clean_data()