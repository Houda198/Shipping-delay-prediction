import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

def run_eda():
    print(" Démarrage de l'Analyse Exploratoire (EDA) Boostée...")
    
    if not os.path.exists(config.CLEAN_DATA_PATH):
        print(f" Dataset propre introuvable ! Lance d'abord data_cleaning.py")
        return

    df = pd.read_csv(config.CLEAN_DATA_PATH)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    # 1. Distribution de la Target (Équilibre des classes)
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=config.TARGET_COL, palette="viridis")
    plt.title("Répartition des retards (0 = À temps, 1 = Retard)")
    plt.savefig(os.path.join(config.OUTPUTS_DIR, "target_distribution.png"))
    plt.close()

    # 2. Matrice de Corrélation (Vérification des nouvelles features)
    plt.figure(figsize=(12, 10)) # On l'agrandit un peu pour les nouvelles colonnes
    existing_num_cols = [c for c in config.NUM_COLS if c in df.columns]
    numeric_df = df[existing_num_cols + [config.TARGET_COL]]
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matrice de Corrélation (Inclus nouvelles features)")
    plt.tight_layout() # Évite que les noms de colonnes soient coupés
    plt.savefig(os.path.join(config.OUTPUTS_DIR, "correlation_matrix.png"))
    plt.close()

    # 3. IMPACT DU DISCOUNT (Validation du Golden Feature)
    if 'is_high_discount' in df.columns:
        plt.figure(figsize=(8, 6))
        # On compare le % de retard selon si le discount est élevé ou non
        sns.barplot(data=df, x='is_high_discount', y=config.TARGET_COL, palette="magma")
        plt.title("Probabilité de Retard vs Remise Élevée (>10%)")
        plt.ylabel("% de Retard")
        plt.savefig(os.path.join(config.OUTPUTS_DIR, "high_discount_impact.png"))
        plt.close()
        print(" Graphique 'high_discount_impact' généré.")

    # 4. RELATION PRIX/POIDS
    if 'price_weight_ratio' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.violinplot(data=df, x=config.TARGET_COL, y="price_weight_ratio", palette="Set3")
        plt.title("Répartition du Ratio Prix/Poids par Statut de Livraison")
        plt.savefig(os.path.join(config.OUTPUTS_DIR, "price_weight_violin.png"))
        plt.close()
        print(" Graphique 'price_weight_violin' généré.")

    print(f" EDA terminée ! Les images sont dans : {config.OUTPUTS_DIR}")

if __name__ == "__main__":
    run_eda()