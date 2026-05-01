import os

# --- CHEMINS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

RAW_DATA_PATH = os.path.join(DATA_DIR, "dataset.csv") 
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "clean_dataset.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "shipping_model.joblib")
ISO_MODEL_PATH = os.path.join(MODELS_DIR, "iso_forest.joblib")
AUTOENCODER_PATH = os.path.join(MODELS_DIR, "autoencoder.h5")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")  # ← Ajouté pour le scaler

# --- VARIABLES (NOMS EXACTS DU DATASET) ---
TARGET_COL = "Reached_on_Time_Y_N"

# On respecte scrupuleusement les majuscules et underscores du CSV
CAT_COLS = ["Warehouse_block", "Mode_of_Shipment", "Product_importance", "Gender"]

NUM_COLS = [
    "Customer_care_calls", 
    "Customer_rating", 
    "Cost_of_the_Product", 
    "Prior_purchases", 
    "Discount_offered", 
    "Weight_in_gms",
    "price_weight_ratio",     
    "is_high_discount",      
    "calls_per_purchase",   
    "criticality_score"
]

COLS_TO_DROP = ["ID"]

# Paramètres globaux
RANDOM_STATE = 42
DEFAULT_CONTAMINATION = 0.05    