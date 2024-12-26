import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Charger le modèle enregistré
model = load_model("mon_modele_signaux.h5")

# Exemple d'entrée : un tableau avec les mêmes colonnes que celles utilisées pour entraîner le modèle
example_input = pd.DataFrame({
    'MA_signal': [0],
    'RSI_signal': [0],
    'MACD_signal': [0],
    'Bollinger_signal': [0],

})

# Convertir en numpy array (format attendu par Keras)
input_array = example_input.values

# Faire une prédiction
prediction = model.predict(input_array)

# Afficher la prédiction
print("Prédiction du modèle :", 1 if prediction > 0.4 else 0)
