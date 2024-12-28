import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Charger les données d'entrée
data_test = pd.DataFrame({
    'timestamp': [
        "2020-01-02 09:00:00+00:00", "2020-01-02 10:00:00+00:00", "2020-01-02 11:00:00+00:00",
        "2020-01-02 12:00:00+00:00", "2020-01-02 13:00:00+00:00", "2020-01-02 14:00:00+00:00",
        "2020-01-02 15:00:00+00:00", "2020-01-02 16:00:00+00:00", "2020-01-02 17:00:00+00:00",
        "2020-01-02 18:00:00+00:00"
    ],
    'open': [130.21, 129.97, 129.73, 129.52, 129.97, 129.34, 129.37, 129.58, 127.6, 127.42],
    'high': [130.21, 130.28, 129.94, 129.99, 130.01, 129.86, 129.8, 129.78, 127.68, 127.75],
    'low': [129.57, 129.66, 129.44, 129.52, 128.9, 129.21, 129.36, 126.94, 126.38, 127.13],
    'close': [129.98, 129.72, 129.53, 129.96, 129.35, 129.37, 129.59, 127.6, 127.4, 127.49],
    'volume': [5828.65169, 8014.34789, 4293.24985, 3718.46074, 8547.00635, 5538.92471, 5352.62034, 32996.89479, 40429.91193, 7360.65888],
    'MACD': [-0.31928969783260186, -0.322347662661457, -0.336226721571137, -0.3089669902716423, -0.332749628253282, 
             -0.3459952890280249, -0.33488011967460807, -0.4811017947432674, -0.606134710597729, -0.6900079670580794],
    'MACD_signal_line': [-0.21527797487289982, -0.23669191243061127, -0.2565988742587164, -0.26707249746130163, -0.2802079236196977, 
                         -0.2933653967013632, -0.30166834129601217, -0.3375550319854632, -0.3912709677079164, -0.45101836757794905],
    'BB_lower': [128.44388366466555, 128.64983516041707, 128.88450842889188, 128.88734260868088, 128.91332215214186, 
                 128.9895620681668, 129.01455308744036, 128.26969444316825, 127.7475934286215, 127.34751117598056],
    'BB_upper': [131.94040204962013, 131.38302198244008, 130.82692014253666, 130.77980024846195, 130.55096356214386, 
                 130.2890093604046, 130.18401834113106, 130.62601984254604, 130.90526371423567, 131.01106025259082],
    'OBV': [15255.592379999973, 7241.244489999973, 2947.9946399999726, 6666.455379999972, -1880.5509700000275, 
            3658.3737399999727, 9010.994079999973, -23985.900710000024, -64415.812640000026, -57055.15376000002],
    'RSI': [47.768852160135516, 43.19650282948644, 39.93737665277479, 49.91490044628272, 39.15100328063808, 
            39.64882292012775, 45.38317708403726, 22.660867204604873, 21.404309291773547, 23.62762722332354],
    'Fib_Max': [130.67, 130.21, 130.21, 130.21, 130.21, 130.21, 130.21, 130.21, 130.21, 129.98],
    'Fib_Min': [129.1, 129.1, 129.1, 129.1, 129.26, 129.26, 129.26, 127.6, 127.4, 127.4],
    'Fib_382': [130.07026, 129.78598, 129.78598, 129.78598, 129.8471, 129.8471, 129.8471, 129.21298000000002, 129.13658, 128.99444],
    'fng_value': [39, 39, 39, 39, 39, 39, 39, 39, 39, 39]
})

# Préparation des données
timestamps = data_test['timestamp']  # Sauvegarder les timestamps pour référence
data_test = data_test.drop(columns=['timestamp'])  # Supprimer les timestamps pour normalisation

# Charger le scaler et normaliser les données
scaler = MinMaxScaler()
df_train = pd.read_csv("signaux_indicateurs.csv").drop(columns=['timestamp'])  # Charger les données d'entraînement
scaler.fit(df_train)  # Adapter le scaler sur les données d'entraînement
data_test_normalized = scaler.transform(data_test)

# Charger le modèle LSTM
model = load_model("eth_usdt_model.h5")

# Reshape pour correspondre au modèle LSTM (batch_size, time_steps, features)
X_test = data_test_normalized.reshape(1, 10, -1)  # 10 bougies

# Prédire la 11ème bougie
predicted_normalized = model.predict(X_test)[0]

# Dénormaliser les prédictions
predicted_real = scaler.inverse_transform([predicted_normalized])[0]

# Afficher les résultats
predicted_close_price_real = predicted_real[3]  # Indice 3 pour "close"
print("Prix de clôture prédit pour la 11ème bougie (réel) :", predicted_close_price_real)
