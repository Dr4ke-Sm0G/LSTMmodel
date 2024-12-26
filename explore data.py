import pandas as pd
# Charger le fichier CSV
df = pd.read_csv('CryptoGreedFear.csv', parse_dates=['date'])

# Filtrer les dates entre le 11 et le 16 août 2024
df_filtered = df[(df['date'] >= '2024-08-11') & (df['date'] <= '2024-08-16')]

# Enregistrer le nouveau DataFrame
df_filtered.to_csv("CryptoGreedFear_filtered.csv", index=False)

# Charger les deux fichiers CSV
df1 = pd.read_csv('CryptoGreedFear_filtered.csv', parse_dates=['date'])
df2 = pd.read_csv('signaux_indicateurs.csv', parse_dates=['timestamp'])

# Extraire la date de 'timestamp' et convertir en objet date
df2['date'] = df2['timestamp'].dt.date

# Fusionner les DataFrames en castant la colonne 'date' de df1 en objet date
df_merged = pd.merge(df2, df1.assign(date=df1['date'].dt.date), on='date', how='left')

# Renommer la colonne 'fng_value' en 'sentiment'
df_merged.rename(columns={'fng_value': 'sentiment'}, inplace=True)

# Sélectionner les colonnes d'intérêt
df_result = df_merged[['timestamp', 'open', 'high', 'low', 'close', 'volume',
                       'MACD', 'MACD_signal_line', 'BB_lower', 'BB_upper', 'OBV',
                       'RSI', 'Fib_Max', 'Fib_Min', 'Fib_382', 'sentiment']]

# Enregistrer le DataFrame résultant
df_result.to_csv('fichier_avec_sentiment.csv', index=False)


import pandas as pd



# Charger les deux fichiers CSV filtré et financier
df1 = pd.read_csv('CryptoGreedFear_filtered.csv', parse_dates=['date'])
df2 = pd.read_csv('signaux_indicateurs.csv', parse_dates=['timestamp'])

# Extraire la date de 'timestamp' et convertir en objet date
df2['date'] = df2['timestamp'].dt.date

# Fusionner les DataFrames en castant la colonne 'date' de df1 en objet date
df_merged = pd.merge(df2, df1.assign(date=df1['date'].dt.date), on='date', how='left')

# Enregistrer le DataFrame fusionné
df_merged.to_csv('fichier_fusionne.csv', index=False)