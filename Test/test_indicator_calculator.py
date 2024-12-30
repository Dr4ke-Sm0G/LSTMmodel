import pandas as pd
import pandas_ta as ta

def generate_signals(csv_file: str, fng_file: str) -> pd.DataFrame:
    """
    Lit le CSV, calcule les indicateurs, et ajoute les résultats
    """
    # Chargement des données de prix
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)  # Convertir en datetime avec UTC
    if df['timestamp'].isnull().any():
        invalid_timestamps = df[df['timestamp'].isnull()]
        print("Lignes avec timestamps invalides :")
        print(invalid_timestamps)
        raise ValueError("La colonne 'timestamp' contient des valeurs non convertibles en datetime.")

    df.reset_index(drop=True, inplace=True)

    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # Chargement des données FNG
    fng_df = pd.read_csv(fng_file, parse_dates=['date'])
    fng_df['date'] = fng_df['date'].dt.date  # Extraire uniquement la partie date

    # Ajouter la colonne fng_value au DataFrame principal
    df['date_only'] = df['timestamp'].dt.date  # Extraire uniquement la date
    df = pd.merge(df, fng_df[['date', 'fng_value']], left_on='date_only', right_on='date', how='left')
    df.drop(columns=['date', 'date_only'], inplace=True)  # Supprimer les colonnes inutiles

    # ---------------------------------------------------------------------
    # Calcul des indicateurs
    # ---------------------------------------------------------------------

    # Bandes de Bollinger (période ajustée à 14 pour un horizon court terme)
    bb_df = ta.bbands(df['close'], length=14, std=2)
    df['BB_upper'] = bb_df['BBU_14_2.0']

    # OBV
    df['OBV'] = ta.obv(df['close'], df['volume'])

    # Fibonacci Retracements (période ajustée à 10 bougies)
    df['Fib_Max'] = df['close'].rolling(window=10).max()
    df['Fib_Min'] = df['close'].rolling(window=10).min()
    df['Fib_382'] = df['Fib_Max'] - 0.382 * (df['Fib_Max'] - df['Fib_Min'])

    # Réorganiser les colonnes pour mettre fng_value à la fin
    #columns_order = ['timestamp', 'open', 'high', 'low', 'close', 'BB_lower', 'OBV', 'Fib_382', 'fng_value']

    columns_order = ['timestamp', 'open', 'high', 'low', 'close', 'BB_upper', 'OBV', 'Fib_382', 'fng_value']
    df = df[columns_order[0:9]]

    # ---------------------------------------------------------------------
    # Retourner le DataFrame avec les indicateurs calculés
    # ---------------------------------------------------------------------
    return df

if __name__ == "__main__":
    # Mettez ici le chemin vers vos fichiers CSV
    CSV_FILE = "Test/test_eth_usdt.csv"
    FNG_FILE = "fear_greed_index.csv"

    try:
        df = generate_signals(CSV_FILE, FNG_FILE)
        # Affiche les dernières lignes du DataFrame
        print(df.tail())

        # Sauvegarde du DataFrame final
        df.to_csv("Test/test_signaux_indicateurs.csv", index=False)
    except Exception as e:
        print(f"Erreur : {e}")

