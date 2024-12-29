import pandas as pd
import pandas_ta as ta

def generate_signals(csv_file: str, fng_file: str) -> pd.DataFrame:
    """
    Lit le CSV, calcule les indicateurs, et ajoute les résultats.
    Gère les fichiers contenant déjà la colonne 'fng_value'.
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

    # Vérifier si la colonne fng_value existe déjà
    if 'fng_value' not in df.columns:
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
    # MACD
    macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd_df['MACD_12_26_9']
    df['MACD_signal_line'] = macd_df['MACDs_12_26_9']

    # Bandes de Bollinger (période ajustée à 14 pour un horizon court terme)
    bb_df = ta.bbands(df['close'], length=14, std=2)
    df['BB_lower'] = bb_df['BBL_14_2.0']
    df['BB_upper'] = bb_df['BBU_14_2.0']

    # OBV
    df['OBV'] = ta.obv(df['close'], df['volume'])

    # RSI (inchangé, période courte déjà optimale)
    df['RSI'] = ta.rsi(df['close'], length=7)

    # Fibonacci Retracements (période ajustée à 10 bougies)
    df['Fib_Max'] = df['close'].rolling(window=10).max()
    df['Fib_Min'] = df['close'].rolling(window=10).min()
    df['Fib_382'] = df['Fib_Max'] - 0.382 * (df['Fib_Max'] - df['Fib_Min'])

    # Réorganiser les colonnes pour mettre fng_value à la fin
    columns_order = ['timestamp', 'open', 'high', 'low', 'close', 'BB_upper', 'OBV', 'Fib_382', 'fng_value', 'volume', 'MACD', 'MACD_signal_line', 'BB_lower', 'RSI', 'Fib_Max', 'Fib_Min']
    df = df[columns_order]

    # ---------------------------------------------------------------------
    # Retourner le DataFrame avec les indicateurs calculés
    # ---------------------------------------------------------------------
    return df

if __name__ == "__main__":
    # Mettez ici le chemin vers vos fichiers CSV
    CSV_FILE = "test_eth_usdt.csv"
    FNG_FILE = "fng.csv"

    try:
        df = generate_signals(CSV_FILE, FNG_FILE)
        # Affiche les dernières lignes du DataFrame
        print(df.tail())

        # Sauvegarde du DataFrame final
        df.to_csv("test_signaux_indicateurs.csv", index=False)
    except Exception as e:
        print(f"Erreur : {e}")
