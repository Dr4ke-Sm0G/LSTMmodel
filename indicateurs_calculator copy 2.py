import pandas as pd
import pandas_ta as ta

def generate_signals(csv_file: str) -> pd.DataFrame:
    """
    Lit le CSV, calcule les indicateurs, et ajoute les résultats
    """
    df = pd.read_csv(csv_file, parse_dates=['timestamp'])
    df.reset_index(drop=True, inplace=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

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


    # ---------------------------------------------------------------------
    # Retourner le DataFrame avec les indicateurs calculés
    # ---------------------------------------------------------------------
    return df


if __name__ == "__main__":
    # Mettez ici le chemin vers votre fichier CSV
    CSV_FILE = "btc_usdt.csv"

    df = generate_signals(CSV_FILE)
    
    # Affiche les dernières lignes du DataFrame
    print(df.tail())

    # Sauvegarde du DataFrame final
    df.to_csv("signaux_indicateurs.csv", index=False)