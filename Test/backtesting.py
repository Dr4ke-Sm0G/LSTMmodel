import pandas as pd

def backtest_trading_strategy(file_path, initial_capital, buy_threshold, sell_threshold, stop_loss_threshold, trading_fee):
    # Charger les données
    data = pd.read_csv(file_path)
    
    # Initialisation des variables
    capital = initial_capital
    position = 0  # Nombre d'ETH en possession
    entry_price = 0  # Prix d'achat de l'ETH
    trade_log = []  # Journal des trades
    
    # Parcourir les données pour simuler les trades
    for i in range(len(data) - 1):
        current_close = data.loc[i, 'real_close']
        next_predicted_close = data.loc[i + 1, 'predicted_close']
        
        # Stratégie d'achat : acheter si la prochaine prédiction est supérieure à la bougie actuelle de 1.5%
        if position == 0 and next_predicted_close > current_close * (1 + buy_threshold):
            # Calculer les frais d'achat
            fee = capital * trading_fee
            capital_after_fee = capital - fee

            # Acheter avec le capital restant après frais
            position = capital_after_fee / current_close  # Quantité d'ETH achetée
            entry_price = current_close  # Enregistrer le prix d'achat
            capital = 0  # Tout le capital est investi
            trade_log.append({'action': 'buy', 'price': current_close, 'quantity': position, 'fee': fee, 'index': i})
        
        # Stratégie de vente ou Stop-Loss
        elif position > 0:
            # Stop-Loss : vendre si le prix réel tombe en dessous de 3% du prix d'achat
            if current_close < entry_price * (1 - stop_loss_threshold):
                # Calculer la valeur de vente et les frais
                sell_value = position * current_close
                fee = sell_value * trading_fee
                capital = sell_value - fee  # Capital net après frais
                trade_log.append({'action': 'stop-loss', 'price': current_close, 'quantity': position, 'fee': fee, 'index': i})
                position = 0  # Réinitialiser la position
                entry_price = 0  # Réinitialiser le prix d'achat
            
            # Vendre si la prédiction de la prochaine bougie est inférieure au prix d'achat de 1%
            elif next_predicted_close < entry_price * (1 - sell_threshold):
                # Calculer la valeur de vente et les frais
                sell_value = position * current_close
                fee = sell_value * trading_fee
                capital = sell_value - fee  # Capital net après frais
                trade_log.append({'action': 'sell', 'price': current_close, 'quantity': position, 'fee': fee, 'index': i})
                position = 0  # Réinitialiser la position
                entry_price = 0  # Réinitialiser le prix d'achat
    
    # Calcul de la valeur finale
    final_capital = capital + (position * data.loc[len(data) - 1, 'real_close'] if position > 0 else 0)
    return trade_log, final_capital

# Configuration
file_path = "Test/predicted_closes_comparison.csv"  # Remplacez par le chemin de votre fichier
initial_capital = 1000  # Capital initial en euros
buy_threshold = 0.015  # Seuil d'achat (1.5%)
sell_threshold = 0.015  # Seuil de vente (1%)
stop_loss_threshold = 0.05  # Seuil de Stop-Loss (3%)
trading_fee = 0.001  # Frais de trading (0.1%)

# Exécution du backtesting
trade_log, final_capital = backtest_trading_strategy(
    file_path, initial_capital, buy_threshold, sell_threshold, stop_loss_threshold, trading_fee
)

# Afficher les résultats
print(f"Capital final : {final_capital:.2f} €")
print("Journal des trades :")
for trade in trade_log:
    print(trade)
