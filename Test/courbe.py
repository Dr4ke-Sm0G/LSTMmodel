import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier CSV
file_path = "Test/predicted_closes_comparison.csv"  # Remplacez par le chemin de votre fichier
data = pd.read_csv(file_path)

# Extraire les colonnes nécessaires
predicted_close = data['predicted_close']
real_close = data['real_close']

# Tracer la dispersion et la ligne idéale
plt.figure(figsize=(8, 8))
plt.scatter(real_close, predicted_close, alpha=0.6, label='Prédictions', color='blue')
plt.plot([real_close.min(), real_close.max()], [real_close.min(), real_close.max()], 'r--', label='Ligne Idéale (y = x)', linewidth=2)

# Personnalisation du graphique
plt.title('Prédictions vs Valeurs Réelles')
plt.xlabel('Valeurs Réelles (Prix Close)')
plt.ylabel('Valeurs Prédites (Prix Close)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Afficher le graphique
plt.show()
