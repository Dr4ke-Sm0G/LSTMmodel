import pandas as pd

# Charger le fichier CSV
file_path = 'test_eth_usdt.csv'
df = pd.read_csv(file_path)

# Ajouter une nouvelle colonne 'fng_value' avec la valeur 73 pour toutes les lignes
df['fng_value'] = 73

# Sauvegarder le fichier CSV avec les modifications
output_file_path = 'test_eth_usdt.csv'
df.to_csv(output_file_path, index=False)

print(f"Une colonne 'fng_value' avec la valeur 73 a été ajoutée. Le fichier modifié a été sauvegardé sous '{output_file_path}'.")
