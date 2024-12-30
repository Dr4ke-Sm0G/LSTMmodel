import pandas as pd
import requests

# URL de l'API ou de la source de données (remplacez avec l'URL correcte)
api_url = "https://api.alternative.me/fng/?limit=366"

# Récupération des données depuis l'API
response = requests.get(api_url)
if response.status_code == 200:
    data = response.json()['data']
else:
    print("Erreur lors de la récupération des données:", response.status_code)
    exit()

# Création d'une liste structurée pour pandas
formatted_data = [
    {
        "date": item["timestamp"],
        "fng_value": int(item["value"]),
        "fng_classification": item["value_classification"]
    }
    for item in data
]

# Transformation en DataFrame pandas
df = pd.DataFrame(formatted_data)

# Formatage des colonnes
df['date'] = pd.to_datetime(df['date'], unit='s').dt.strftime('%Y-%m-%d')

# Affichage du DataFrame
print(df)

# Sauvegarde dans un fichier CSV
df.to_csv("fear_greed_index.csv", index=False)
