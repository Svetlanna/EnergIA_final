import pandas as pd

df = pd.read_csv("data/dataset_final.csv")

df["datetime"] = pd.to_datetime(
    df["date"].astype(str) + " " + df["heure"].astype(str)
)

df["mois"] = df["datetime"].dt.month
df["jour"] = df["datetime"].dt.day
df["jour_semaine"] = df["datetime"].dt.dayofweek
df["heure_num"] = df["datetime"].dt.hour
df["minute"] = df["datetime"].dt.minute

df["weekend"] = df["jour_semaine"].isin([5, 6])

def get_saison(mois):
    if mois in [12, 1, 2]:
        return "hiver"
    elif mois in [3, 4, 5]:
        return "printemps"
    elif mois in [6, 7, 8]:
        return "ete"
    else:
        return "automne"

df["saison"] = df["mois"].apply(get_saison)

def get_feries(ferie):
    if ferie == True:
        return 1
    else:
        return 0

df["ferie"] = df["ferie"].apply(get_feries)
print(df["ferie"].sum())