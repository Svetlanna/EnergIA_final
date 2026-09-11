import pandas as pd
from pathlib import Path


def preparer_donnees(df):
    df = df.copy()

    df.columns = df.columns.str.strip()

    dates_originales = df["date_locale"].astype("string").str.strip()

    dates_converties = pd.to_datetime(
        dates_originales,
        format="%Y-%m-%d",
        errors="coerce"
    )

    problemes = dates_converties.isna()

    if problemes.any():
        print("\nNombre de dates problématiques :", problemes.sum())
        print("\nExemples de valeurs originales :")
        print(dates_originales[problemes].head(20).to_string())

        raise ValueError("Vérifie les dates affichées ci-dessus.")

    df["date_locale"] = dates_converties


    correspondance = {
        "VRAI": 1,
        "FAUX": 0,
        "TRUE": 1,
        "FALSE": 0,
        "1": 1,
        "0": 0,
    }

    for colonne in ["ferie", "vacances_scolaires"]:
        texte = (
            df[colonne]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        valeurs = texte.map(correspondance)
        inconnues = texte.notna() & valeurs.isna()

        if inconnues.any():
            exemples = texte[inconnues].unique().tolist()
            raise ValueError(
                f"Valeurs non reconnues dans {colonne} : {exemples}"
            )

        df[colonne] = valeurs.astype("Int8")

    # extraire les informations du calendrier
    df["annee"] = df["date_locale"].dt.year
    df["mois"] = df["date_locale"].dt.month
    df["jour_semaine"] = df["date_locale"].dt.dayofweek

    #lundi = 0, ..., samedi = 5, dimanche = 6
    df["weekend"] = (df["jour_semaine"] >= 5).astype("int8")

    #Ajouter les saisons météorologiques
    mois = df["mois"]

    df["saison_hiver"] = mois.isin([12, 1, 2]).astype("int8")
    df["saison_printemps"] = mois.isin([3, 4, 5]).astype("int8")
    df["saison_ete"] = mois.isin([6, 7, 8]).astype("int8")
    df["saison_automne"] = mois.isin([9, 10, 11]).astype("int8")

    #convertir les mesures en nombres
    for colonne in [
        "consommation",
        "temperature_2m",
        "relative_humidity_2m",
        "heure_locale",
    ]:
        texte = (
            df[colonne]
            .astype("string")
            .str.strip()
            .str.replace(",", ".", regex=False)
        )

        df[colonne] = pd.to_numeric(texte, errors="coerce")

    return df

# __file__ contient le chemin du script Python.
dossier_script = Path(__file__).resolve().parent

chemin_csv = dossier_script / "data" / "dataset_final.csv"


df = pd.read_csv(
    chemin_csv,
    sep=None,
    engine="python",
    encoding="utf-8-sig"
)

df.columns = df.columns.str.strip()


df = preparer_donnees(df)





# print(df[[
#     "date_locale",
#     "libelle_region",
#     "consommation",
#     "jour_semaine",
#     "weekend",
#     "ferie",
#     "vacances_scolaires",
#     "saison_hiver",
# ]])

# print("\nValeurs manquantes :")

print(df[[
    "consommation",
    "ferie",
    "vacances_scolaires",
    "heure_locale",
]])