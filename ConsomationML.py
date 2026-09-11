import pandas as pd

df = pd.read_csv("data/dataset_final.csv")


def nettoyer_donnees(df):

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # devient NaN : un nombre manquant.
    df["consommation"] = pd.to_numeric(
        df["consommation"],
        errors="coerce"
    )
    df = df.dropna(subset=["date"])

    df = df.sort_values("date")

    return df


def ajouter_calendrier(df):
    df = df.copy()

    df["jour_semaine"] = df["date"].dt.dayofweek

    df["est_weekend"] = (
        df["jour_semaine"] >= 5
    ).astype(int)

    df["mois"] = df["date"].dt.month


    df["heure"] = df["date"].dt.hour

    return df



