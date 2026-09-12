import pandas as pd
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder #transformer les noms des régions en indicateurs numériques
from sklearn.compose import ColumnTransformer #Appliquer cette transformation à la colonne région
from sklearn.pipeline import Pipeline #Apprendre une formule pour estimer la consommation
from sklearn.linear_model import LinearRegression #Enchaîner la transformation et le modèle
from sklearn.metrics import mean_absolute_error



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

df["minute"] = df["heure"].str.split(":").str[1].astype(int)



colonnes = [
    "libelle_region",
    "mois",
    "jour_semaine",
    "heure_locale",
    "minute",
    "ferie",
    "vacances_scolaires",
]
X = df[colonnes].copy()
y = df["consommation"].copy() #à apprendre

masque_train = df["date_locale"] < "2025-09-01"
masque_validation = (
    (df["date_locale"] >= "2025-09-01")
    & (df["date_locale"] < "2025-11-01")
)

masque_test = df["date_locale"] >= "2025-11-01"
X_train = X.loc[masque_train].copy()
y_train = y.loc[masque_train].copy()


X_validation = X.loc[masque_validation].copy()
y_validation = y.loc[masque_validation].copy()

#test final.
X_test = X.loc[masque_test].copy()
y_test = y.loc[masque_test].copy()


#transformer les noms des régions en indicateurs numériques
preparation = ColumnTransformer(
    transformers=[
        (
            "region",
            OneHotEncoder(handle_unknown="ignore"),#évite une erreur si une région inconnue apparaît plus tard
            ["libelle_region"],
        )
    ],
    remainder="passthrough",#conserve les autres colonnes, déjà numériques.
)

modele = Pipeline(
    steps=[
        ("preparation", preparation),
        ("regression", LinearRegression()),
    ]
)

modele.fit(X_train, y_train)

# prediction
predictions = modele.predict(X_validation)

comparaison = X_validation[["libelle_region"]].copy()
comparaison["consommation_reelle"] = y_validation
comparaison["consommation_predite"] = predictions




print("\nExemples de prédictions :")
print(comparaison.head(10).round(1))

# 3. Calculer l'erreur moyenne sur toute la validation.
mae = mean_absolute_error(y_validation, predictions)

print("\nErreur absolue moyenne :", round(mae, 2))