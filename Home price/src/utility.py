import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from typing import List, Dict

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.compose import TransformedTargetRegressor

from xgboost import XGBRegressor

# funzione che conta gli outlier > 3 sigma
def count_outliers_iqr(x):
    q1 = x.quantile(0.25)   # 25° percentile
    q3 = x.quantile(0.75)   # 75° percentile
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((x < lower) | (x > upper)).sum()

def drop_outliers(group):
    q1=group["SalePrice"].quantile(0.25)
    q3=group["SalePrice"].quantile(0.75)
    iqr=q3-q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return group[(group["SalePrice"]>=lower) & (group["SalePrice"]<=upper)]


def make_preprocessor(df: pd.DataFrame, use_scaler: bool = True, do_impute: bool = True,
                      sklearn_new: bool = True):
    # 1) tipologie base
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # escludi bool dai numerici (verranno trattati come categorici binari)
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    for c in bool_cols:
        if c in num_cols:
            num_cols.remove(c)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist() + bool_cols

    # 3) Ordinali con ordine esplicito (Ames)
    qual_like = ["ExterQual","ExterCond","KitchenQual","FireplaceQu",
                 "BsmtQual","BsmtCond","GarageQual","GarageCond","HeatingQC"]
    bsmt_exposure = ["BsmtExposure"]
    bsmt_fintype = ["BsmtFinType1","BsmtFinType2"]
    ordinal_cols = [c for c in (qual_like + bsmt_exposure + bsmt_fintype) if c in df.columns]

    # correzioni nomi note
    # (se nel tuo df ci fossero varianti errate, rinominale prima)

    # categorie in ordine
    qual_order = ["Po","Fa","TA","Gd","Ex"]
    exposure_order = ["No","Mn","Av","Gd"]
    finish_order = ["Unf","LwQ","Rec","BLQ","ALQ","GLQ"]

    categories_map: Dict[str, List[str]] = {}
    for c in ordinal_cols:
        if c in qual_like:
            categories_map[c] = qual_order
        elif c in bsmt_exposure:
            categories_map[c] = exposure_order
        elif c in bsmt_fintype:
            categories_map[c] = finish_order

    # CentralAir è binaria → OHE, non ordinale
    if "CentralAir" in ordinal_cols:
        ordinal_cols.remove("CentralAir")
        if "CentralAir" not in cat_cols:
            cat_cols.append("CentralAir")

    # 4) cat OHE = resto dei categorici non ordinali
    cat_ohe_cols = [c for c in cat_cols if c not in ordinal_cols]

    # 5) Pipeline numerici
    if do_impute:
        num_pipe_steps = [("impute", SimpleImputer(strategy="median"))]
        if use_scaler:
            num_pipe_steps += [("scale", StandardScaler())]
        num_pipe = Pipeline(num_pipe_steps)
    else:
        num_pipe = Pipeline([("scale", StandardScaler())]) if use_scaler else "passthrough"

    # 6) Pipeline categorici OHE
    # per compatibilità versioni sklearn
    ohe_kwargs = dict(handle_unknown="ignore", min_frequency=0.01)
    if sklearn_new:
        ohe_kwargs["sparse_output"] = False
    else:
        ohe_kwargs["sparse"] = False

    if do_impute:
        cat_ohe_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(**ohe_kwargs)),
        ])
    else:
        cat_ohe_pipe = OneHotEncoder(**ohe_kwargs)

    # 7) Pipeline ordinali
    # Se vuoi che i NaN diventino 0 (più basso), imputa prima a una label “Missing”
    # oppure usa OrdinalEncoder con unknown/missing a un valore fuori scala.
    if do_impute:
        ord_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(
                categories=[categories_map[c] for c in ordinal_cols],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1
            )),
        ])
    else:
        ord_pipe = OrdinalEncoder(
            categories=[categories_map[c] for c in ordinal_cols],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1
        )

    pre = ColumnTransformer(
        transformers=[
            ("num",     num_pipe,     num_cols),
            ("cat_ohe", cat_ohe_pipe, cat_ohe_cols),
            ("cat_ord", ord_pipe,     ordinal_cols),
        ],
        remainder="drop"
    )
    return pre


def histo_err(y_pred,y_test):
    y_true = np.expm1(y_test)
    y_pred_lin = np.expm1(y_pred)
    #errors = np.abs(y_pred_lin - y_true)
    errors = np.sqrt((y_pred_lin - y_true) ** 2)/y_true

    df_err = pd.DataFrame({"y_true": y_true, "error": errors})

    bins = np.arange(0, df_err["y_true"].max() + 100_000, 100_000)
    labels = [f"{int(bins[i]/1000)}k-{int(bins[i+1]/1000)}k" for i in range(len(bins)-1)]
    df_err["bin"] = pd.cut(df_err["y_true"], bins=bins, labels=labels, include_lowest=True)

    mean_errors = df_err.groupby("bin")["error"].mean()

    mean_errors.plot(kind="bar", figsize=(10,5), edgecolor="black")
    plt.ylabel("RMSE/true_value (in €)")
    plt.xlabel("SalePrice (fasce 100k)")
    plt.title("Errore medio per fasce (scala originale)")
    plt.xticks(rotation=45, ha="right")
    plt.show()

def plot_scatter(y_pred,y_test): 
    plt.scatter(y_pred,y_test)

    # Genera valori
    x = np.logspace(0.9, 1.2, 1000)
    y = x  # retta a 45°

    plt.plot(x, y, color="red", linestyle="--", label="y = x")

    # imposta assi uguali per vedere davvero i 45°
    plt.axis("equal")
    plt.xlabel("y_predette")
    plt.ylabel("y_vere")
    plt.legend()
    plt.title("Retta a 45° (y = x)")
    plt.show()

def optimize_xgb(
    X, y,
    n_trials: int = 100,
    use_gpu: bool = False,
    cv_splits: int = 5,
    seed: int = 42,
    n_jobs: int = 6):
    # Per evitare clash con vecchie sessioni
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.trial.Trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 500, 4000),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_float("min_child_weight", 1.0, 20.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma":             trial.suggest_float("gamma", 0.0, 10.0),
            # fissi
            "random_state":      seed,
            "n_jobs":            n_jobs,
            "objective":         "reg:squarederror",
            "eval_metric":       "rmse",
            "tree_method":       "gpu_hist" if use_gpu else "hist",
            "predictor":         "gpu_predictor" if use_gpu else "auto",
            "verbosity":         0,
        }

        model = XGBRegressor(**params)

        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        # sklearn usa "neg_root_mean_squared_error" → restituisce valori negativi
        scores = cross_val_score(
            model, X, y,
            scoring="neg_root_mean_squared_error",
            cv=kf,
            n_jobs=n_jobs
        )
        rmse_mean = float(-scores.mean())
        return rmse_mean

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Fit finale con i migliori iperparametri su TUTTO X,y
    best_params = study.best_trial.params.copy()
    best_params.update({
        "random_state": seed,
        "n_jobs": n_jobs,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "gpu_hist" if use_gpu else "hist",
        "predictor": "gpu_predictor" if use_gpu else "auto",
        "verbosity": 0,
    })
    best_model = XGBRegressor(**best_params)
    best_model.fit(X, y)

    return study, best_model


def rmse(y_true, y_pred):
    #y_pred_log = np.log1p(y_pred)
    return np.sqrt(mean_squared_error(y_true, y_pred))

DROP_MANY = ["PoolQC", "Fence", "MiscFeature", "MasVnrType", "Alley", "FireplaceQu"]

def prepare_ames(df_in: pd.DataFrame, *, DROP_MANY=DROP_MANY, drop_id=True) -> pd.DataFrame:
    """
    Pulisce e arricchisce il dataset Ames (train o test) con imputazioni manuali e feature engineering.
    - Non tocca SalePrice (se presente, resta dove sta: decidilo fuori).
    - Se drop_id=True, droppa la colonna Id se presente.
    - Imputa LotFrontage con la mediana per Neighborhood (senza leakage esterno: usa solo il df passato).
    Ritorna un DataFrame pronto per il preprocessing/pipeline.
    """
    
    df = df_in.copy()

    # --- drop colonne "rumorose"/alta missingness + Id opzionale ---
    to_drop = [c for c in DROP_MANY if c in df.columns]
    if drop_id and "Id" in df.columns:
        to_drop.append("Id")
    df.drop(columns=to_drop, inplace=True, errors="ignore")

    # --- imputazioni categoriche "NA" per assenza fisica/qualitativa ---
    cat_na = ["BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
              "GarageType","GarageFinish","GarageCond","GarageQual"]
    for c in cat_na:
        if c in df.columns:
            df[c] = df[c].fillna("NA")

    # --- imputazioni numeriche semplici by-regola ---
    if "MasVnrArea" in df.columns:
        df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

    # --- GarageYrBlt: flag missing + mantieni il NaN per il calcolo di GarageAge ---
    if "GarageYrBlt" in df.columns:
        df["GarageYrBlt_missing"] = df["GarageYrBlt"].isna().astype(int)

    # --- LotFrontage: mediana per Neighborhood (intra-df) ---
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        med_by_nbhd = df.groupby("Neighborhood")["LotFrontage"].transform("median")
        df["LotFrontage"] = df["LotFrontage"].fillna(med_by_nbhd)
        # se qualche NA resta (neighborhood tutto NA), usa mediana globale
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # --- feature derivate di età (robuste) ---
    # garantisci colonne richieste
    for col, default in [("YrSold", np.nan), ("YearBuilt", np.nan), ("YearRemodAdd", np.nan)]:
        if col not in df.columns:
            df[col] = default

    # Age e RemodAge: clip a >= 0 quando possibile
    df["Age"]      = (df["YrSold"] - df["YearBuilt"]).where(df["YrSold"].notna() & df["YearBuilt"].notna())
    df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).where(df["YrSold"].notna() & df["YearRemodAdd"].notna())
    df["Age"]      = df["Age"].clip(lower=0)
    df["RemodAge"] = df["RemodAge"].clip(lower=0)

    # GarageAge: usa GarageYrBlt originale (senza riempire 0), così i NaN restano NaN (più sensato)
    if "GarageYrBlt" in df.columns:
        df["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"]).where(df["YrSold"].notna() & df["GarageYrBlt"].notna())
        df["GarageAge"] = df["GarageAge"].clip(lower=0)

    # --- total surfaces ---
    for c in ["TotalBsmtSF","1stFlrSF","2ndFlrSF"]:
        if c not in df.columns:
            df[c] = 0
    df["TotalSF"] = df[["TotalBsmtSF","1stFlrSF","2ndFlrSF"]].sum(axis=1)

    # --- total bathrooms (0.5 per half) ---
    for c in ["BsmtFullBath","BsmtHalfBath","FullBath","HalfBath"]:
        if c not in df.columns:
            df[c] = 0
    df["TotalBath"] = (df["BsmtFullBath"] + 0.5*df["BsmtHalfBath"] +
                       df["FullBath"]     + 0.5*df["HalfBath"])

    # --- total porches ---
    for c in ["OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch"]:
        if c not in df.columns:
            df[c] = 0
    df["TotalPorchSF"] = df[["OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch"]].sum(axis=1)

    # --- rapporti/derivate robuste ---
    if "LotArea" in df.columns and "LotFrontage" in df.columns:
        denom = df["LotFrontage"].replace({0: np.nan})
        df["LotAreaRatio"] = df["LotArea"] / denom
    else:
        df["LotAreaRatio"] = np.nan

    if "TotRmsAbvGrd" in df.columns:
        denom = df["TotRmsAbvGrd"].replace({0: np.nan})
        df["Bath_over_bedrooms"] = df["TotalBath"] / denom
    else:
        df["Bath_over_bedrooms"] = np.nan

    if "GrLivArea" in df.columns and "OverallQual" in df.columns:
        df["Area_per_Qual"] = df["GrLivArea"] * df["OverallQual"]
    else:
        df["Area_per_Qual"] = np.nan

    # --- LuxuryHome binaria (criteri invariati) ---
    req = ["GrLivArea","OverallQual","GarageCars","TotalBsmtSF","Age","TotalSF","LotFrontage","TotalBath","TotRmsAbvGrd"]
    for r in req:
        if r not in df.columns:
            df[r] = 0
    df["LuxuryHome"] = (
        (df["GrLivArea"] > 2000) &
        (df["OverallQual"] >= 9) &
        (df["GarageCars"] >= 3) &
        (df["TotalBsmtSF"] > 1000) &
        (df["Age"] <= 20) &
        (df["TotalSF"] >= 4200) &
        (df["LotFrontage"] >= 80) &
        (df["TotalBath"] >= 3) &
        (df["TotRmsAbvGrd"] >= 10)
    ).astype(int)

    return df