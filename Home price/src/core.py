import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns  # opzionale, non usato sotto

from sklearn.model_selection import train_test_split

import optuna
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from utility import (
    count_outliers_iqr, drop_outliers, make_preprocessor, histo_err,
    plot_scatter, optimize_xgb, rmse, prepare_ames
)

print("core.py is running")

# -----------------------
# 1) Load & base prep
# -----------------------
DROP_MANY = ["PoolQC", "Fence", "MiscFeature", "MasVnrType", "Alley", "FireplaceQu"]

train_raw = pd.read_csv(r"D:\Home price\data\train.csv")
test_raw  = pd.read_csv(r"D:\Home price\data\test.csv")

y = train_raw["SalePrice"].copy()
X_raw = train_raw.drop(columns=["SalePrice"]).copy()

# Preprocessing "grezzo" identico su train e test
X_proc      = prepare_ames(X_raw,   DROP_MANY=DROP_MANY, drop_id=True)
X_test_proc = prepare_ames(test_raw, DROP_MANY=DROP_MANY, drop_id=True)

test_id = test_raw["Id"].copy()  # per submission

# -----------------------
# 2) (OPZ) EDA / VIF su copia
# -----------------------
X_vif = X_proc.select_dtypes(include=["number"]).copy()
X_vif = X_vif.dropna().loc[:, X_vif.std() > 0]
X_vif_const = add_constant(X_vif)
vif = pd.Series(
    [variance_inflation_factor(X_vif_const.values, i) for i in range(X_vif_const.shape[1])],
    index=X_vif_const.columns
)
# print(vif.sort_values(ascending=False))

# -----------------------
# 3) Trasformazioni numeriche (log1p) scelte SOLO sul training
# -----------------------
num_cols = X_proc.select_dtypes(include=["number"])
skew_vals = num_cols.skew()

exclude = {"LuxuryHouse", "LuxuryHome"}
log_candidates = [
    c for c in num_cols.columns
    if (skew_vals[c] > 1) and (X_proc[c].min() >= 0) and (c not in exclude)
]
print("Colonne trasformate con log1p:", log_candidates)

# Applica log1p alle stesse colonne su train e test (se presenti)
X_proc[log_candidates] = np.log1p(X_proc[log_candidates])
cols_test = [c for c in log_candidates if c in X_test_proc.columns]
X_test_proc[cols_test] = np.log1p(X_test_proc[cols_test])

# Target in log-space
y_log = np.log1p(y)

# -----------------------
# 4) Split & pipeline
# -----------------------
X_tr, X_te, y_tr, y_te = train_test_split(X_proc, y_log, test_size=0.2, random_state=42)

pre = make_preprocessor(X_tr, 0)   # firma come da tuo utility
pre.fit(X_tr)

X_tr_f = pre.transform(X_tr)
X_te_f = pre.transform(X_te)

# -----------------------
# 5) Train modello (log-space)
# -----------------------
optuna.logging.set_verbosity(optuna.logging.WARNING)
study, xgb_model = optimize_xgb(
    X_tr_f, y_tr, n_trials=100, use_gpu=False, cv_splits=5, seed=42, n_jobs=6
)

# -----------------------
# 6) Valutazione hold-out
# -----------------------
pred_log_te = xgb_model.predict(X_te_f)

# Plot in log-space (pulito)
plt.scatter(np.expm1(pred_log_te), np.expm1(y_te), alpha=0.6)
mn = float(min(np.expm1(pred_log_te).min(), np.expm1(y_te).min()))
mx = float(max(np.expm1(pred_log_te).max(), np.expm1(y_te).max()))
plt.plot([mn, mx], [mn, mx], linestyle="--", label="y = x")
plt.axis("equal")
plt.xlabel("Prezzo Predetto")
plt.ylabel("Prezzo Vero")
plt.legend()
plt.title("Prezzo vero vs prezzo prodotto")
plt.show()

# Errori per fasce in scala originale
y_true = np.expm1(y_te)
y_pred = np.expm1(pred_log_te)
rel_err = np.abs(y_pred - y_true) / y_true

df_err = pd.DataFrame({"y_true": y_true, "rel_err": rel_err})
bins = np.arange(0, df_err["y_true"].max() + 100_000, 100_000)
labels = [f"{int(bins[i]/1000)}k-{int(bins[i+1]/1000)}k" for i in range(len(bins)-1)]
df_err["bin"] = pd.cut(df_err["y_true"], bins=bins, labels=labels, include_lowest=True)

mean_errors = df_err.groupby("bin")["rel_err"].mean()
mean_errors.plot(kind="bar", figsize=(10,5), edgecolor="black")
plt.ylabel("Errore relativo medio")  # adimensionale
plt.xlabel("SalePrice (fasce 100k)")
plt.title("Errore relativo medio per fascia (scala originale)")
plt.xticks(rotation=45, ha="right")
plt.show()

print("rmse",rmse(pred_log_te, y_te))

# -----------------------
# 7) Predizioni su test Kaggle + submission
# -----------------------
X_sub_f      = pre.transform(X_test_proc)
pred_log_sub = xgb_model.predict(X_sub_f)
y_pred_sub   = np.expm1(pred_log_sub)

submission = pd.DataFrame({"Id": test_id, "SalePrice": y_pred_sub})
submission.to_csv("submission.csv", index=False)
print("✅ File submission.csv creato!")
print(submission.head())
