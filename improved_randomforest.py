# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 21:56:35 2025

@author: samue
"""

# improved_randomforest_fd001.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ===============================
# 1) Charger les données
# ===============================
col_names = ['unit','cycle'] + [f'set{i}' for i in range(1,4)] + [f's{i}' for i in range(1,22)]

train = pd.read_csv("train_FD001.txt", sep="\\s+", header=None, names=col_names)
test  = pd.read_csv("test_FD001.txt",  sep="\\s+", header=None, names=col_names)
rul   = pd.read_csv("RUL_FD001.txt",   sep="\\s+", header=None, names=['RUL'])

# ===============================
# 2) Créer RUL train
# ===============================
max_cycles = train.groupby('unit')['cycle'].max().reset_index()
max_cycles.columns = ['unit','max_cycle']
train = train.merge(max_cycles, on='unit', how='left')
train['RUL'] = train['max_cycle'] - train['cycle']

# ===============================
# 3) Créer RUL test
# ===============================
max_cycles_test = test.groupby('unit')['cycle'].max().reset_index()
max_cycles_test.columns = ['unit','max_cycle']
test = test.merge(max_cycles_test, on='unit', how='left')
rul['unit'] = rul.index + 1
test = test.merge(rul, on='unit', how='left')
test['RUL'] = test['RUL'] + (test['max_cycle'] - test['cycle'])
#%%
# ===============================
# 4) Fonction features glissantes
# ===============================
def add_rolling_features(df, window=30):
    FEATURES = [c for c in df.columns if c.startswith("s")]  # capteurs uniquement
    df_feat = df.copy()
    for col in FEATURES:
        df_feat[f"{col}_mean"] = df_feat.groupby("unit")[col].transform(lambda x: x.rolling(window, min_periods=5).mean())
        df_feat[f"{col}_std"]  = df_feat.groupby("unit")[col].transform(lambda x: x.rolling(window, min_periods=5).std())
    df_feat = df_feat.dropna()  # supprime les débuts de séries
    return df_feat

train_feat = add_rolling_features(train, window=30)
test_feat  = add_rolling_features(test,  window=30)

# ===============================
# 5) Préparer X, y
# ===============================
FEATURES = [c for c in train_feat.columns if c.startswith("s")]
X_train = train_feat[FEATURES]
y_train = train_feat['RUL']

X_test  = test_feat[FEATURES]
y_test  = test_feat['RUL']

# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
#%%
# ===============================
# 6) Modèle RandomForest amélioré
# ===============================
rf = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# ===============================
# 7) Évaluation
# ===============================
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)

print(f"RandomForest + rolling features → RMSE = {rmse:.2f}, MAE = {mae:.2f}")

#%%
# ===============================
# 8) Visualisations RF amélioré
# ===============================
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Évaluer modèle
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

# ---------------------------------
# Scatter plot
# ---------------------------------
plt.figure(figsize=(7,7))
sns.kdeplot(x=y_test, y=y_pred, cmap="Blues", fill=True, thresh=0.05, levels=20, alpha=0.6)
plt.scatter(y_test, y_pred, alpha=0.15, s=10, color="darkblue", label="Prédictions")
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--', label="Idéal (y=x)")
plt.xlabel("RUL réelle", fontsize=12)
plt.ylabel("RUL prédite", fontsize=12)
plt.title("RandomForest amélioré - RUL prédite vs réelle", fontsize=14, weight="bold")
plt.legend()
plt.text(10, max(y_pred)*0.9, f"RMSE={rmse:.1f}\nMAE={mae:.1f}\nR²={r2:.2f}", 
         fontsize=11, bbox=dict(facecolor="white", alpha=0.7))
plt.grid(True, alpha=0.3)
plt.show()

# ---------------------------------
# Exemple moteur 1
# ---------------------------------
unit_id = 1
df_unit = test_feat[test_feat['unit']==unit_id].copy()
df_unit['pred'] = rf.predict(scaler.transform(df_unit[FEATURES]))

plt.figure(figsize=(9,5))
plt.plot(df_unit['cycle'], df_unit['RUL'], label="RUL réelle", color="steelblue", linewidth=2)
plt.plot(df_unit['cycle'], df_unit['pred'], label="RUL prédite", color="orange", linestyle="--", linewidth=2)
plt.fill_between(df_unit['cycle'],
                 df_unit['pred']-10, df_unit['pred']+10,
                 color="orange", alpha=0.2, label="±10 cycles (incertitude)")

plt.title(f"Moteur {unit_id} - Suivi RUL (RandomForest amélioré)", fontsize=14, weight="bold")
plt.xlabel("Cycle", fontsize=12)
plt.ylabel("RUL", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
