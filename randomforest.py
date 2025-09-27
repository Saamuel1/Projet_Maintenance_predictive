# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 20:22:31 2025

@author: samue
"""

# predictive_maintenance_fd001.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ===============================
# 1) Définir colonnes
# ===============================
col_names = (
    ['unit','cycle'] +
    [f'set{i}' for i in range(1,4)] +   # 3 settings
    [f's{i}' for i in range(1,22)]      # 21 capteurs
)

# ===============================
# 2) Charger les données
# ===============================
train = pd.read_csv("train_FD001.txt", sep="\\s+", header=None, names=col_names)
test  = pd.read_csv("test_FD001.txt",  sep="\\s+", header=None, names=col_names)
rul   = pd.read_csv("RUL_FD001.txt",   sep="\\s+", header=None, names=['RUL'])

print("Train shape:", train.shape)
print("Test shape :", test.shape)
print("RUL shape  :", rul.shape)

# ===============================
# 3) Créer RUL pour train
# ===============================
max_cycles = train.groupby('unit')['cycle'].max().reset_index()
max_cycles.columns = ['unit','max_cycle']
train = train.merge(max_cycles, on='unit', how='left')
train['RUL'] = train['max_cycle'] - train['cycle']

# ===============================
# 3) Créer RUL pour test
# ===============================
max_cycles_test = test.groupby('unit')['cycle'].max().reset_index()
max_cycles_test.columns = ['unit','max_cycle']
test = test.merge(max_cycles_test, on='unit', how='left')
rul['unit'] = rul.index + 1
test = test.merge(rul, on='unit', how='left')
test['RUL'] = test['RUL'] + (test['max_cycle'] - test['cycle'])

#%%

# ===============================
# 4) Features simples
# ===============================
FEATURES = [c for c in train.columns if c.startswith("s")]  # uniquement capteurs

X_train = train[FEATURES]
y_train = train['RUL']

X_test  = test[FEATURES]
y_test  = test['RUL']

# ===============================
# 5) Modèle RandomForest
# ===============================
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# ===============================
# 6) Évaluation
# ===============================
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)

print(f"Baseline RandomForest → RMSE = {rmse:.2f}, MAE = {mae:.2f}")

#%%
# ===============================
# 7) Visualisation résultats
# ===============================
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Calcul des métriques
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

# ---------------------------------
# Scatter plot : RUL prédite vs réelle
# ---------------------------------
plt.figure(figsize=(7,7))
sns.kdeplot(x=y_test, y=y_pred, cmap="Blues", fill=True, thresh=0.05, levels=20, alpha=0.6)
plt.scatter(y_test, y_pred, alpha=0.15, s=10, color="darkblue", label="Prédictions")
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--', label="Idéal (y=x)")
plt.xlabel("RUL réelle", fontsize=12)
plt.ylabel("RUL prédite", fontsize=12)
plt.title("RandomForest - RUL prédite vs réelle", fontsize=14, weight="bold")
plt.legend()
plt.text(10, max(y_pred)*0.9, f"RMSE={rmse:.1f}\nMAE={mae:.1f}\nR²={r2:.2f}", 
         fontsize=11, bbox=dict(facecolor="white", alpha=0.7))
plt.grid(True, alpha=0.3)
plt.show()

# ---------------------------------
# Exemple : suivre un moteur du test
# ---------------------------------
unit_id = 1
df_unit = test[test['unit']==unit_id].copy()
df_unit['pred'] = rf.predict(df_unit[FEATURES])

plt.figure(figsize=(9,5))
plt.plot(df_unit['cycle'], df_unit['RUL'], label="RUL réelle", color="steelblue", linewidth=2)
plt.plot(df_unit['cycle'], df_unit['pred'], label="RUL prédite", color="orange", linestyle="--", linewidth=2)
plt.fill_between(df_unit['cycle'], 
                 df_unit['pred']-10, df_unit['pred']+10, 
                 color="orange", alpha=0.2, label="±10 cycles (incertitude)")

plt.title(f"Moteur {unit_id} - Suivi RUL (RandomForest)", fontsize=14, weight="bold")
plt.xlabel("Cycle", fontsize=12)
plt.ylabel("RUL", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

