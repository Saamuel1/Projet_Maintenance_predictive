# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 21:56:52 2025

@author: samue
"""

# lstm_fd001_improved.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# ===============================
# 1) Charger les données
# ===============================
col_names = ['unit','cycle'] + [f'set{i}' for i in range(1,4)] + [f's{i}' for i in range(1,22)]

train = pd.read_csv("train_FD001.txt", sep="\s+", header=None, names=col_names)
test  = pd.read_csv("test_FD001.txt",  sep="\s+", header=None, names=col_names)
rul   = pd.read_csv("RUL_FD001.txt",   sep="\s+", header=None, names=['RUL'])

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

# ===============================
# 4) Préparer séquences
# ===============================
FEATURES = [c for c in train.columns if c.startswith("s")]
scaler = MinMaxScaler()

train_scaled = train.copy()
train_scaled[FEATURES] = scaler.fit_transform(train[FEATURES])

test_scaled = test.copy()
test_scaled[FEATURES] = scaler.transform(test[FEATURES])

def create_sequences(df, seq_length=50):
    X, y = [], []
    for unit_id in df['unit'].unique():
        df_unit = df[df['unit']==unit_id]
        data = df_unit[FEATURES].values
        labels = df_unit['RUL'].values
        for i in range(len(df_unit) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(labels[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LEN = 50
X_train, y_train = create_sequences(train_scaled, SEQ_LEN)
X_test,  y_test  = create_sequences(test_scaled,  SEQ_LEN)

print("X_train:", X_train.shape, "y_train:", y_train.shape)

# ===============================
# 5) Modèle LSTM amélioré
# ===============================
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, len(FEATURES)))),
    Dropout(0.3),
    LSTM(64),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# ===============================
# 6) Entraînement avec early stopping
# ===============================
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    verbose=1,
    callbacks=[es]
)

# ===============================
# 7) Évaluation
# ===============================
y_pred = model.predict(X_test).flatten()

rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)

print(f"LSTM amélioré → RMSE = {rmse:.2f}, MAE = {mae:.2f}")

# ===============================
# 8) Visualisations LSTM amélioré
# ===============================
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)


# ---------------------------------
# Courbe d'apprentissage
# ---------------------------------
plt.figure(figsize=(7,4))
plt.plot(history.history['loss'], label="Train loss", color="steelblue")
plt.plot(history.history['val_loss'], label="Validation loss", color="orange")
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("MSE", fontsize=12)
plt.title("Courbe d'entraînement LSTM amélioré", fontsize=14, weight="bold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ---------------------------------
# Scatter plot préd vs réel
# ---------------------------------
plt.figure(figsize=(7,7))
sns.kdeplot(x=y_test, y=y_pred, cmap="Purples", fill=True, thresh=0.05, levels=20, alpha=0.6)
plt.scatter(y_test, y_pred, alpha=0.15, s=10, color="purple", label="Prédictions")
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--', label="Idéal (y=x)")
plt.xlabel("RUL réelle", fontsize=12)
plt.ylabel("RUL prédite", fontsize=12)
plt.title("LSTM amélioré - RUL prédite vs réelle", fontsize=14, weight="bold")
plt.legend()
plt.text(10, max(y_pred)*0.9, f"RMSE={rmse:.1f}\nMAE={mae:.1f}\nR²={r2:.2f}",
         fontsize=11, bbox=dict(facecolor="white", alpha=0.7))
plt.grid(True, alpha=0.3)
plt.show()
#%%
# ---------------------------------
# Exemple moteur 1 (LSTM amélioré)
# ---------------------------------
unit_id = 10
df_unit = test_scaled[test_scaled['unit']==unit_id].copy()
X_unit, y_unit = create_sequences(df_unit, SEQ_LEN)

if len(X_unit) > 0:
    # Prédictions
    y_unit_pred = model.predict(X_unit).flatten()
    
    # Graphique
    plt.figure(figsize=(11,6))
    plt.plot(df_unit['cycle'][SEQ_LEN:], y_unit, 
             label="RUL réelle", color="steelblue", linewidth=2)
    plt.plot(df_unit['cycle'][SEQ_LEN:], y_unit_pred, 
             label="RUL prédite", color="purple", linestyle="--", linewidth=2)
    plt.fill_between(df_unit['cycle'][SEQ_LEN:], 
                     y_unit_pred-10, y_unit_pred+10,
                     color="purple", alpha=0.2, label="±10 cycles (incertitude)")

    plt.title(f"Moteur 1 - Suivi RUL (LSTM amélioré)", fontsize=16, weight="bold")
    plt.xlabel("Cycle", fontsize=13)
    plt.ylabel("RUL", fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print(f"⚠️ Pas assez de cycles pour créer des séquences (unit {unit_id}, SEQ_LEN={SEQ_LEN})")
#%%

plt.figure(figsize=(9,5))
plt.plot(df_unit['cycle'][SEQ_LEN:], y_unit, label="RUL réelle", color="steelblue", linewidth=2)
plt.plot(df_unit['cycle'][SEQ_LEN:], y_unit_pred, label="RUL prédite", color="purple", linestyle="--", linewidth=2)
plt.fill_between(df_unit['cycle'][SEQ_LEN:], 
                 y_unit_pred-10, y_unit_pred+10,
                 color="purple", alpha=0.2, label="±10 cycles (incertitude)")

plt.title(f"Moteur {unit_id} - Suivi RUL (LSTM amélioré)", fontsize=14, weight="bold")
plt.xlabel("Cycle", fontsize=12)
plt.ylabel("RUL", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

