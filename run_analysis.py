import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
import os

df = pd.read_csv(r'd:\TRPL\SEM 5\PENAMBANGAN DATA\Pertemuan-5\tesla\TSLA.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 1. Dataset Trend Plot
plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Close'], label='Close Price', color='#1f77b4', linewidth=1.5)
plt.plot(df['Date'], df['Open'], label='Open Price', color='#ff7f0e', alpha=0.7, linewidth=1)
plt.title('Tren Harga Saham Tesla (TSLA) 2010 - 2020', fontsize=14, fontweight='bold')
plt.xlabel('Tahun', fontsize=12)
plt.ylabel('Harga Saham (USD)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(r'd:\TRPL\SEM 5\PENAMBANGAN DATA\LAPRAK\images\tugas-dataset-trend.png', dpi=300)
plt.close()
print('Trend plot saved.')

# Sliding window & stats
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[out_end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def stats_features(input_data):
    inp = list()
    for i in range(len(input_data)):
        inp2 = list()
        inp2 = input_data[i]
        min_v = float(np.min(inp2))
        max_v = float(np.max(inp2))
        diff = max_v - min_v
        std_v = float(np.std(inp2))
        mean_v = float(np.mean(inp2))
        median_v = float(np.median(inp2))
        kurt = float(kurtosis(inp2))
        sk = float(skew(inp2))
        inp2 = np.append(inp2, [min_v, max_v, diff, std_v, mean_v, median_v, kurt, sk])
        inp = np.append(inp, inp2)
    inp = inp.reshape(len(input_data), -1)
    return inp

def train_and_eval(X_train, X_test, y_train, y_test):
    models = {
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42),
        'KNN': KNeighborsRegressor(),
        'DT': DecisionTreeRegressor(random_state=42),
        'RF': RandomForestRegressor(random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.round(y_pred, 2)
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        corr, _ = pearsonr(y_test, y_pred)
        results[name] = {'rmse': rmse, 'corr': corr, 'y_pred': y_pred}
    return results

# Task 2
df_s2 = df[['Close', 'Close']].astype(float).values
X2, y2 = split_sequences(df_s2, 7, 2)
X2 = X2.reshape((X2.shape[0], X2.shape[1] * X2.shape[2]))
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42, shuffle=False)
X2_train = stats_features(X2_train)
X2_test = stats_features(X2_test)
res2 = train_and_eval(X2_train, X2_test, y2_train, y2_test)

# Plot Task 2 (first 100 test samples)
plt.figure(figsize=(14, 6))
plt.plot(y2_test[:100], label='Aktual (Ground Truth)', color='black', linewidth=2)
colors = {'MLP': '#1f77b4', 'KNN': '#2ca02c', 'DT': '#d62728', 'RF': '#ff7f0e'}
for name in ['MLP', 'KNN', 'DT', 'RF']:
    lbl = f"{name} (RMSE: {res2[name]['rmse']:.2f}, r: {res2[name]['corr']:.3f})"
    plt.plot(res2[name]['y_pred'][:100], label=lbl, color=colors[name], linestyle='--', alpha=0.85)
plt.title('Perbandingan Prediksi Harga Close 2 Hari ke Depan (Input: Close 7 Hari) - 100 Data Uji Pertama', fontsize=13, fontweight='bold')
plt.xlabel('Indeks Data Uji', fontsize=11)
plt.ylabel('Harga Saham (USD)', fontsize=11)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(r'd:\TRPL\SEM 5\PENAMBANGAN DATA\LAPRAK\images\tugas-soal2-prediksi.png', dpi=300)
plt.close()
print('Task 2 plot saved.')

# Task 3
df_s3 = df[['Close', 'Open', 'Close']].astype(float).values
X3, y3 = split_sequences(df_s3, 7, 2)
X3 = X3.reshape((X3.shape[0], X3.shape[1] * X3.shape[2]))
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42, shuffle=False)
X3_train = stats_features(X3_train)
X3_test = stats_features(X3_test)
res3 = train_and_eval(X3_train, X3_test, y3_train, y3_test)

# Plot Task 3 (first 100 test samples)
plt.figure(figsize=(14, 6))
plt.plot(y3_test[:100], label='Aktual (Ground Truth)', color='black', linewidth=2)
for name in ['MLP', 'KNN', 'DT', 'RF']:
    lbl = f"{name} (RMSE: {res3[name]['rmse']:.2f}, r: {res3[name]['corr']:.3f})"
    plt.plot(res3[name]['y_pred'][:100], label=lbl, color=colors[name], linestyle='--', alpha=0.85)
plt.title('Perbandingan Prediksi Harga Close 2 Hari ke Depan (Input: Close & Open 7 Hari) - 100 Data Uji Pertama', fontsize=13, fontweight='bold')
plt.xlabel('Indeks Data Uji', fontsize=11)
plt.ylabel('Harga Saham (USD)', fontsize=11)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(r'd:\TRPL\SEM 5\PENAMBANGAN DATA\LAPRAK\images\tugas-soal3-prediksi.png', dpi=300)
plt.close()
print('Task 3 plot saved.')

# Plot Comparison Bar Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
model_names = ['MLP', 'KNN', 'DT', 'RF']
x = np.arange(len(model_names))
width = 0.35

rmse_s2 = [res2[m]['rmse'] for m in model_names]
rmse_s3 = [res3[m]['rmse'] for m in model_names]
ax1.bar(x - width/2, rmse_s2, width, label='Soal 2 (Close 7 Hari)', color='#4c72b0')
ax1.bar(x + width/2, rmse_s3, width, label='Soal 3 (Close & Open 7 Hari)', color='#55a868')
ax1.set_ylabel('RMSE (Semakin rendah semakin baik)', fontsize=11)
ax1.set_title('Perbandingan Nilai RMSE', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.4, axis='y')
for i in range(len(model_names)):
    ax1.text(i - width/2, rmse_s2[i] + 0.5, f"{rmse_s2[i]:.2f}", ha='center', va='bottom', fontsize=9)
    ax1.text(i + width/2, rmse_s3[i] + 0.5, f"{rmse_s3[i]:.2f}", ha='center', va='bottom', fontsize=9)

corr_s2 = [res2[m]['corr'] for m in model_names]
corr_s3 = [res3[m]['corr'] for m in model_names]
ax2.bar(x - width/2, corr_s2, width, label='Soal 2 (Close 7 Hari)', color='#4c72b0')
ax2.bar(x + width/2, corr_s3, width, label='Soal 3 (Close & Open 7 Hari)', color='#55a868')
ax2.set_ylabel('Pearson Correlation (Semakin tinggi semakin baik)', fontsize=11)
ax2.set_title('Perbandingan Koefisien Korelasi Pearson', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, fontsize=11)
ax2.set_ylim(0.7, 1.0)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.4, axis='y')
for i in range(len(model_names)):
    ax2.text(i - width/2, corr_s2[i] + 0.005, f"{corr_s2[i]:.3f}", ha='center', va='bottom', fontsize=9)
    ax2.text(i + width/2, corr_s3[i] + 0.005, f"{corr_s3[i]:.3f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(r'd:\TRPL\SEM 5\PENAMBANGAN DATA\LAPRAK\images\tugas-evaluasi-komparasi.png', dpi=300)
plt.close()
print('Comparison bar chart saved.')

# Print markdown summary table
print('\n=== HASIL REKAPITULASI EVALUASI MODEL ===')
print('| Model | RMSE Soal 2 (Close) | Pearson Soal 2 | RMSE Soal 3 (Close & Open) | Pearson Soal 3 | Delta RMSE | Delta Pearson |')
print('|---|---|---|---|---|---|---|')
for m in model_names:
    d_rmse = res3[m]['rmse'] - res2[m]['rmse']
    d_corr = res3[m]['corr'] - res2[m]['corr']
    print(f"| {m} | {res2[m]['rmse']:.3f} | {res2[m]['corr']:.3f} | {res3[m]['rmse']:.3f} | {res3[m]['corr']:.3f} | {d_rmse:+.3f} | {d_corr:+.3f} |")
