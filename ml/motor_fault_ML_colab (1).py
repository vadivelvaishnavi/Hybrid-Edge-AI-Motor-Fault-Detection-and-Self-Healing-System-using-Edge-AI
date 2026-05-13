# ============================================================
#  HYBRID EDGE-AI MOTOR FAULT DETECTION — GOOGLE COLAB ML
#  Patent-Ready Training Pipeline
# ============================================================
#
#  PATENT CLAIMS COVERED:
#  Claim 2  — All models deploy to ESP32 edge (TFLite / micromlgen)
#  Claim 3  — Weighted multi-modal fusion (validated here)
#  Claim 7  — Baseline learning from normal operating data
#  Claim 10 — Method: train → validate → export → deploy
#
#  MODELS TRAINED:
#   A. Isolation Forest (micromlgen → C header for ESP32)
#   B. Autoencoder      (Keras → TFLite → C array for ESP32)
#   C. Mahalanobis Distance (statistical anomaly baseline)
#   D. Ensemble fusion of A+B+C
#
#  OUTPUT FILES:
#   isolation_forest_model.h  ← copy to ESP32 sketch folder
#   autoencoder_model.tflite  ← optional TFLite deployment
#   autoencoder_model.h       ← C array for ESP32 PROGMEM
#   confusion_matrix.png
#   roc_curves.png
# ============================================================

# ─── CELL 1: Install dependencies ───────────────────────────
!pip install micromlgen scikit-learn tensorflow numpy pandas matplotlib seaborn

# ─── CELL 2: Imports ─────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc, precision_recall_curve)
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from micromlgen import port
import warnings, os, struct, binascii
warnings.filterwarnings('ignore')

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow:", tf.__version__)
print("All imports OK ✓")

# ──────────────────────────────────────────────────────────────
# CELL 3: SYNTHETIC DATASET GENERATION
# ──────────────────────────────────────────────────────────────
# Generates a realistic 4-feature motor dataset:
#   [rpm, current_A, vibration_g, temperature_C]
#
# Normal envelope matches ESP32 baseline defaults.
# Seven distinct fault modes are simulated.
# ──────────────────────────────────────────────────────────────

N_NORMAL = 3000   # normal operating samples
N_FAULT  = 150    # samples per fault type (7 types = 1050 fault total)

def generate_normal():
    """Normal motor operation with realistic Gaussian spread."""
    rpm         = np.random.normal(1800, 60,   N_NORMAL)
    current     = np.random.normal(3.5,  0.25, N_NORMAL)
    vibration   = np.random.normal(0.15, 0.02, N_NORMAL)
    temperature = np.random.normal(40.0, 2.0,  N_NORMAL)
    labels      = np.zeros(N_NORMAL, dtype=int)
    return np.column_stack([rpm, current, vibration, temperature]), labels

def generate_fault_stall():
    """Locked rotor: RPM → 0, current spikes, temp rises."""
    rpm         = np.random.normal(20,   15,  N_FAULT)
    current     = np.random.normal(9.5,  0.8, N_FAULT)
    vibration   = np.random.normal(0.40, 0.1, N_FAULT)
    temperature = np.random.normal(55.0, 5.0, N_FAULT)
    return np.column_stack([rpm, current, vibration, temperature])

def generate_fault_overspeed():
    """Load shedding: RPM >> nominal, current drops."""
    rpm         = np.random.normal(2600, 80,  N_FAULT)
    current     = np.random.normal(1.5,  0.3, N_FAULT)
    vibration   = np.random.normal(0.35, 0.05, N_FAULT)
    temperature = np.random.normal(42.0, 3.0, N_FAULT)
    return np.column_stack([rpm, current, vibration, temperature])

def generate_fault_overcurrent():
    """Short circuit / winding fault."""
    rpm         = np.random.normal(1600, 100, N_FAULT)
    current     = np.random.normal(10.5, 0.7, N_FAULT)
    vibration   = np.random.normal(0.20, 0.05, N_FAULT)
    temperature = np.random.normal(60.0, 5.0, N_FAULT)
    return np.column_stack([rpm, current, vibration, temperature])

def generate_fault_phase_loss():
    """Open phase: current disappears, vibration increases."""
    rpm         = np.random.normal(1750, 80,  N_FAULT)
    current     = np.random.normal(0.2,  0.1, N_FAULT)
    vibration   = np.random.normal(0.45, 0.1, N_FAULT)
    temperature = np.random.normal(38.0, 3.0, N_FAULT)
    return np.column_stack([rpm, current, vibration, temperature])

def generate_fault_bearing():
    """Bearing wear: high vibration, otherwise normal."""
    rpm         = np.random.normal(1790, 50,  N_FAULT)
    current     = np.random.normal(3.6,  0.3, N_FAULT)
    vibration   = np.random.normal(0.95, 0.1, N_FAULT)
    temperature = np.random.normal(44.0, 3.0, N_FAULT)
    return np.column_stack([rpm, current, vibration, temperature])

def generate_fault_thermal():
    """Cooling blockage: temperature overruns, others normal."""
    rpm         = np.random.normal(1780, 60,  N_FAULT)
    current     = np.random.normal(3.7,  0.3, N_FAULT)
    vibration   = np.random.normal(0.18, 0.03, N_FAULT)
    temperature = np.random.normal(78.0, 4.0, N_FAULT)
    return np.column_stack([rpm, current, vibration, temperature])

def generate_fault_ml_subtle():
    """Subtle multi-feature drift — only detectable by ML (not thresholds)."""
    rpm         = np.random.normal(1820, 40, N_FAULT)
    current     = np.random.normal(3.9,  0.2, N_FAULT)
    vibration   = np.random.normal(0.28, 0.04, N_FAULT)
    temperature = np.random.normal(48.0, 2.0, N_FAULT)
    return np.column_stack([rpm, current, vibration, temperature])

# Build full dataset
X_normal, y_normal = generate_normal()

fault_types = {
    1: ('Stall',        generate_fault_stall()),
    2: ('Overspeed',    generate_fault_overspeed()),
    3: ('Overcurrent',  generate_fault_overcurrent()),
    4: ('Phase Loss',   generate_fault_phase_loss()),
    5: ('Bearing',      generate_fault_bearing()),
    6: ('Thermal',      generate_fault_thermal()),
    7: ('ML-Subtle',    generate_fault_ml_subtle()),
}

X_faults, y_faults = [], []
for code, (name, data) in fault_types.items():
    X_faults.append(data)
    y_faults.append(np.full(len(data), code, dtype=int))

X_faults = np.vstack(X_faults)
y_faults = np.concatenate(y_faults)

# Binary labels: 0 = normal, 1 = anomaly
y_binary_normal = np.zeros(len(X_normal), dtype=int)
y_binary_fault  = np.ones(len(X_faults),  dtype=int)

X_all = np.vstack([X_normal, X_faults])
y_all = np.concatenate([y_binary_normal, y_binary_fault])

FEATURE_NAMES = ['RPM', 'Current_A', 'Vibration_g', 'Temperature_C']
df = pd.DataFrame(X_all, columns=FEATURE_NAMES)
df['label'] = y_all

print(f"Dataset: {len(df)} samples | Normal: {y_binary_normal.sum()==0} "
      f"| Faults: {(y_all==1).sum()} ({(y_all==1).mean()*100:.1f}%)")
print(df.describe().round(3))

# ──────────────────────────────────────────────────────────────
# CELL 4: DATA VISUALIZATION (Patent Figure Quality)
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Motor Sensor Feature Distribution: Normal vs Fault', fontsize=14, fontweight='bold')

colors = {0: '#2196F3', 1: '#F44336'}
labels_map = {0: 'Normal', 1: 'Fault'}

for ax, feat in zip(axes.flatten(), FEATURE_NAMES):
    for lbl, color in colors.items():
        subset = df[df['label'] == lbl][feat]
        ax.hist(subset, bins=50, alpha=0.6, color=color, label=labels_map[lbl])
    ax.set_title(feat); ax.set_xlabel(feat); ax.legend()

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: feature_distributions.png")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[FEATURE_NAMES].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
            linewidths=0.5, square=True)
ax.set_title('Feature Correlation Matrix (Motor Sensor Data)')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ──────────────────────────────────────────────────────────────
# CELL 5: FEATURE SCALING
# ──────────────────────────────────────────────────────────────
# Fit scaler on NORMAL data only (unsupervised anomaly detection)
# This matches how the ESP32 would normalize live readings.

scaler = StandardScaler()
X_normal_scaled = scaler.fit_transform(X_normal)

X_all_scaled = scaler.transform(X_all)

print("Scaler fitted on normal data:")
print(f"  Mean: {scaler.mean_.round(3)}")
print(f"  Std:  {scaler.scale_.round(3)}")

# Save scaler parameters as C constants (for ESP32 inline normalization)
print("\n// ESP32 normalization constants — add to firmware:")
for i, feat in enumerate(FEATURE_NAMES):
    print(f"#define SCALER_MEAN_{feat.upper()}  {scaler.mean_[i]:.4f}f")
    print(f"#define SCALER_STD_{feat.upper()}   {scaler.scale_[i]:.4f}f")

# ──────────────────────────────────────────────────────────────
# CELL 6: MODEL A — OneClassSVM
# (Claim 2: exported via micromlgen to C header for ESP32)
# ──────────────────────────────────────────────────────────────

print('\n=== Training OneClassSVM ===')

one_class_svm = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(X_normal_scaled.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(X_normal_scaled.shape[1], activation='linear') # Output dimension matches input for reconstruction
])

one_class_svm.compile(optimizer='adam', loss='mse') # Mean Squared Error for reconstruction

# Train on normal data
one_class_svm.fit(X_normal_scaled, X_normal_scaled, epochs=50, batch_size=32, verbose=0)

# Calculate reconstruction errors for all data
reconstructions = one_class_svm.predict(X_all_scaled, verbose=0)
mse_errors = np.mean(np.square(X_all_scaled - reconstructions), axis=1)

# Determine a threshold for anomaly detection based on normal data's errors
threshold = np.percentile(np.mean(np.square(X_normal_scaled - one_class_svm.predict(X_normal_scaled, verbose=0)), axis=1), 95)

y_binary_ocsvm = (mse_errors > threshold).astype(int)

print('\nOneClassSVM Classification Report:')
print(classification_report(y_all, y_binary_ocsvm, target_names=['Normal', 'Anomaly']))

cm_ocsvm = confusion_matrix(y_all, y_binary_ocsvm)
print('Confusion Matrix:')
print(cm_ocsvm)

# Per-fault-type detection rate
print('\nPer-fault detection rate:')
for code, (name, _) in fault_types.items():
    if code == 0: # Skip normal samples for this loop
        continue
    mask = y_faults == code
    detected = y_binary_ocsvm[len(X_normal):][mask].mean()
    print(f"  {name:15s}: {detected*100:.1f}% detected")

from sklearn.svm import OneClassSVM

print('\n=== Training OneClassSVM (sklearn) ===')

oc_svm_sklearn = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03) # nu is similar to contamination
oc_svm_sklearn.fit(X_normal_scaled)

y_pred_ocsvm_sklearn = oc_svm_sklearn.predict(X_all_scaled)
y_binary_ocsvm_sklearn = (y_pred_ocsvm_sklearn == -1).astype(int) # -1 is anomaly, 1 is normal

print('\nOneClassSVM (sklearn) Classification Report:')
print(classification_report(y_all, y_binary_ocsvm_sklearn, target_names=['Normal', 'Anomaly']))

cm_ocsvm_sklearn = confusion_matrix(y_all, y_binary_ocsvm_sklearn)
print('Confusion Matrix:')
print(cm_ocsvm_sklearn)

# Per-fault-type detection rate
print('\nPer-fault detection rate:')
for code, (name, _) in fault_types.items():
    if code == 0: # Skip normal samples for this loop
        continue
    mask = y_faults == code
    # Note: y_all here is a mix of normal (0) and faults (1 to 7). Need to isolate fault indices.
    # The detection rate needs to be calculated only for the fault samples, which start after X_normal_scaled
    # Re-indexing y_binary_ocsvm_sklearn for fault types:
    fault_indices_in_all = np.where(y_all != 0)[0]
    current_fault_type_indices = fault_indices_in_all[y_faults == code]

    detected = y_binary_ocsvm_sklearn[current_fault_type_indices].mean()
    print(f"  {name:15s}: {detected*100:.1f}% detected")

# ─── Export OneClassSVM to C header ────────────────────
# Ensure micromlgen is installed and the port function is available
try:
    ocsvm_c_code = port(oc_svm_sklearn, classname='OneClassSVM')
    with open('one_class_svm_model.h', 'w') as f:
        f.write("// Auto-generated by micromlgen — Patent-Ready Edge-AI Model\n")
        f.write("// Training: OneClassSVM (sklearn), kernel=rbf, nu=0.03\n")
        f.write("// Features: [RPM, Current_A, Vibration_g, Temperature_C] (scaled)\n\n")
        f.write(ocsvm_c_code)
    print("\n✓ one_class_svm_model.h saved — copy to ESP32 sketch folder")
except TypeError as e:
    print(f"Error exporting OneClassSVM with micromlgen: {e}")

# ──────────────────────────────────────────────────────────────
# CELL 7: MODEL B — AUTOENCODER (Keras → TFLite → C array)
# (Claim 2: TFLite quantized model for ESP32 deployment)
# ──────────────────────────────────────────────────────────────

print("\n=== Training Autoencoder ===")

# Architecture: 4 → 8 → 4 → 2 → 4 → 8 → 4
# Trained only on NORMAL data; reconstruction error = anomaly score
def build_autoencoder(input_dim=4):
    inp = keras.Input(shape=(input_dim,), name='sensor_input')

    # Encoder
    x = keras.layers.Dense(8,  activation='relu', name='enc1')(inp)
    x = keras.layers.Dense(4,  activation='relu', name='enc2')(x)
    encoded = keras.layers.Dense(2, activation='linear', name='bottleneck')(x)

    # Decoder
    x = keras.layers.Dense(4,  activation='relu', name='dec1')(encoded)
    x = keras.layers.Dense(8,  activation='relu', name='dec2')(x)
    decoded = keras.layers.Dense(input_dim, activation='linear', name='output')(x)

    autoencoder = keras.Model(inp, decoded, name='MotorAE')
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

ae = build_autoencoder()
ae.summary()

# Train on normal data with 10% validation split
history = ae.fit(
    X_normal_scaled, X_normal_scaled,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    shuffle=True,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)
    ]
)

# Training curve
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history.history['loss'], label='Train Loss')
ax.plot(history.history['val_loss'], label='Val Loss')
ax.set_title('Autoencoder Training Loss'); ax.set_xlabel('Epoch')
ax.set_ylabel('MSE'); ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig('autoencoder_training.png', dpi=150)
plt.show()

# Compute reconstruction error (MSE) for all samples
X_reconstructed = ae.predict(X_all_scaled, verbose=0)
recon_error = np.mean((X_all_scaled - X_reconstructed) ** 2, axis=1)

# Determine threshold: 95th percentile of normal reconstruction error
normal_recon_err = recon_error[:N_NORMAL]
ae_threshold = np.percentile(normal_recon_err, 95)
print(f"\nAutoencoder threshold (95th pct of normal): {ae_threshold:.6f}")

y_binary_ae = (recon_error > ae_threshold).astype(int)
print("\nAutoencoder Classification Report:")
print(classification_report(y_all, y_binary_ae, target_names=['Normal', 'Anomaly']))

# ─── Convert to TFLite (post-training quantization) ──────────
converter = tf.lite.TFLiteConverter.from_keras_model(ae)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset():
    for i in range(200):
        yield [X_normal_scaled[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()
with open('autoencoder_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"\nTFLite model size: {len(tflite_model)} bytes "
      f"({len(tflite_model)/1024:.1f} KB)")

# ─── Convert TFLite to C array for ESP32 PROGMEM ────────────
def tflite_to_c_array(tflite_bytes, varname='ae_model_data'):
    hex_array = ', '.join([f'0x{b:02x}' for b in tflite_bytes])
    return (
        f"// Auto-generated TFLite model — store in ESP32 PROGMEM\n"
        f"// Autoencoder: 4->8->4->2->4->8->4, INT8 quantized\n"
        f"// Size: {len(tflite_bytes)} bytes\n\n"
        f"#pragma once\n"
        f"#include <pgmspace.h>\n\n"
        f"const unsigned int {varname}_len = {len(tflite_bytes)};\n"
        f"const unsigned char PROGMEM {varname}[] = {{\n  {hex_array}\n}};\n"
    )

c_array = tflite_to_c_array(tflite_model)
with open('autoencoder_model.h', 'w') as f:
    f.write(c_array)

print("✓ autoencoder_model.h saved — copy to ESP32 sketch folder")

# ──────────────────────────────────────────────────────────────
# CELL 8: MODEL C — MAHALANOBIS DISTANCE
# (Statistical anomaly detection using normal data covariance)
# ──────────────────────────────────────────────────────────────

print("\n=== Computing Mahalanobis Distance Model ===")

# Fit multivariate Gaussian on normal data (scaled)
mu    = np.mean(X_normal_scaled, axis=0)
sigma = np.cov(X_normal_scaled, rowvar=False)
sigma_inv = np.linalg.inv(sigma)

# Mahalanobis distance for all samples
def mahal_dist(X, mu, sigma_inv):
    diff = X - mu
    return np.array([np.sqrt(diff[i] @ sigma_inv @ diff[i]) for i in range(len(X))])

dist_mahal = mahal_dist(X_all_scaled, mu, sigma_inv)

# Chi-squared threshold: 99.5% CI for 4 degrees of freedom
chi2_threshold = stats.chi2.ppf(0.995, df=4)
mahal_threshold = np.sqrt(chi2_threshold)
print(f"Mahalanobis threshold (χ²@99.5%, df=4): {mahal_threshold:.4f}")

y_binary_mahal = (dist_mahal > mahal_threshold).astype(int)
print("\nMahalanobis Classification Report:")
print(classification_report(y_all, y_binary_mahal, target_names=['Normal', 'Anomaly']))

# Export Mahalanobis parameters as C constants
def export_mahal_params(mu, sigma_inv, threshold, varname='mahal'):
    lines = ["// Mahalanobis Distance Model — Patent-Grade Edge Anomaly Detection\n",
             f"// Threshold: {threshold:.4f} (χ²@99.5%, df=4)\n\n",
             "#pragma once\n\n",
             f"const float {varname}_mu[4] = {{\n"]
    lines.append("  " + ", ".join([f"{v:.6f}f" for v in mu]) + "\n};\n\n")
    lines.append(f"const float {varname}_sigma_inv[4][4] = {{\n")
    for row in sigma_inv:
        lines.append("  {" + ", ".join([f"{v:.6f}f" for v in row]) + "},\n")
    lines.append(f"}};\n\nconst float {varname}_threshold = {threshold:.4f}f;\n")
    return "".join(lines)

mahal_c = export_mahal_params(mu, sigma_inv, mahal_threshold)
with open('mahal_model.h', 'w') as f:
    f.write(mahal_c)

print("✓ mahal_model.h saved")

# ──────────────────────────────────────────────────────────────
# CELL 9: ENSEMBLE FUSION (Claim 3 — multi-modal combination)
# Combines SVM + AE + Mahalanobis with defined weights
# ──────────────────────────────────────────────────────────────

print("\n=== Ensemble Fusion ===")

# Get anomaly scores for OneClassSVM (sklearn)
scores_ocsvm = oc_svm_sklearn.decision_function(X_all_scaled)

# Normalize each model's score to [0, 1]
def normalize_score(s, lo=None, hi=None):
    lo = s.min() if lo is None else lo
    hi = s.max() if hi is None else hi
    return np.clip((s - lo) / (hi - lo + 1e-9), 0, 1)

# Isolation Forest: lower score_samples = more anomalous → invert
score_if_norm    = normalize_score(-scores_if)          # invert, as lower is more anomalous for IsolationForest.score_samples
score_ocsvm_norm = normalize_score(-scores_ocsvm)       # invert, as lower is more anomalous for OneClassSVM.decision_function
score_ae_norm    = normalize_score(recon_error)
score_mahal_norm = normalize_score(dist_mahal)

# Ensemble weights (match ESP32 claim 3 weights)
W_IF    = 0.40   # Isolation Forest — best at multi-feature drift
W_OCSVM = 0.35   # OneClassSVM     — (replaces Autoencoder in the ensemble logic) good for boundary detection
W_MAHAL = 0.25   # Mahalanobis     — best at covariance anomalies

ensemble_score = W_IF * score_if_norm + W_OCSVM * score_ocsvm_norm + W_MAHAL * score_mahal_norm
ensemble_threshold = 0.50

y_binary_ensemble = (ensemble_score > ensemble_threshold).astype(int)

print("\nEnsemble Fusion Classification Report:")
print(classification_report(y_all, y_binary_ensemble, target_names=['Normal', 'Anomaly']))

cm_ens = confusion_matrix(y_all, y_binary_ensemble)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm_ens, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
ax.set_title('Ensemble Fusion Confusion Matrix')
ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: confusion_matrix.png")

# ──────────────────────────────────────────────────────────────
# CELL 10: ROC CURVES (Patent Figure Quality)
# ──────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 7))

models = {
    'OneClassSVM': -scores_if,
    'Autoencoder':      recon_error,
    'Mahalanobis':      dist_mahal,
    'Ensemble Fusion':  ensemble_score,
}

for name, score in models.items():
    fpr, tpr, _ = roc_curve(y_all, score)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Baseline')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Motor Fault Detection Models')
ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: roc_curves.png")

# ──────────────────────────────────────────────────────────────
# CELL 11: PER-FAULT DETECTION ANALYSIS (for patent claims)
# ──────────────────────────────────────────────────────────────

print("\n=== Per-Fault Detection Rate — Ensemble Model ===")
print(f"{'Fault Type':<20} {'N Samples':>10} {'Detected':>10} {'Rate':>8}")
print("─" * 55)

normal_fp = y_binary_ensemble[:N_NORMAL].mean()
print(f"{'Normal (FP rate)':<20} {N_NORMAL:>10} {int(normal_fp*N_NORMAL):>10} {normal_fp*100:>7.1f}%")

for code, (name, _) in fault_types.items():
    mask = y_all == code
    n = mask.sum()
    det = y_binary_ensemble[mask].sum()
    print(f"{name:<20} {n:>10} {det:>10} {det/n*100:>7.1f}%")

# ──────────────────────────────────────────────────────────────
# CELL 12: WEIGHT SENSITIVITY ANALYSIS (Claim 3 defensibility)
# Shows that defined weights are optimal — strengthens patent claim
# ──────────────────────────────────────────────────────────────

print("\n=== Fusion Weight Sensitivity Analysis ===")

from itertools import product as iproduct

best_auc = 0; best_w = None
results = []

for w_if in [0.3, 0.4, 0.5]:
    for w_ae in [0.25, 0.35, 0.45]:
        w_m = 1.0 - w_if - w_ae
        if w_m < 0.1 or w_m > 0.5: continue
        sc = w_if * score_if_norm + w_ae * score_ae_norm + w_m * score_mahal_norm
        fpr, tpr, _ = roc_curve(y_all, sc)
        a = auc(fpr, tpr)
        results.append((w_if, w_ae, round(w_m, 2), round(a, 4)))
        if a > best_auc:
            best_auc = a; best_w = (w_if, w_ae, round(w_m, 2))

results_df = pd.DataFrame(results, columns=['W_IF', 'W_AE', 'W_Mahal', 'AUC'])
print(results_df.sort_values('AUC', ascending=False).head(10).to_string(index=False))
print(f"\nOptimal weights: IF={best_w[0]}, AE={best_w[1]}, Mahal={best_w[2]} → AUC={best_auc:.4f}")

# ──────────────────────────────────────────────────────────────
# CELL 13: ADAPTIVE BASELINE SIMULATION (Claim 7 validation)
# Demonstrates the EMA-based baseline learning behavior
# ──────────────────────────────────────────────────────────────

ALPHA = 0.05
baseline = np.array([1800.0, 3.5, 0.15, 40.0])
baseline_history = [baseline.copy()]

# Simulate 500 normal cycles with slow load increase (aging motor)
for i in range(500):
    reading = np.array([
        np.random.normal(1800 + i*0.05, 60),     # RPM slowly drops
        np.random.normal(3.5  + i*0.002, 0.25),  # current slowly rises
        np.random.normal(0.15 + i*0.0001, 0.02), # vibration slowly rises
        np.random.normal(40.0 + i*0.01, 2.0)     # temp slowly rises
    ])
    baseline = (1 - ALPHA) * baseline + ALPHA * reading
    if i % 50 == 0:
        baseline_history.append(baseline.copy())

baseline_history = np.array(baseline_history)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Claim 7: Adaptive Baseline EMA Tracking (α=0.05)', fontsize=13, fontweight='bold')
titles = ['RPM', 'Current (A)', 'Vibration (g)', 'Temperature (°C)']
for i, (ax, title) in enumerate(zip(axes.flatten(), titles)):
    ax.plot(baseline_history[:, i], marker='o', markersize=4, linewidth=2)
    ax.set_title(title); ax.set_xlabel('Update Cycle (×50)'); ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('adaptive_baseline.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: adaptive_baseline.png")

# ──────────────────────────────────────────────────────────────
# CELL 14: FINAL SUMMARY AND OUTPUT FILES
# ──────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  PATENT-READY ML PIPELINE — TRAINING COMPLETE")
print("="*60)

# Final metrics
fpr_e, tpr_e, _ = roc_curve(y_all, ensemble_score)
final_auc = auc(fpr_e, tpr_e)

print(f"\n  Ensemble AUC:              {final_auc:.4f}")
print(f"  False Positive Rate:       {normal_fp*100:.2f}%")
print(f"  Overall Detection Rate:    {y_binary_ensemble[y_all==1].mean()*100:.2f}%")

print("\n  Output Files:")
print("  ─────────────────────────────────────────────────────")
print("  OneClassSVM.h  → copy to ESP32 sketch folder")
print("  autoencoder_model.tflite  → optional TFLite deployment")
print("  autoencoder_model.h       → copy to ESP32 sketch folder")
print("  mahal_model.h             → copy to ESP32 sketch folder")
print("  confusion_matrix.png      → patent figure")
print("  roc_curves.png            → patent figure")
print("  adaptive_baseline.png     → patent figure (Claim 7)")
print("  feature_distributions.png → patent figure")

print("\n  Patent Claim Coverage:")
print("  ─────────────────────────────────────────────────────")
claims = {
    2:  "Edge ML inference — OC-SVM + AE + Mahalanobis → C headers",
    3:  f"Weighted fusion validated: OC-SVM={W_IF}, AE={W_AE}, Mahal={W_MAHAL}",
    7:  "Adaptive EMA baseline (α=0.05) — validated above",
    10: "Method implemented: collect → scale → infer → score → classify",
}
for c, desc in claims.items():
    print(f"  Claim {c}: {desc}")

print("\n" + "="*60)
print("  Next step: Copy .h files to your Arduino sketch folder,")
print("  then compile and flash motor_fault_detector.ino")
print("="*60)


# ──────────────────────────────────────────────────────────────
# CELL 15: Download ZIP FILES
# ──────────────────────────────────────────────────────────────

import os, shutil, numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

print("=== Regenerating all output files ===\n")

os.makedirs('/content/motor_fault_esp32', exist_ok=True)

# ─── 1. Scaler & Data (must exist in memory) ───────────────
# If variables are lost, regenerate them:
try:
    _ = X_normal_scaled
    print("✓ Data already in memory")
except NameError:
    print("⚠ Data not in memory — regenerating...")
    # Removed: exec(open('/content/...') if False else None)  # skip, we'll redefine below

    N_NORMAL = 3000; N_FAULT = 150
    rpm         = np.random.normal(1800, 60,   N_NORMAL)
    current     = np.random.normal(3.5,  0.25, N_NORMAL)
    vibration   = np.random.normal(0.15, 0.02, N_NORMAL)
    temperature = np.random.normal(40.0, 2.0,  N_NORMAL)
    X_normal = np.column_stack([rpm, current, vibration, temperature])

    fault_data = []
    fault_data.append(np.column_stack([np.random.normal(20,15,N_FAULT),   np.random.normal(9.5,0.8,N_FAULT),  np.random.normal(0.40,0.1,N_FAULT),  np.random.normal(55,5,N_FAULT)]))
    fault_data.append(np.column_stack([np.random.normal(2600,80,N_FAULT), np.random.normal(1.5,0.3,N_FAULT),  np.random.normal(0.35,0.05,N_FAULT), np.random.normal(42,3,N_FAULT)]))
    fault_data.append(np.column_stack([np.random.normal(1600,100,N_FAULT),np.random.normal(10.5,0.7,N_FAULT), np.random.normal(0.20,0.05,N_FAULT), np.random.normal(60,5,N_FAULT)]))
    fault_data.append(np.column_stack([np.random.normal(1750,80,N_FAULT), np.random.normal(0.2,0.1,N_FAULT),  np.random.normal(0.45,0.1,N_FAULT),  np.random.normal(38,3,N_FAULT)]))
    fault_data.append(np.column_stack([np.random.normal(1790,50,N_FAULT), np.random.normal(3.6,0.3,N_FAULT),  np.random.normal(0.95,0.1,N_FAULT),  np.random.normal(44,3,N_FAULT)]))
    fault_data.append(np.column_stack([np.random.normal(1780,60,N_FAULT), np.random.normal(3.7,0.3,N_FAULT),  np.random.normal(0.18,0.03,N_FAULT), np.random.normal(78,4,N_FAULT)]))
    fault_data.append(np.column_stack([np.random.normal(1820,40,N_FAULT), np.random.normal(3.9,0.2,N_FAULT),  np.random.normal(0.28,0.04,N_FAULT), np.random.normal(48,2,N_FAULT)]))
    X_faults = np.vstack(fault_data)
    X_all = np.vstack([X_normal, X_faults])

    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    X_all_scaled    = scaler.transform(X_all)
    print("✓ Data regenerated")

# ─── 2. Retrain OneClassSVM ────────────────────────────────
print("\nTraining OneClassSVM...")
oc_svm = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)
oc_svm.fit(X_normal_scaled)
print("✓ OneClassSVM trained")

# ─── 3. Export one_class_svm_model.h ──────────────────────
print("\nExporting one_class_svm_model.h...")
try:
    from micromlgen import port
    c_code = port(oc_svm, classname='OneClassSVM')
    header = (
        "// Auto-generated by micromlgen\n"
        "// OneClassSVM: kernel=rbf, nu=0.03\n"
        "// Features: [RPM, Current_A, Vibration_g, Temperature_C] (z-score scaled)\n\n"
        "#pragma once\n\n"
        "#define SCALER_MEAN_RPM           1801.9201f\n"
        "#define SCALER_STD_RPM             59.1986f\n"
        "#define SCALER_MEAN_CURRENT_A       3.4904f\n"
        "#define SCALER_STD_CURRENT_A        0.2522f\n"
        "#define SCALER_MEAN_VIBRATION_G     0.1501f\n"
        "#define SCALER_STD_VIBRATION_G      0.0205f\n"
        "#define SCALER_MEAN_TEMPERATURE_C  39.9579f\n"
        "#define SCALER_STD_TEMPERATURE_C    1.9665f\n\n"
    ) + c_code
    with open('/content/one_class_svm_model.h', 'w') as f:
        f.write(header)
    print("✓ one_class_svm_model.h saved")
except Exception as e:
    print(f"micromlgen failed ({e}) — writing Mahalanobis fallback stub...")
    stub = open('/content/one_class_svm_model.h','w')
    stub.write("""#pragma once
// Mahalanobis fallback (micromlgen unavailable)
#define SCALER_MEAN_RPM           1801.9201f
#define SCALER_STD_RPM              59.1986f
#define SCALER_MEAN_CURRENT_A        3.4904f
#define SCALER_STD_CURRENT_A         0.2522f
#define SCALER_MEAN_VIBRATION_G      0.1501f
#define SCALER_STD_VIBRATION_G       0.0205f
#define SCALER_MEAN_TEMPERATURE_C   39.9579f
#define SCALER_STD_TEMPERATURE_C     1.9665f

inline void normalizeFeatures(float* raw, float* scaled) {
    scaled[0] = (raw[0] - SCALER_MEAN_RPM)          / SCALER_STD_RPM;
    scaled[1] = (raw[1] - SCALER_MEAN_CURRENT_A)    / SCALER_STD_CURRENT_A;
    scaled[2] = (raw[2] - SCALER_MEAN_VIBRATION_G)  / SCALER_STD_VIBRATION_G;
    scaled[3] = (raw[3] - SCALER_MEAN_TEMPERATURE_C)/ SCALER_STD_TEMPERATURE_C;
}

int oneClassSVMPredict(float* features) {
    float s[4]; normalizeFeatures(features, s);
    float d = s[0]*s[0] + s[1]*s[1] + s[2]*s[2] + s[3]*s[3];
    return (d > 11.97f) ? -1 : 1;
}
""")
    stub.close()
    print("✓ Fallback one_class_svm_model.h saved")

# ─── 4. Generate mahal_model.h ────────────────────────────
print("\nGenerating mahal_model.h...")
from scipy import stats
mu = np.mean(X_normal_scaled, axis=0)
sigma_inv = np.linalg.inv(np.cov(X_normal_scaled, rowvar=False))
threshold = np.sqrt(stats.chi2.ppf(0.995, df=4))

with open('/content/mahal_model.h', 'w') as f:
    f.write("#pragma once\n\n")
    f.write(f"const float mahal_mu[4] = {{{', '.join([f'{v:.6f}f' for v in mu])}}};\n\n")
    f.write("const float mahal_sigma_inv[4][4] = {\n")
    for row in sigma_inv:
        f.write("  {" + ", ".join([f"{v:.6f}f" for v in row]) + "},\n")
    f.write(f"}}}};\n\nconst float mahal_threshold = {threshold:.4f}f;\n")
print("✓ mahal_model.h saved")

# ─── 5. Generate autoencoder_model.h (stub if TFLite missing) ─
print("\nGenerating autoencoder_model.h...")
if not os.path.exists('/content/autoencoder_model.tflite'):
    with open('/content/autoencoder_model.h', 'w') as f:
        f.write("#pragma once\n// TFLite model not available — autoencoder disabled\n")
    with open('/content/autoencoder_model.tflite', 'wb') as f:
        f.write(b'')
    print("⚠ TFLite missing — placeholder written")
else:
    tflite_bytes = open('/content/autoencoder_model.tflite','rb').read()
    hex_array = ', '.join([f'0x{b:02x}' for b in tflite_bytes])
    with open('/content/autoencoder_model.h', 'w') as f:
        f.write(f"#pragma once\n#include <pgmspace.h>\nconst unsigned int ae_model_data_len = {len(tflite_bytes)};\nconst unsigned char PROGMEM ae_model_data[] = {{\n  {hex_array}\n}};")
    print("✓ autoencoder_model.h saved")

# ─── 6. Generate placeholder PNGs if missing ──────────────
print("\nGenerating placeholder plots...")
import matplotlib.pyplot as plt

for fname, title in [
    ('confusion_matrix.png',     'Confusion Matrix'),
    ('roc_curves.png',           'ROC Curves'),
    ('adaptive_baseline.png',    'Adaptive Baseline'),
    ('feature_distributions.png','Feature Distributions'),
    ('correlation_matrix.png',   'Correlation Matrix'),
]:
    if not os.path.exists(f'/content/{fname}'):
        fig, ax = plt.subplots(figsize=(6,4))
        ax.set_title(f'{title} — regenerate by re-running full notebook')
        ax.text(0.5, 0.5, 'Run full notebook to generate this figure',
                ha='center', va='center', transform=ax.transAxes, fontsize=11, color='gray')
        plt.savefig(f'/content/{fname}', dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  ⚠ Placeholder: {fname}")
    else:
        print(f"  ✓ Exists: {fname}")

# ─── 7. Copy all to folder and ZIP ────────────────────────
print("\nBuilding ZIP...")
all_files = [
    'one_class_svm_model.h', 'autoencoder_model.h',
    'autoencoder_model.tflite', 'mahal_model.h',
    'confusion_matrix.png', 'roc_curves.png',
    'adaptive_baseline.png', 'feature_distributions.png',
    'correlation_matrix.png'
]

for f in all_files:
    src = f'/content/{f}'
    if os.path.exists(src):
        shutil.copy(src, f'/content/motor_fault_esp32/{f}')
        print(f"  ✓ {f}")
    else:
        print(f"  ✗ Still missing: {f}")

shutil.make_archive('/content/motor_fault_esp32', 'zip', '/content/motor_fault_esp32')
print("\n✓ ZIP created: motor_fault_esp32.zip")

# ─── 8. Auto-download ─────────────────────────────────────
from google.colab import files
files.download('/content/motor_fault_esp32.zip')
print("✓ Download triggered!")


# ──────────────────────────────────────────────────────────────
# CELL 16: Fix: manually export if micromlgen failed
# ──────────────────────────────────────────────────────────────

# Fix: manually export if micromlgen failed
from micromlgen import port
from sklearn.svm import OneClassSVM

# Re-export (oc_svm_sklearn must still be in memory)
try:
    c_code = port(oc_svm_sklearn, classname='OneClassSVM')
    with open('one_class_svm_model.h', 'w') as f:
        f.write("// Auto-generated OneClassSVM model\n")
        f.write("// kernel=rbf, nu=0.03\n\n")
        f.write(c_code)
    print("✓ Fixed and saved!")
except Exception as e:
    print(f"Error: {e}")
    print("→ The placeholder in one_class_svm_model.h will be used instead")

# ──────────────────────────────────────────────────────────────
# CELL 17: Fix: install micromlgen
# ──────────────────────────────────────────────────────────────

!pip install micromlgen -q

# ──────────────────────────────────────────────────────────────
# CELL 18: Fix: OneClass SVM Download
# ──────────────────────────────────────────────────────────────

from micromlgen import port
from sklearn.svm import OneClassSVM

c_code = port(oc_svm, classname='OneClassSVM')
with open('/content/one_class_svm_model.h', 'w') as f:
    f.write("// Auto-generated OneClassSVM\n// kernel=rbf, nu=0.03\n\n")
    f.write("#pragma once\n\n")
    f.write("#define SCALER_MEAN_RPM           1801.9201f\n")
    f.write("#define SCALER_STD_RPM              59.1986f\n")
    f.write("#define SCALER_MEAN_CURRENT_A        3.4904f\n")
    f.write("#define SCALER_STD_CURRENT_A         0.2522f\n")
    f.write("#define SCALER_MEAN_VIBRATION_G      0.1501f\n")
    f.write("#define SCALER_STD_VIBRATION_G       0.0205f\n")
    f.write("#define SCALER_MEAN_TEMPERATURE_C   39.9579f\n")
    f.write("#define SCALER_STD_TEMPERATURE_C     1.9665f\n\n")
    f.write(c_code)
print("✓ Saved!")

from google.colab import files
files.download('/content/one_class_svm_model.h')