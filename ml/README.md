# Machine Learning Folder

This folder contains the complete ML training pipeline, exported models, and visualizations.

---

# Files

| File | Description |
|---|---|
| motor_fault_ML_colab (1).py | Complete ML training pipeline |
| autoencoder.png | Autoencoder architecture |
| ocsvm.png | OneClassSVM results |
| feature_distributions.png | Sensor feature distributions |

---

# ML Pipeline

The pipeline performs:

1. Synthetic dataset generation
2. Feature scaling
3. OneClassSVM training
4. Autoencoder training
5. Mahalanobis analysis
6. Ensemble fusion scoring
7. ESP32 model export

---

# Models Used

| Model | Purpose |
|---|---|
| OneClassSVM | Boundary anomaly detection |
| Autoencoder | Reconstruction anomaly detection |
| Mahalanobis Distance | Statistical anomaly detection |

---

# Dataset Features

- RPM
- Current_A
- Vibration_g
- Temperature_C

---

# Exported Outputs

The training pipeline exports:

- ESP32 header files
- TFLite models
- ROC curves
- Confusion matrices
- Correlation matrices
- EMA baseline plots

---

# Performance

| Metric | Value |
|---|---|
| ROC AUC | 1.000 |
| False Positive Rate | 0.00% |
| OneClassSVM Accuracy | 98% |
