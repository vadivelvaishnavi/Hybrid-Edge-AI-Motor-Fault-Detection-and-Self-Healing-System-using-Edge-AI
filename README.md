# Hybrid Edge-AI Motor Fault Detection & Self-Healing System

ESP32-based intelligent motor monitoring system using:

- Edge AI
- OneClassSVM
- Autoencoder
- Mahalanobis Distance
- Multi-sensor fusion
- EEPROM logging
- Self-healing control

---

# Features

Edge inference (<1ms)  
Multi-sensor fusion  
TinyML deployment on ESP32  
Ensemble anomaly detection  
Adaptive EMA baseline  
Patent-ready ML pipeline  

---

# Edge-AI Inference Pipeline

![Architecture](docs/architecture.png)

---

# ML Training Pipeline

![ML Pipeline](docs/block_diagram.png)

---

# Feature Distributions

![Feature Distribution](ml/plots/feature_distributions.png)

---

# Correlation Matrix

![Correlation](ml/plots/correlation_matrix.png)

---

# ROC Curves

![ROC](results/roc_curve.png)

---

# Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

---

# Adaptive EMA Tracking

![EMA](ml/plots/adaptive_baseline.png)

---

# Hardware

- ESP32
- ACS712
- MPU6050
- DS18B20
- IR Sensor
- L298N Driver

---

# ML Models

| Model | Purpose |
|---|---|
| OneClassSVM | Boundary anomaly detection |
| Autoencoder | Reconstruction anomaly detection |
| Mahalanobis | Statistical anomaly detection |
| Ensemble Fusion | Final scoring |

---

# Performance

| Metric | Value |
|---|---|
| Ensemble AUC | 1.000 |
| False Positive Rate | 0.00% |
| OneClassSVM Accuracy | 98% |
| Mahalanobis Accuracy | 99% |

---

# Authors

Kesihambigai S  
VIT University — ECE
