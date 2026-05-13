# Firmware Folder

This folder contains the complete ESP32 firmware and exported machine learning models.

---

# Files

| File | Description |
|---|---|
| motor_fault_detector.ino | Main ESP32 firmware |
| config.h | Thresholds and configuration |
| one_class_svm_model.h.h | Exported OneClassSVM header |
| autoencoder_model.h | Autoencoder model header |

---

# Firmware Features

- Real-time sensor acquisition
- Feature normalization
- OneClassSVM inference
- Autoencoder anomaly scoring
- Fusion score computation
- Serial monitoring
- Fault classification

---

# Sensor Inputs

| Sensor | Parameter |
|---|---|
| IR Sensor | RPM |
| ACS712 | Current |
| MPU6050 | Vibration |
| DS18B20 | Temperature |

---

# Runtime Pipeline

```text
Sensor Readings
    ↓
Feature Scaling
    ↓
ML Inference
    ↓
Fusion Scoring
    ↓
Fault Classification
```

---

# Deployment

The firmware runs directly on ESP32 without requiring cloud connectivity.
