# Wiring Connections

## ESP32 Connections

| Sensor | GPIO |
|---|---|
| IR Sensor | GPIO 34 |
| ACS712 | GPIO 35 |
| MPU6050 SDA | GPIO 21 |
| MPU6050 SCL | GPIO 22 |
| DS18B20 | GPIO 4 |

---

# Notes

- MPU6050 requires 4.7kΩ pull-ups
- DS18B20 uses OneWire protocol
- ACS712 requires offset calibration
