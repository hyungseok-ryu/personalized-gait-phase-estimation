# Raw IMU and FSR Walking Dataset

This repository provides raw thigh-mounted IMU and heel FSR data
collected during walking experiments.

Only the sensor segments used for data analysis are shared. The walking
IMU files contain the filtered values used to construct the model inputs;
windowing, alignment, event labels, and phase targets are not applied.

---

## Dataset Structure

```text
SUB1/
  static/
    imu_static.csv
  normal_trial_1/
    imu_thigh_raw.csv
    fsr_raw.csv
  ...
```

---

## Subject Information

| Subject ID | Affected Side |
|------------|---------------|
| SUB1 | Left |
| SUB2 | Right |
| SUB3 | Left |
| SUB4 | Left |
| SUB5 | Left |

All subjects are anonymized.
No personally identifiable or clinical information is included.

---

## Sensor Description

### IMU (Thigh-mounted)
- 3-axis linear acceleration
- 3-axis angular velocity
- Thigh pitch angle
- Sampling rate: 100 Hz

### FSR (Heel)
- Single-channel force-sensitive resistor
- Sampling rate: 100 Hz

IMU and FSR signals are time-synchronized using timestamps.

---

## Notes

- The walking IMU files contain the retained analysis intervals and the filtered IMU values used to construct the model inputs. They are not windowed or aligned.
- The FSR files contain the corresponding sensor values without phase labels. FSR-derived HS annotations and the preprocessing implementation are distributed with the accompanying training-code release.

---

## License

This dataset is licensed under the Creative Commons Attribution 4.0
International License (CC BY 4.0). See the repository-level `DATA_LICENSE` file
for attribution guidance and the applicable terms.
