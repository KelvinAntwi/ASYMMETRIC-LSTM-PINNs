# Asymmetric Dual-LSTM with Physics-Informed Loss for Hybrid Solar PV Forecasting

Daily energy-generation forecasting for two co-located solar plants at the Bui Power Authority site: a **50 MW land-based PV plant** and a **5 MW floating PV (FPV) plant**. Each plant gets its own LSTM network, trained with a physics-informed loss that nudges predictions toward a temperature-corrected irradiance-to-energy relationship.

## Overview

| | Land-based PV | Floating PV (FPV) |
|---|---|---|
| Rated capacity | 50 MW | 5 MW |
| Target | `LANDBASED_ENERGY_MWH` | `FLOATING_ENERGY_MWH` |
| Usable daily records | 409 | 576 |
| Architecture | Conv1D → LSTM → Dense → Dropout(0.1) | Conv1D → LSTM → Dropout(0.2) |

The architecture is **asymmetric**: the land model is slightly deeper (an extra dense layer, lighter dropout) because land generation is the more stable system. The floating model is shallower with heavier dropout because its output is noisier and needs more regularisation.

## A Note on Units: Why the Peak Is ~250 MWh, Not 50 MW

The 50 MW figure is the plant's **rated power capacity** (an instantaneous rate). This model forecasts **daily energy generation in MWh** (power accumulated over time). With roughly five effective peak sun hours per day, after accounting for system losses, a 50 MW plant produces about:

> 50 MW × ~5 h ≈ **250 MWh/day**

The ~250 MWh daily peak in the land-based results is therefore consistent with the expected output of Bui's commissioned 50 MW facility.

## Data

- **File:** `merged_hybrid_generation_weather_filled.csv`
- **Generation / site inputs:** global irradiation (Wh/m²), sunshine time (min), plant hours run (hrs)
- **Weather inputs:** max temperature (°C), max wind speed (km/h), max relative humidity (%), shortwave radiation sum (MJ/m²)

### Preprocessing
1. **Active days only:** days with zero generation are removed per plant.
2. **Dynamic efficiency filtering (IQR):** efficiency is computed as `energy / global irradiation`, and days outside the 1.0×IQR fences are dropped as outliers, separately for each plant.
3. **Per-plant datasets:** land and floating data are *not* merged. Each plant keeps all of its own active days instead of only the days both were running.

### Features (11 per plant)
Global irradiation, sunshine time, `MONTH_SIN` / `MONTH_COS` (cyclical month encoding), hours run, `SOLAR_INTENSITY` (irradiation ÷ (sunshine minutes + 1)), `Lag1` (previous record's generation), and the four weather variables.

`Lag1` replaces same-day generation as an input, which removes a target leak.

## Physics-Informed Loss

A physics target is computed for each day from an adapted Sandia-style relation:

```
E_physics = max( η_cal · G · (1 + α · (T − T_STC)), 0 )
```

- `G`: global irradiation, `T`: daily max temperature
- `T_STC = 25 °C`, `α = −0.004 /°C` (crystalline-silicon power temperature coefficient)
- `η_cal`: median `energy / irradiation` ratio, calibrated on **training data only**

The training loss combines a data term with a physics term:

```
L = MSE(y_true, y_pred) + λ · MSE(E_physics, y_pred),   λ = 0.1
```

The physics term acts as a soft prior that pulls predictions toward physically plausible, temperature-aware output.

## Training Setup

- **Split:** chronological 60 / 20 / 20 (train / validation / test), no shuffling
- **Scaling:** MinMax scaling fitted on the training split only
- **Optimiser:** Adam (lr = 0.002), batch size 16, up to 120 epochs
- **Callbacks:** EarlyStopping (patience 15, restores best weights), ReduceLROnPlateau (factor 0.5, patience 6)
- **Seed selection:** each plant model is trained with seeds `[1, 2, 3, 42, 100]`, and the seed with the best **validation** R² is kept

| Plant | Best seed val R² |
|---|---|
| Land | 0.8828 (seed 42) |
| Floating | 0.8606 (seed 3) |

## Results

Metrics below are computed on **MinMax-normalised** targets (0–1), not raw MWh.

| Plant | Split | R² | RMSE | MAE |
|---|---|---|---|---|
| Land | Train | 0.9092 | 0.0617 | 0.0480 |
| Land | Test | 0.7040 | 0.0938 | 0.0821 |
| Floating | Train | 0.9395 | 0.0500 | 0.0385 |
| Floating | Test | 0.8935 | 0.0807 | 0.0592 |

The notebook also produces actual-vs-predicted plots in MWh, residual scatter and histograms, and training/validation loss curves for both plants.

## Limitations

- Each sample is a **single day** (input shape `(1, 11)`), so the LSTM does not see a multi-day history. Temporal context comes only from the `Lag1` feature.
- The land model shows a larger train-to-test gap (R² 0.91 → 0.70) than the floating model, suggesting some overfitting on the smaller land dataset.
- Reported metrics are on normalised values; MWh-scale errors would need to be reported separately.

## Requirements

`python`, `tensorflow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`

## Usage

Place `merged_hybrid_generation_weather_filled.csv` in the working directory and run `ASYMMETRIC_DUAL_LSTM_MODEL_PINNs.ipynb` top to bottom.
