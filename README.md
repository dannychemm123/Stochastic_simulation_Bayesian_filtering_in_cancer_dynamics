# Patient-Specific NSCLC Tumor Dynamics Modeling with Bayesian Filtering

Framework for modeling NSCLC tumor-immune dynamics using mathematical ODEs, Bayesian filtering (EKF/UKF/PF), and CNN prediction.

**Pipeline**: TCGA Data → Data Prep → Parameter Estimation → Bayesian Filtering → CNN → Clinical Simulations

---

## Overview

- **22-parameter ODE** for tumor-immune interactions with PD-1/PD-L1 checkpoint dynamics
- **Bayesian Filtering** (EKF/UKF/Particle Filter) applied to NSCLC data for state estimation
- **Particle Filter outputs** used as training data for CNN deep learning model
- **CNN predictions**: 1-2 ms/patient, MAE 2.6 mm, R² 0.83
- **Clinical simulations** for drug response scenarios (anti-PD-1 therapy)

**Data**: ~1,000 NSCLC patients (LUAD/LUSC) from TCGA LUNG cohort

---

## 📊 Data Preparation (Critical Foundation)

All analyses depend on rigorously prepared NSCLC cohort. **Key steps**:

1. **TCGA Acquisition** (`TCGA_Data_prep.R`): Download ~1,000 NSCLC RNA-seq + clinical via GDC API
2. **RNA-seq Normalization**: TMM normalization (edgeR), extract PD-1/PD-L1 immune markers
3. **Clinical Cleaning** (`UCSC_Xena_Data_prep.R`): Standardize T-stage, map to tumor volumes, remove incomplete samples
4. **Output**: `NSCLC_Clinical_Cleaned.csv` (~800 patients), merged with genomic data

**T-stage mapping** (critical for parameter estimation):
- T1 (~15 mm) → 1.77 billion cells | T2 (~40 mm) → 33.5 billion cells
- T3 (~60 mm) → 113 billion cells | T4 (~80 mm) → 268 billion cells

**Quality checks**: Remove duplicates, validate stage consistency, ensure genomic-clinical alignment

---

## ⚡ Quick Start

```bash
# 1. Parameter estimation from NSCLC T-stage distribution
python least_square_params.py

# 2. Run three Bayesian filters on NSCLC data
python extended_kalman_least_square.py && python ekf_export.py
python unscented_kalman_least_square.py && python ukf_export.py
python particle_filter_least_square2.py && python pf_export.py

# 3. Train CNN on Particle Filter outputs
python cnn_tumor_pred.py

# 4. Clinical simulations
python magnified_baseline_6m.py magnified_drug_6m.py magnified_drug_comparison.py
```

---

## 📁 Project Files

```
├── least_square_params.py                # Parameter estimation
├── extended_kalman_least_square.py       # EKF filtering
├── unscented_kalman_least_square.py      # UKF filtering
├── particle_filter_least_square2.py      # Particle Filter
├── cnn_tumor_pred.py, cnn_patient_forcast.py  # CNN training/prediction
├── magnified_baseline_6m.py, magnified_drug_6m.py  # Simulations
├── Data_preparation/: TCGA_Data_prep.R, UCSC_Xena_Data_prep.R
└── Datasets/: NSCLC_Clinical_Cleaned.csv, EKF/UKF/PF_Export.csv
```
