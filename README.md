# Patient-Specific NSCLC Tumor Dynamics Modeling with Bayesian Filtering

A comprehensive computational framework for modeling non-small cell lung cancer (NSCLC) tumor-immune system dynamics, parameter estimation, and patient-specific forecasting using Bayesian filtering techniques and deep learning.

**Master's Thesis Project** | Stochastic Simulation & Bayesian Filtering in Cancer Dynamics

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Results & Outputs](#results--outputs)
- [File Descriptions](#file-descriptions)
- [References](#references)
- [License](#license)

---

## 🎯 Overview

This project develops a patient-specific computational framework for modeling NSCLC (non-small cell lung cancer) tumor dynamics by:

1. **Mathematical Modeling**: Implementing ODE/SDE tumor-immune system dynamics with realistic biological parameters
2. **Parameter Estimation**: Using nonlinear least squares optimization to estimate patient-specific parameters from TCGA cohort data
3. **Bayesian Filtering**: Applying Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), and Particle Filter (PF) for state estimation and forecasting
4. **Deep Learning Prediction**: Training CNNs for high-speed, accurate tumor size forecasting
5. **Clinical Translation**: Providing tools for simulating treatment scenarios and generating patient-specific predictions

**Target Data**: TCGA (The Cancer Genome Atlas) LUNG cohort with genomic and clinical information

---

## ✨ Key Features

### Mathematical Models
- **ODE System**: 22-parameter tumor-immune interaction model capturing:
  - High-density (H) and low-density (L) tumor cell populations
  - T-cell immune responses
  - PD-1/PD-L1 checkpoint interactions (immunotherapy response)
  - Drug treatment dynamics (anti-PD-1 therapy)

### Bayesian State Estimation
- **Extended Kalman Filter (EKF)**: Linear approximation for state estimation
- **Unscented Kalman Filter (UKF)**: Sigma-point based nonlinear filtering
- **Particle Filter (PF)**: Sequential Monte Carlo for highly nonlinear scenarios
- Real-time uncertainty quantification and confidence intervals

### Deep Learning
- **CNN Architecture**: 1D Convolutional Neural Network optimized for tumor forecasting
  - Inference time: 1-2 ms per patient
  - Mean Absolute Error: 2.6 mm
  - R² Score: 0.83
  - MC Dropout for uncertainty estimation

### Clinical Simulations
- Baseline tumor dynamics (no treatment)
- Drug response scenarios (anti-PD-1 therapy at various dose levels)
- Patient-specific drug response profiles
- 6-month and long-term forecasts

---

## 📁 Project Structure

```
├── README.md                                    # This file
├── LICENSE                                      # MIT License
│
├── Core Scripts (Python)
├── least_square_params.py                       # Parameter estimation via nonlinear least squares
├── least_square_parameters.json                 # Estimated population-average parameters
│
├── Bayesian Filtering & Forecasting
├── extended_kalman_least_square.py              # EKF implementation
├── unscented_kalman_least_square.py             # UKF implementation  
├── particle_filter_least_square2.py             # Particle Filter implementation
├── ekf_export.py                                # EKF result export & visualization
├── ukf_export.py                                # UKF result export & visualization
├── pf_export.py                                 # Particle Filter result export & visualization
│
├── Deep Learning
├── cnn_tumor_pred.py                            # CNN model training pipeline
├── cnn_patient_forcast.py                       # CNN patient-specific forecasting module
│
├── Clinical Simulations & Visualizations
├── magnified_baseline_6m.py                     # 6-month baseline tumor dynamics (no treatment)
├── magnified_drug_6m.py                         # 6-month drug response simulation
├── magnified_drug_comparison.py                 # Multi-scenario drug response comparison
├── Fig2c_Initial_Magnified.py                   # Publication figure generation
│
├── Data Preparation/
├── TCGA_Data_prep.R                             # TCGA RNA-seq data download & processing
├── UCSC_Xena_Data_prep.R                        # UCSC Xena clinical data preparation
│
└── Datasets/
    ├── TCGA_Lung_All_Features.csv               # Complete TCGA cohort (genomic + clinical)
    ├── TCGA_Lung_Clinical_Cleaned.csv           # Clinical data (cleaned)
    ├── TCGA_Lung_Genomic_Integrated.csv         # Genomic data (integrated)
    ├── NSCLC_Clinical_Cleaned.csv               # NSCLC subset (cleaned for modeling)
    ├── merged_and_cleaned_data.csv              # Final merged dataset
    ├── lung_all.csv                             # All lung cancer samples
    │
    ├── EKF_Export.csv                           # EKF filtering results
    ├── UKF_Export.csv                           # UKF filtering results
    ├── PF_Export.csv                            # Particle Filter results
    │
    ├── PF_Results_least_square.csv              # Particle Filter optimization results
    ├── PF_Results_least_square2.csv             # PF results (variant 2)
    └── UKF_Results_least_square.csv             # UKF optimization results
```

---

## 📊 Data Sources

### Primary Data
- **TCGA LUNG Cohort**: ~1,000 NSCLC patients
  - Downloaded via `TCGAbiolinks` R package (GDC API)
  - RNA-seq gene expression (STAR-aligned counts)
  - Clinical data: T-stage, histology, survival, demographics

### Data Preparation Pipeline
1. **TCGA RNA-seq Download** (`TCGA_Data_prep.R`)
   - Queries GDC for LUAD/LUSC primary tumors
   - Processes raw counts with edge/limma normalization
   - Extracts PD-1, PD-L1, immune markers

2. **Clinical Data Preparation** (`UCSC_Xena_Data_prep.R`)
   - Harmonizes staging information
   - Maps T-stages to initial tumor volumes
   - Cleans missing/inconsistent values

---

## 🔬 Methodology

### 1. Mathematical Model
The system is described by a 22-parameter ODE:

```
dH/dt = α_H * H * (1 - (H+L)/K) - δ_HS * (1-p₁) * H/denom * T - δ_HF * p₁ * H * T
dL/dt = α_L * L * (1 - (H+L)/K) - δ_LS * (1-p₂) * L/denom * T - δ_LF * p₂ * L * T
dT/dt = (μ + α_HT * H/(κ₂+H) * T + α_LT * L/(κ₂+L) * T) * F(P,L) - δ_HT * H * T - δ_LT * L * T - δ_T * T
```

Where:
- **H, L**: High/Low-density tumor cell populations
- **T**: T-cell immune response
- **F(P,L)**: PD-1/PD-L1 checkpoint function = 1 / (1 + P*L/k_TQ)
- **P**: PD-1 on T cells = ρ_p * T
- **L**: PD-L1 on tumor = ρ_l * (T + ε_c * (N + M))
- **Parameters**: α, δ, p, κ, ρ, k_TQ, ε_c (biologically motivated)

### 2. Parameter Estimation
- **Method**: Nonlinear least squares optimization
- **Data**: TCGA NSCLC T-stage distribution (mapped to initial tumor volumes)
- **Objective**: Minimize error between model predictions and population statistics
- **Output**: `least_square_parameters.json` (population-average parameters)

### 3. Bayesian Filtering
Three complementary approaches for state estimation:

| Filter | Approach | Advantages | Use Case |
|--------|----------|-----------|----------|
| **EKF** | First-order linearization | Fast, closed-form | Real-time applications |
| **UKF** | Sigma-point sampling | Higher-order accuracy | Improved predictions |
| **PF** | Sequential Monte Carlo | Handles arbitrary distributions | Nonlinear uncertainty |

### 4. CNN Architecture
- **Input**: Last 3 time-steps of tumor size (look-back = 3)
- **Layers**: Conv1D → BatchNorm → ReLU → (×3 blocks) → GlobalAvgPool → Dense
- **Regularization**: Dropout (p=0.3), L2 penalty
- **Training**: Adam optimizer, MAE loss, 80/20 train/test split
- **Inference**: 1-2 ms per patient (vs. 5-10 ms for ensemble)

### 5. Clinical Simulations
- **Baseline**: Untreated tumor trajectory (A_drug = 0)
- **Drug Scenarios**: F (checkpoint inhibition) varies with anti-PD-1 concentration
- **Visualization**: Multi-panel plots showing H, L, T dynamics over time
- **Output**: Predictions with confidence intervals

---

## 🛠️ Installation

### Requirements
```
Python 3.9+
R 4.0+  (for data preparation only)
```

### Python Packages
```bash
pip install numpy scipy pandas matplotlib
pip install scikit-learn
pip install tensorflow>=2.10
pip install TCGAbiolinks  # Optional: if re-downloading data
```

### R Packages (optional, for data prep)
```R
install.packages(c("TCGAbiolinks", "SummarizedExperiment", "edgeR", "limma", "dplyr"))
# BiocManager required for some packages
```

### Quick Setup
```bash
cd /home/daniel/Desktop/Masters_thesis/Stochastic_simulation_Bayesian_filtering_in_cancer_dynamics
python -m pip install -r requirements.txt  # if available
```

---

## 📖 Usage Guide

### 1. Parameter Estimation (One-Time)
Estimate population-average model parameters from TCGA data:
```bash
python least_square_params.py
```
**Output**: `least_square_parameters.json`

### 2. Bayesian Filtering & Forecasting

#### Extended Kalman Filter
```bash
python extended_kalman_least_square.py
python ekf_export.py
```
**Outputs**:
- `EKF_Export.csv` (state estimates, covariances, predictions)
- Visualization plots

#### Unscented Kalman Filter
```bash
python unscented_kalman_least_square.py
python ukf_export.py
```
**Outputs**:
- `UKF_Export.csv` (state estimates, predictions)
- Comparison plots

#### Particle Filter
```bash
python particle_filter_least_square2.py
python pf_export.py
```
**Outputs**:
- `PF_Export.csv` (ensemble predictions)
- Confidence interval visualizations

### 3. CNN Training & Forecasting

#### Train Model
```bash
python cnn_tumor_pred.py
```
**Outputs**:
- `cnn_model/tumor_cnn_model.keras` (trained model)
- Training history plots
- Performance metrics (MAE, R², predictions vs. actual)

#### Single Patient Forecast
```bash
python cnn_patient_forcast.py
```
**Outputs**:
- Patient-specific 6-month predictions
- Confidence intervals (MC Dropout)



**`least_square_params.py`**
- Implements nonlinear least squares optimization
- Fits population-average model parameters to TCGA T-stage distribution
- Uses scipy.optimize.minimize with Nelder-Mead
- Output: `least_square_parameters.json` (22 parameters)

**`extended_kalman_least_square.py`**
- Extended Kalman Filter for nonlinear ODE systems
- Jacobian-based state propagation and update
- Assumes Gaussian noise, known covariance
- Output: State estimates, covariances, predictions

**`unscented_kalman_least_square.py`**
- Unscented Kalman Filter using sigma-point approximation
- Higher-order accuracy than EKF without explicit Jacobians
- Tunable alpha, beta, kappa parameters
- Output: State estimates, predictions with reduced linearization error

**`particle_filter_least_square2.py`**
- Sequential Importance Resampling (SIR) particle filter
- Multiprocessing for efficiency (parallel likelihood evaluation)
- Adaptive resampling (ESS threshold)
- Output: Ensemble of particle trajectories, prediction distribution

### Visualization & Export

**`ekf_export.py`, `ukf_export.py`, `pf_export.py`**
- Load filtering results
- Generate comparison plots (predictions vs. actual, uncertainties)
- Export to CSV for further analysis
- Create publication-quality figures

### Deep Learning

**`cnn_tumor_pred.py`**
- Build and train 1D CNN model
- Data pipeline: T → time-series → sequences
- 80/20 train/test split with stratification
- Batch normalization, dropout, L2 regularization
- Output: Trained `.keras` model, performance metrics

**`cnn_patient_forcast.py`**
- Load pre-trained CNN model
- High-speed patient-specific forecasting (1-2 ms)
- Monte Carlo dropout for confidence intervals
- Batch prediction capability



### Data Preparation

**`Data_preparation/TCGA_Data_prep.R`**
- Download TCGA LUAD/LUSC RNA-seq via GDC API
- Normalize counts using edgeR/limma (TMM normalization)
- Extract PD-1, PD-L1, T-cell signature scores
- Merge with clinical metadata

**`Data_preparation/UCSC_Xena_Data_prep.R`**
- Fetch harmonized clinical data from UCSC Xena
- Standardize T-stage encoding
- Handle missing values
- Output: `NSCLC_Clinical_Cleaned.csv`

---

## 🔧 Parameter Reference


## 📚 References

Key papers and methodologies:

1. **Tumor-Immune ODE Model**: Wang et al. (2023) - Original formulation
2. **Kalman Filtering**: Welch & Bishop (2006) - EKF/UKF fundamentals
3. **Particle Filtering**: Doucet et al. (2000) - Sequential Monte Carlo methods
4. **CNN Architecture**: Krizhevsky et al. (2012) - Deep learning for time series
5. **TCGA Data**: TCGA Consortium (Nature 2013) - NSCLC genomics and clinical data
6. **Checkpoint Immunotherapy**: Topalian et al. (2015) - PD-1/PD-L1 biology

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details

Copyright (c) 2026 dannychemm123

---

## 🤝 Contributing

This is a Master's thesis project. For questions or contributions, please contact the author.

---

## 📧 Contact & Citation

**Author**: Daniel Chem  
**Affiliation**: Master's Thesis Program  
**Email**: dannychemm123@[institution]

**Cite as**:
```
.
```

---

## 🎓 Acknowledgments

- TCGA/TCIA for public genomic and clinical data
- Supervision and guidance from thesis advisor(s)
- Open-source community (NumPy, SciPy, TensorFlow, R packages)
