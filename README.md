# Patient-Specific NSCLC Tumor Dynamics Modeling with Bayesian Filtering

Computational framework for modeling NSCLC (non-small cell lung cancer) tumor-immune dynamics, parameter estimation, and forecasting using Bayesian filtering and deep learning.

---

## Overview

**Pipeline**: TCGA Data → Parameter Estimation → Bayesian Filtering (EKF/UKF/PF) → CNN Prediction → Clinical Simulations

This project:
1. Estimates 22-parameter ODE model from NSCLC TCGA cohort (T-stage distribution)
2. Applies three Bayesian filters to NSCLC data for state estimation and forecasting
3. Uses Particle Filter (PF) estimates as training data for CNN
4. Trains CNN for fast, accurate tumor size predictions
5. Simulates drug response scenarios with anti-PD-1 therapy

**Data**: ~1,000 NSCLC patients from TCGA LUNG cohort (LUAD/LUSC)

---

## 📁 Project Structure

```
├── least_square_params.py                # Parameter estimation (NSCLC)
├── least_square_parameters.json          # Estimated parameters
│
├── extended_kalman_least_square.py       # EKF filtering
├── unscented_kalman_least_square.py      # UKF filtering  
├── particle_filter_least_square2.py      # Particle Filter (→ CNN training data)
├── ekf_export.py, ukf_export.py, pf_export.py  # Results export
│
├── cnn_tumor_pred.py                     # Train CNN on PF estimates
├── cnn_patient_forcast.py                # CNN predictions
│
├── magnified_baseline_6m.py              # No treatment simulation
├── magnified_drug_6m.py                  # Drug response simulation
├── magnified_drug_comparison.py          # Multi-dose comparison
├── Fig2c_Initial_Magnified.py            # Publication figures
│
├── Data_preparation/
│   ├── TCGA_Data_prep.R
│   └── UCSC_Xena_Data_prep.R
│
└── Datasets/
    ├── NSCLC_Clinical_Cleaned.csv        # NSCLC patient data
    ├── EKF_Export.csv, UKF_Export.csv    # Filter results
    ├── PF_Export.csv                     # PF predictions (used for CNN)
    └── [other processed datasets]
```

---

## 🔬 Methodology

**ODE Model** (22 parameters)
```
dH/dt = α_H·H·(1-(H+L)/K) - δ_HS·H·T - δ_HF·H·T
dL/dt = α_L·L·(1-(H+L)/K) - δ_LS·L·T - δ_LF·L·T  
dT/dt = (μ + α_HT·H·T + α_LT·L·T)·F(P,L) - δ_HT·H·T - δ_LT·L·T - δ_T·T
```
- **H, L**: High/low-density tumor cells
- **T**: T-cell immune response
- **F(P,L)**: PD-1/PD-L1 checkpoint function (affects immunotherapy response)

**Step 1: Parameter Estimation**
- Fit 22-parameter ODE to NSCLC T-stage distribution using least squares
- Output: `least_square_parameters.json`

**Step 2: Bayesian Filtering (on NSCLC data)**
- Apply EKF, UKF, and PF to estimate state trajectories and predictions
- Particle Filter generates state estimates → used as training data for CNN

**Step 3: Deep Learning**
- Train CNN on Particle Filter outputs
- Input: Last 3 time-steps of tumor size
- Output: Fast predictions (1-2 ms/patient) with confidence intervals
- Performance: MAE 2.6 mm, R² 0.83

**Step 4: Clinical Simulations**
- Model untreated baseline vs. drug scenarios
- Simulate anti-PD-1 therapy effects

---

## 🛠️ Installation

**Requirements**: Python 3.9+

```bash
pip install numpy scipy pandas matplotlib scikit-learn tensorflow>=2.10
```

---

## 📖 Quick Start

**1. Parameter Estimation (NSCLC)**
```bash
python least_square_params.py
```

**2. Run Bayesian Filters (NSCLC)**
```bash
python extended_kalman_least_square.py && python ekf_export.py
python unscented_kalman_least_square.py && python ukf_export.py
python particle_filter_least_square2.py && python pf_export.py
```

**3. Train CNN on Particle Filter Data**
```bash
python cnn_tumor_pred.py
```

**4. Clinical Simulations**
```bash
python magnified_baseline_6m.py          # Untreated
python magnified_drug_6m.py              # With drug
python magnified_drug_comparison.py      # Multi-dose
```

---

## � Data Preparation (Critical Pipeline)

This project relies on rigorous data preparation to ensure valid NSCLC patient cohort for modeling. The data pipeline involves three main stages:

### **Stage 1: TCGA Data Acquisition**

**Source**: The Cancer Genome Atlas (TCGA) LUNG project
- **Cohort**: LUAD (Lung Adenocarcinoma) + LUSC (Lung Squamous Cell Carcinoma)
- **Sample Size**: ~1,000 NSCLC patients with complete clinical & genomic data
- **Download Method**: GDC (Genomic Data Commons) API via TCGAbiolinks R package

**Data Types Retrieved**:
- **RNA-seq Gene Expression**: STAR-aligned raw counts (HTSeq quantification)
- **Clinical Variables**: T-stage, N-stage, M-stage, histology, age, smoking status, survival outcomes
- **Sample Type**: Primary tumor tissue only (no metastatic or recurrent samples)

**Script**: `Data_preparation/TCGA_Data_prep.R`
- Queries GDC database for TCGA-LUAD and TCGA-LUSC projects
- Downloads primary tumor RNA-seq data in parallel (files-per-chunk = 10)
- Extracts sample metadata and clinical phenotypes
- Processes ~50GB of genomic data efficiently

### **Stage 2: RNA-seq Normalization & Feature Extraction**

**Processing Steps**:
1. **Read Counting**: Extract raw gene counts from HTSeq quantification files
2. **Quality Control**: Filter genes with low expression (CPM < 1 across samples)
3. **Normalization**: Apply TMM (Trimmed Mean of M-values) normalization via edgeR package
4. **Log-transformation**: Convert normalized counts to log₂(CPM + 1) scale

**Immune Marker Extraction**:
- **PD-1 (PDCD1)**: T-cell exhaustion marker
- **PD-L1 (CD274)**: Tumor immune evasion marker
- **CD8A, CD4**: T-cell abundance markers
- **FOXP3**: Regulatory T-cell marker
- Create immune signature scores reflecting T-cell activation/suppression

**Output**: Normalized expression matrix with ~20,000 genes × 1,000 patients

### **Stage 3: Clinical Data Cleaning & Harmonization**

**UCSC Xena Data Integration** (`UCSC_Xena_Data_prep.R`):
- Download harmonized TCGA clinical data from UCSC Xena platform
- Standardize T-stage encoding (1, 2, 3, 4 classification)
- Map T-stage to initial tumor volumes:
  - T1 (~15 mm) → ~1.77 billion cells
  - T2 (~40 mm) → ~33.5 billion cells
  - T3 (~60 mm) → ~113 billion cells
  - T4 (~80 mm) → ~268 billion cells
  
**Exclusion Criteria**:
- Patients with missing T-stage information
- Non-primary tumor samples
- Incomplete follow-up data
- M-stage > 0 (metastatic disease)

**Data Quality Checks**:
- Remove duplicate samples
- Validate clinical variable consistency
- Check for data entry errors (e.g., impossible dates)
- Ensure genomic-clinical sample alignment

### **Stage 4: Final Dataset Merging**

**Output Files Generated**:
- `NSCLC_Clinical_Cleaned.csv`: ~800 patients with complete clinical data
- `TCGA_Lung_Clinical_Cleaned.csv`: Raw clinical variables
- `TCGA_Lung_Genomic_Integrated.csv`: Normalized expression + signatures
- `merged_and_cleaned_data.csv`: Complete integrated dataset ready for modeling

**Dataset Characteristics**:
- Patients: ~800 NSCLC (LUAD: 60%, LUSC: 40%)
- T-stage distribution: T1/T2/T3/T4 (balanced across stages)
- Clinical variables: Age, gender, smoking, stage, survival
- Genomic features: 20,000 genes + immune signatures
- No missing values in critical fields

**Critical Notes**:
- All parameter estimation uses cleaned T-stage distribution
- Bayesian filtering applied only to NSCLC-derived initial conditions
- CNN training uses Particle Filter estimates from NSCLC models
- Ensures internal consistency: mathematical model → filtering → deep learning

---

## �📄 File Descriptions

**`least_square_params.py`**
- Parameter estimation (NSCLC) using nonlinear least squares
- Fits ODE to NSCLC T-stage distribution

**Filtering Scripts** (applied to NSCLC data)
- `extended_kalman_least_square.py`: EKF filtering
- `unscented_kalman_least_square.py`: UKF filtering  
- `particle_filter_least_square2.py`: Particle Filter (output used for CNN training)

**Export & Visualization**
- `ekf_export.py`, `ukf_export.py`, `pf_export.py`: Export results and plots

**Deep Learning**
- `cnn_tumor_pred.py`: Train CNN on Particle Filter estimates
- `cnn_patient_forcast.py`: Fast predictions using trained CNN

**Clinical Simulations**
- `magnified_baseline_6m.py`: No treatment dynamics
- `magnified_drug_6m.py`: Anti-PD-1 drug response
- `magnified_drug_comparison.py`: Multi-dose scenarios
- `Fig2c_Initial_Magnified.py`: Publication figures

---




## 📚 References

1. Wang et al. (2023) - Tumor-Immune ODE Model
2. Welch & Bishop (2006) - Kalman Filtering
3. Doucet et al. (2000) - Sequential Monte Carlo
4. TCGA Consortium - NSCLC genomic and clinical data

---

## 📝 License

MIT License - See LICENSE file

**Author**: Daniel Chem
