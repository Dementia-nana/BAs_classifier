# BAs Classifier

## Overview

BAs Classifier is a machine learning pipeline developed for the supervised classification of bile acids (BAs) using structured raw feature matrix datasets.

The program trains multiple classification models, performs cross-validation, automatically selects the best-performing model based on cross-validation accuracy, and generates prediction results in Excel format.

This repository is intended for academic research purposes and supports reproducible model training and evaluation.



## 1. System Requirements

### 1.1 Script Dependencies

The script requires:
- Python 3.10
- numpy
- pandas
- matplotlib
- scikit-learn
- openpyxl

All required packages are listed in requirements.txt.

### 1.2 Operating System

The script has been tested on:
- Windows 10 (64-bit)
It should also be compatible with other operating systems (e.g., macOS, Linux), although this has not been explicitly tested.

### 1.3 Tested Python Version

- Python 3.10

### 1.4 Hardware Requirements

No non-standard hardware is required.
The script runs on a standard desktop computer (Intel i5/i7 class CPU, 8 GB RAM).

## 2. Installation Guide

### 2.1 Installation Instructions

1. Install Python 3.10.
2. Clone or download this repository.
3. Install required dependencies:

```bash
pip install -r requirements.txt
```

### Typical Installation Time

On a standard desktop computer with a normal internet connection, installation typically takes less than 5 minutes.

## 3. Demo

### 3.1 Example Dataset Size

A typical experimental setup includes:
- Training dataset:
  - 10 Excel files
  - File format: `.xlsx` or `.xls`
  - Approximately 1550 total samples (rows)
  - Numeric feature columns
  - One target column representing class labels
- Prediction dataset:
  - 1 Excel file
  - File format: `.xlsx` or `.xls`
  - Must contain feature columns only
  - No label column is required
  - Approximately 820 samples (rows)
  
### 3.2 Instructions to Run the Script

Run the main script using:

```bash
python BAs_classifier.py
```
### 3.3 Expected Output
The script generates the following outputs in the outputs/ directory:
1. Model Evaluation Files
- Excel file summarizing model performance:
  - Model name
  - Cross-validation mean accuracy
  - Cross-validation standard deviation
  - Training accuracy
- CSV files for learning curve data (if enabled)
- Feature importance or coefficient files (if enabled)

2. Confusion Matrices
- Training set confusion matrix (column-normalized)
- Cross-validation OOF confusion matrix (column-normalized)
- Corresponding CSV files of counts and rates (if enabled)

3. Prediction Results
- Excel file containing predicted class labels (type_predicted)

4. Additional Visualization Outputs 
After prediction, the script automatically generates:
- Scatter plot 1:
  - X-axis: mean (M1 or M2)
  - Y-axis: SD (S1 or S2)
  - Colored by predicted class
- Scatter plot 2:
  - X-axis: mean (M1 or M2)
  - Y-axis: log10(DT)
  - Colored by predicted class
- Model performance summary table (image format)
  - Model name
  - CV mean accuracy
  - CV standard deviation
  - Training accuracy
- Parallel coordinate plot
  - Five parameters (DT, M1, S1, M2, S2)
  - Colored by class
All figures are saved as high-resolution PNG files.

### 3.4 Expected Runtime

For a training dataset consisting of approximately 1550 samples and a prediction dataset of approximately 820 samples, 
the typical runtime is about 3 minutes on a standard desktop computer (Intel i5/i7 CPU, 8 GB RAM).

## 4. Instructions for Use

To apply the script to new data:
1. Prepare input data in Excel format consistent with the manuscript.
2. Ensure required column names are correctly formatted.
3. Run the script using the command shown above.
4. Check the output files and console metrics.

## 5. Reproducibility

To reproduce the quantitative results reported in the manuscript:
1. Use the same training and prediction datasets described in the manuscript.
2. Ensure Python 3.10 and the specified dependency versions are installed.
3. Run the script with identical input files and parameter settings.
4. The output metrics and predictions should match the reported results within numerical tolerance.

## 6. License
This project is released under the MIT License.

