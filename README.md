# Total Perspective Vortex: EEG Motor Imagery Classification

**Short Description:**  
This repository provides tools to process EEG data for motor imagery tasks using Python, MNE, and scikit-learn. We focus on filtering, feature extraction, dimensionality reduction (PCA/CSP), and classification of EEG signals.

## Installation

1. **Clone the repo**:
   ```bash
   git clone https://github.com/your_username/total_perspective_vortex.git
   cd total_perspective_vortex

2. **Setup virtual environment and download data**:
   ```bash
   source ./setup_environment

3. **Usage**:

**Train your model**:
   ```bash
   cd src/main_app
   python train_bci.py


Or for mlflow server and registry upload possibility


python train_bci.py --mlflow=true 

-Loads and filters EEG data
-Performs dimensionality reduction (PCA)
-Classifies and logs results

**Predict**:
   ```bash
   cd src/main_app
   python predict_bci.py


4. **Project structure**:
   ```bash
   
   ├── config
   ├── data #.edf files (not committed)
   ├── logs
   ├── models #.joblib model files (after training)
   ├── src
   │   ├── data_processing
   │   ├── experiments
   │   ├── main_app
   │   ├── mlflow
   │   ├── pipeline
   │   └── utils
   ├── requirements.txt
   ├── cleanup_environment.sh
   ├── setup_environment.sh


5. **Methodology**:
-Preprocessing & Feature extraction (Load raw EDF, Bandpass filter)
-Dimensionality Reduction (PCA)
-Classification (sckit-learn pipelines, cross-validation, aiming >= 60% accuracy on unseen data)

6. **License**:
MIT License

7. **References**:
-MNE documentation: https://mne.tools/stable/index.html
-Feature extraction paper: https://arxiv.org/pdf/1312.2877
-PCA


