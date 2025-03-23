<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<br />
<div align="center">
  <!-- PROJECT LOGO (Optional) -->
  <!-- <a href="https://github.com/your_username/total_perspective_vortex">
    <img src="assets/logo.png" alt="Logo" width="80" height="80">
  </a> -->

  <h3 align="center">Total Perspective Vortex: EEG Motor Imagery Classification Machine Learning</h3>

  <p align="center">
    A tool to process EEG data for motor imagery tasks using Python, scikit-learn, MNE, and MLflow.
    <br />
    <a href="https://github.com/your_username/total_perspective_vortex"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/your_username/total_perspective_vortex">View Demo</a>
    ·
    <a href="https://github.com/your_username/total_perspective_vortex/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/your_username/total_perspective_vortex/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#media">Media</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#additional-tips">Additional Tips</a></li>
    <li><a href="#sources">Sources</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

---

<!-- ABOUT THE PROJECT -->
## About The Project

**Short Description:**  
This repository provides tools to process EEG data for motor imagery tasks using Python, MNE, and scikit-learn. The focus is on filtering, feature extraction, dimensionality reduction (PCA/CSP), and classification of EEG signals. MLflow is integrated for experiment tracking and reproducibility, making it a valuable resource for brain-computer interface (BCI) research and development.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- BUILT WITH -->
## Built With

- [Python](https://www.python.org/)
- [scikit-learn](https://scikit-learn.org/)
- [MLflow](https://mlflow.org/)
- [MNE](https://mne.tools/stable/index.html)


<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- GETTING STARTED -->
## Getting Started

Follow these instructions to set up the project locally.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your_username/total_perspective_vortex.git
   cd total_perspective_vortex

2. **Setup the environment:**
   ```bash
   source ./setup_environment.sh

3. **Download EEG Motor Movement Data: Manually download the dataset from:**
   https://physionet.org/static/published-projects/eegmmidb/eeg-motor-movementimagery-dataset-1.0.0.zip


4. **Unzip the downloaded EEG data:**
   ```bash
   .
   ├── cleanup_environment.sh
   ├── config
   ├── data
   ├── logs
   ├── models
   ├── requirements.txt
   ├── setup_environment.sh
   ├── src
   ├── venv
   └── assets   # Contains logos, screenshots, GIFs, etc.
   ----------------------------------
   data
   ├── 64_channel_sharbrough-old.png
   ├── 64_channel_sharbrough.pdf
   ├── 64_channel_sharbrough.png
   ├── ANNOTATORS
   ├── RECORDS
   ├── S001
   ├── S002
   └── ... (S003 to S109)
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE -->
5. **Train your model:**
   ```bash
   cd src/main_app
   python train_bci.py

6. **Enable MLFLOW (optional):**
   ```bash
   cd src/main_app
   python train_bci.py --mlflow=true

7. **Make predictions:**
   ```bash
   cd src/main_app
   python predict_bci.py

8. **Make predictions with EEG plot (optional):**
   ```bash
   cd src/main_app
   python predict_bci.py --plot_eeg_predictions=true

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Media

This project includes demo GIFs and screenshots to showcase its functionality. All media files are stored in the `assets` folder.

**Demo GIFs:**

- **Training Process:**  
  ![Training Process](assets/training_1.gif)

- **Prediction Demo:**  
  ![Prediction Demo](assets/predictions_with_eeg_plot.gif)

- **MLFLOW Overview:**  
  ![Workflow Overview](assets/mlflow_1.gif)




   
