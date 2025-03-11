<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<!-- PROJECT SHIELDS (Optional) -->
<!--
*** You can remove any badges that are not relevant, or update the URLs to match your repo.
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<div align="center">
  <!-- PROJECT LOGO (Optional) -->
  <!-- <a href="https://github.com/your_username/total_perspective_vortex">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

  <h3 align="center">Total Perspective Vortex: EEG Motor Imagery Classification</h3>

  <p align="center">
    We provide tools to process EEG data for motor imagery tasks using Python, MNE, and scikit-learn.
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

---

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
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

---

<!-- ABOUT THE PROJECT -->
## About The Project

**Short Description:**  
This repository provides tools to process EEG data for motor imagery tasks using Python, MNE, and scikit-learn. We focus on filtering, feature extraction, dimensionality reduction (PCA/CSP), and classification of EEG signals.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- BUILT WITH -->
## Built With

- [Python](https://www.python.org/)
- [MNE](https://mne.tools/stable/index.html)
- [scikit-learn](https://scikit-learn.org/)
- [MLflow](https://mlflow.org/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- GETTING STARTED -->
## Getting Started

Below you will find the instructions to set up the project locally and prepare your environment.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your_username/total_perspective_vortex.git
   cd total_perspective_vortex
2. **Setup environment and download data:**
   ```bash
   source ./setup_environment.sh
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Usage
3. **Train your model:**
   ```bash
   cd src/main_app
   python train_bci.py
4. **(Optional) Enable Mlflow:**
   ```bash
   cd src/main_app
   python train_bci.py
<p align="right">(<a href="#readme-top">back to top</a>)</p>

5. **Predict:**
   ```bash
   cd src/main_app
   python train_bci.py --mlflow=true
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Sources
  -MNE: https://mne.tools/stable/index.html<br>
    -Feature exctraction paper: https://arxiv.org/pdf/1312.2877.pdf

## Project Structure
  ```bash
  total_perspective_vortex/
  ├── config/                  #Configuration files
  ├── data/                    #.edf files (not committed)
  ├── logs/                    #Log files
  ├── models/                  #Saved models (.joblib after training)
  ├── src/
  │   ├── data_processing/     #Data preprocessing modules
  │   ├── experiments/         #Experiment-related scripts
  │   ├── main_app/            #Main training and prediction scripts
  │   ├── mlflow/              #MLflow tracking configurations
  │   ├── pipeline/            #Custom pipeline definitions
  │   └── utils/               #Utility functions
  ├── requirements.txt         #Python dependencies
  ├── cleanup_environment.sh   #Clean-up script for the environment
  ├── setup_environment.sh     #Environment setup script
  └── README.md


