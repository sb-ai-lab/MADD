# MADD

A comprehensive platform for AI-driven drug discovery and cheminformatics research. This app enables automated molecule generation, chemical property prediction, and end-to-end machine learning pipeline construction for chemical data.
## ✨ Features

- **Molecule Generation**: Create novel molecular structures using state-of-the-art generative models (VAE, GAN, Transformer-based).
- **AutoML Pipeline**: Automated machine learning workflow for model selection, hyperparameter tuning, and feature engineering specific to chemical data.
- **Chemical Properties Prediction**: Predict key molecular properties including QED, Docking score, Synthetic Accessibility, PAINS alerts, SureChEMBL patents, Glaxo alerts, Brenk alerts, Blood-Brain Barrier permeability, and IC50 values.
- **Database Integration**: Automated downloading and integration of bioactivity data from public repositories (ChEMBL, BindingDB).
- **Data Processing**: Comprehensive cheminformatics pipeline for data cleaning, standardization, and feature engineering.

## 📦 Installation

```bash
pip install git+https://github.com/alinzh/MADD.git
```

## 🚀 Quick Start
### Running MADD with GUI interface
To run it, you need to:
1) Fill out _config.yaml_. It is important to indicate the current addresses where you have deployed generative and predictive models.
2) Run next command in CLI:
   
```bash
python multi_agent_system/MADD_main/MADD_gui.py
```
3) Clik on the link. You will see something similar in your console:
```bash
* Running on local URL:  http://127.0.0.1:7869
* Running on public URL: https://a52bd0e6373fef859d.gradio.live
```

### Running MADD on our benchmark
To run it, you need to:
1) Fill out _config.yaml_ and specify the path to the dataset.
2) Run in CLI:
   
```bash
python multi_agent_system/MADD_main/MADD_run_on_benchmark.py > logs.txt
```
