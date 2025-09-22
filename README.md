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
git clone https://github.com/ITMO-NSS-team/MADD.git
cd MADD
pip install -e .
```

## Instructions for install and run AutoML models server

```
cd infrastructure/automl

path/to/python3.10.exe -m venv env

pip install -r requirements.txt

source env/Scripts/activate

python automl_api.py
```
In automl_api.py script you should set port where you want to run server.

## Instructions for build and run container with generative models

The easiest way to work with this part of the project is to build a container on a server with an available video card.

```
git clone https://github.com/ITMO-NSS-team/MADD.git
```

You need to specify the required parameters in the DockerFile, such as:
```
GEN_APP_PORT (the port on which you plan to deploy the container with generative models),
ML_MODEL_URL (The address (IP and port) where you plan to host the server with predictive models), 
HF_TOK (for downloading trained models), 
GITHUB_TOKEN (for the ability to make commits to the code).
```
```
cd infrastructure/generative_models

docker build -t generative_model_backend .
```

## Running a container

The container may take quite a long time to build, since the environment for its operation requires a long installation and time. However, this is done quite simply.

Next, after you have created an image on your server (or locally), you need to run the container with the command:
```
docker run --name molecule_generator --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=<your device ID> -it --init generative_model_backend:latest bash

OR 

docker run --name molecule_generator --runtime=nvidia -e --gpus all -it --init generative_model_backend:latest bash
```
The container should automatically launch a server with the FastAPI and generative models. However, if this doesn't happen, you should manually run the code
```
bash /projects/MADD/infrastructure/generative_models/api.sh
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
### Running DataGathering agent to download data from ChemBL and BindingDB
To run it, you need to:
1) Fill out _config.yaml_.
2) Run next command in CLI:
   
```bash
python multi_agent_system/MADD_main/run_data_gathering.py
```
3) Paste a query into the console, for example:
```bash
Download data from BindigDB for KRAS with IC50 values.
```
4) See where the downloaded file is located:
```bash
-----RESULT-----
The function 'fetch_BindingDB_data' has been executed with the specified parameters.
The output indicates that 142 entries for KRAS with IC50 values were found in BindingDB and saved
to a file named 'molecules_KRAS.csv' in the 'multi_agent_system/MADD_main/data_from_chem_db' directory.
```

### Running MADD on our benchmark
To run it, you need to:
1) Fill out _config.yaml_ and specify the path to the dataset.
2) Run in CLI:
   
```bash
python multi_agent_system/MADD_main/MADD_run_on_benchmark.py > logs.txt
```
