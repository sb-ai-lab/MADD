# MADD

A modern Python framework for generating drug molecules for the treatment of diseases.

## ✨ Features

- **Molecule generation**: bla bla
- **AutoML**: bla bla
- **Chemical properties prediction**:  bla bla
- **Downloads data from ChemBL and BindingDB**:  bla bla
- **Dataset processing**:  bla bla

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
