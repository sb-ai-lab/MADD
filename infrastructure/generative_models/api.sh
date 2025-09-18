#!/bin/bash
cd /projects/CoScientist
source /root/miniconda3/bin/activate Mol_gen_env
nohup python infrastructure/generative_models/main_api.py > api.txt
