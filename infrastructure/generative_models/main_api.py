from logging import debug
from typing import List, Union
from fastapi import FastAPI,Body
import sys 
import os
from pydantic import BaseModel
import uvicorn
import json
from utils.check_novelty import check_novelty_mol_path
import_path = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from api_utils import *
import socket
from generative_models.ic50_classifire_model.read_ic50 import Ic50
import pandas as pd
import threading
import pickle as pi
import lightgbm
from huggingface_hub import hf_hub_download,snapshot_download

###Docking
from generative_models.autodock_vina_python3.src.docking_score import docking_list
import yaml
import sys
print(sys.path)

with open("infrastructure/generative_models/config.yaml", "r") as file:
    config = yaml.safe_load(file)

is_public_API = config['is_public_API']
local_dir = '/projects/CoScientist/infrastructure/generative_models/autotrain/utils'
if __name__=='__main__':
    # if not os.path.isdir('infrastructure/generative_models/many_prop_CVAE'):
    #     snapshot_download(repo_id="SoloWayG/Molecule_transformer",
    #             allow_patterns ="many_prop_CVAE/*",
    #             local_dir='infrastructure/generative_models',
    #             force_download=True,
    #             token=os.getenv("HF_TOKEN"))
    def get_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(('10.254.254.254', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP
    
    print("Getting IP")
    ip = str(get_ip())
    print(f"Current IP: {ip}")
    print("Starting...")

    app = FastAPI(debug=True)

    # API operations
    @app.get("/")
    def health_check():
        return {'health_check': 'OK'}

    @app.post("/eval_docking_alzheimer")
    def eva_altz(data:Docking_config):
        print('start')
        print(data)
        path = cases[data.case]['docking_paths']
        d_s = docking_list(smiles=data.mol_list,path_receptor_pdb=path)
        return {'docking_scores' : d_s}
    
    @app.post("/eval_docking")
    def eva_altz(data:Docking_config):
        print('start')
        print(data)
        path = docking_paths[data.receptor_case]
        d_s = docking_list(smiles=data.mol_list,path_receptor_pdb=path)
        return {'docking_scores' : d_s}
    
    @app.post("/random_search")
    def search(data:GenData=Body()):
        print('Random generation start!')
        #####FOR TEST####
        if data.numb_mol>100:
            data.numb_mol=100
        ##################
        """Function to init random case generation.

        Args:
            data (GenData): Namespace with case information.
            cuda (bool, optional): Choose cuda usage option. Defaults to False.

        Returns:
            request (dict): With molecules and avalutred propersties.
        """
        data.case_ = 'RNDM'
        return json.dumps(case_generator(data))
    
    @app.post("/search_Skleroz")
    def search(data:GenData=Body()):
        #####FOR TEST####
        if data.numb_mol>100:
            data.numb_mol=100
        ##################
        """Function to init Skleroz case generation.
        Args:
            data (GenData): Namespace with case information.
            cuda (bool, optional): Choose cuda usage option. Defaults to False.
        Returns:
            request (dict): With molecules and avalutred propersties.
        """
        data.case_ = 'Sklrz'
        return json.dumps(case_generator(data))
    
    @app.post("/search_Parkinson")
    def search(data:GenData=Body()):
        #####FOR TEST####
        if data.numb_mol>100:
            data.numb_mol=100
        ##################
        """Function to init Parkinson case generation.
        Args:
            data (GenData): Namespace with case information.
            cuda (bool, optional): Choose cuda usage option. Defaults to False.
        Returns:
            request (dict): With molecules and avalutred propersties.
        """
        data.case_ = 'Prkns'
        return json.dumps(case_generator(data))
    
    @app.post("/search_Canser")
    def search(data:GenData=Body()):
        if data.numb_mol>100:
            data.numb_mol=100
        """Function to init Canser case generation.
        Args:
            data (GenData): Namespace with case information.
            cuda (bool, optional): Choose cuda usage option. Defaults to False.
        Returns:
            request (dict): With molecules and avalutred propersties.
        """
        data.case_ = 'Cnsr'
        return json.dumps(case_generator(data))
    
    @app.post("/search_Dyslipidemia")
    def search(data:GenData=Body()):
        #####FOR TEST####
        if data.numb_mol>100:
            data.numb_mol=100
        """Function to init Dyslipidemia case generation.
        Args:
            data (GenData): Namespace with case information.
            cuda (bool, optional): Choose cuda usage option. Defaults to False.
        Returns:
            request (dict): With molecules and avalutred propersties.
        """
        data.case_ = 'Dslpdm'
        return json.dumps(case_generator(data))
    
    @app.post("/search_Drug_resist")
    def search(data:GenData=Body()):
        #####FOR TEST####
        if data.numb_mol>100:
            data.numb_mol=100
        """Function to init Drug resistance case generation.
        Args:
            data (GenData): Namespace with case information.
            cuda (bool, optional): Choose cuda usage option. Defaults to False.
        Returns:
            request (dict): With molecules and avalutred propersties.
        """
        data.case_ = 'TBLET'
        return json.dumps(case_generator(data))

    @app.post("/search_Alzheimer")
    def search(data:GenData=Body()):
        #####FOR TEST####
        if data.numb_mol>100:
            data.numb_mol=100
        """Function to init Alzheimer case generation.
        Args:
            data (GenData): Namespace with case information.
            cuda (bool, optional): Choose cuda usage option. Defaults to False.
        Returns:
            request (dict): With molecules and avalutred propersties.
        """
        data.case_ = 'Alzhmr'
        return json.dumps(case_generator(data))

    @app.post("/case_generator")
    def case_run(data:GenData=Body()):
        return json.dumps(case_generator(data))
    

    @app.post("/gan_case_generator")
    def case_run(data:GenData=Body()):
        hf_hub_download(repo_id="SoloWayG/Molecule_transformer",
                         filename="state.json",
                         local_dir=local_dir,
                         force_download=True,
                         token=os.getenv("HF_TOKEN"))
        snapshot_download(repo_id="SoloWayG/Molecule_transformer",
                    allow_patterns ="GAN_weights/*",
                    local_dir='/projects/CoScientist/infrastructure/generative_models/autotrain/',
                    force_download=True,
                    token=os.getenv("HF_TOKEN"))
        return json.dumps(gan_auto_generator(data))
    
    @app.post("/train_gen_models")
    def case_run(data:TrainData=Body()):
        
        hf_hub_download(repo_id="SoloWayG/Molecule_transformer",
                         filename="state.json",
                         local_dir=local_dir,
                         force_download=True,
                         token=os.getenv("HF_TOKEN"))
        case_trainer(data)
        #return json.dumps()

    @app.post("/train_gan")
    def case_gan_run(data:TrainData=Body()):
        print(os.getenv("HF_TOKEN"))
        hf_hub_download(repo_id="SoloWayG/Molecule_transformer",
                         filename="state.json",
                         local_dir=local_dir,
                         force_download=True,
                         token=os.getenv("HF_TOKEN"))
        print('train')
        gan_case_trainer(data)

    @app.post("/generate_gen_models_by_case")
    def gen_mol_case(data:TrainData=Body()):
        
        hf_hub_download(repo_id="SoloWayG/Molecule_transformer",
                         filename="state.json",
                         local_dir=local_dir,
                         force_download=True
                         ,token=os.getenv("HF_TOKEN"))
        ret = auto_generator(data)
        return json.dumps(ret)
    
    
    
    @app.get("/check_state")
    def check_state():
        state = TrainState(state_path=f'{local_dir}/state.json')
        calc_properies = state.show_calculateble_propreties()
        current_state = state().copy()
        
        del current_state["Calculateble properties"]
        return {'state': current_state,
                'calc_propreties':list(calc_properies)}

    #############################################
        ######### WILL BE SUPPRESED!!!#########
    @app.post("/eva_altz")
    def eva_altz(data:Molecules):
        print('start')
        print(data)
        Ic50_results,df = Ic50(data.mol_list)
        return {'Ic50_results':list(Ic50_results)}
    
    @app.post("/eva_altz_sorted")
    def eva_altz_sorted(data:Molecules):
        print('start')
        print(data)
        Ic50_results,df = list(Ic50(data.mol_list))
        ar = np.array(Ic50_results)
        result = [(data.mol_list[i],ar[i]) for i in list(np.where(ar<=4)[0])]
        
        return {'mol_list_with_ic50_score':result}
    

        ######### WILL BE SUPPRESED!!!#########
    #############################################

    if is_public_API:
        uvicorn_ip = ip
    else:
        uvicorn_ip = '127.0.0.1' 
    uvicorn.run(app,host=uvicorn_ip,port=int(os.getenv('GEN_APP_PORT')),log_level='info')
