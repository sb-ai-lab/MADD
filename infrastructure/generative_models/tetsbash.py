from typing import List
import requests
import json
import socket
import time 
#import multiprocessing as mp 
from multiprocessing import Process
import os
import pandas as pd



def get_state_from_server(case:str,url:str):
    url_ = url.split("http://")[1]
    resp = requests.get("http://"+url_.split('/')[0]+"/check_state")
    if resp.status_code ==500:
        print(f"Server error:{resp.status_code}")
        return
    if case in json.loads(resp.content)['ml_state']:
        print("Case already trained!")
    state = json.loads(resp.content)
    return state

def train_ml_with_data(
    case = "Brain_cancer",
    data_path = "/projects/generative_models_data/generative_models/transformer/docked_data_for_train/data_4j1r.csv",#path to client data folder
    feature_column=['canonical_smiles'],
    target_column=['docking_score','QED','Synthetic Accessibility','PAINS','SureChEMBL','Glaxo','Brenk','IC50'], #All propreties from dataframe you want to calculate in the end
    regression_props = ['docking_score'], #Column name with data for regression tasks (That not include in calculcateble propreties)
    classification_props = ['IC50'], #Column name with data for classification tasks (That not include in calculcateble propreties)
    description = 'Case for Brain canser',
    timeout = 5, # min
    url: str = "http://10.64.4.243:81/train_ml",
    **kwargs,
):
    start_time = time.time()
    df = pd.read_csv(data_path).to_dict() # Transfer df to dict for server data transfer
    params = {
        'case': case,
        "data" : df,
        'target_column': target_column,
        'feature_column': feature_column,
        'timeout': timeout,
        'description' : description,
        'regression_props' : regression_props,
        'classification_props' : classification_props,
        **kwargs,
    }

    #Get state from server
    state = get_state_from_server(case=case,url=url)

    print(state['ml_state'])
    print(state['calc_propreties']) 
    #Get state from server
    #resp = requests.post(url,json.dumps(params))
    p = Process(target=requests.post,args=[url,json.dumps(params)])
    p.start()

    time.sleep(4)
    p.terminate()
    #resp = requests.post(url, data=json.dumps(params))
    print("--- %s seconds ---" % (time.time() - start_time))

def predict_smiles(smiles_list : List[str],
                   case : str = "Brain_cancer",
                   timeout : int = 10, #30 min
                   url: str = "http://10.64.4.243:81/predict_ml",
                   **kwargs,):
    params = {
        'case': case,
        'smiles_list' : smiles_list,
        'timeout': timeout,
        **kwargs,
    }
    resp = requests.post(url, json.dumps(params))
    return resp, resp.json()



if __name__=='__main__':
###############
#Test train with default params
    train_ml_with_data(case="test_api22")
    print('Process created')

################
#Test predict
    # case = "test_api"
    # smiles = ["Fc1cc(F)c2ccc(Oc3cncc4nnc(-c5ccc(OC(F)F)cc5)n34)cc2c1","Fc1cc(F)c2ccc(Oc3cncc4nnc(-c5ccc(OC(F)F)cc5)n34)cc2c1"]
    
    # print(predict_smiles(smiles_list=smiles,case=case))
    
####################