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
    """
    Retrieves the current training status from a server to avoid redundant training checks.
    
    Args:
        case (str): The identifier for the specific training case to query.
        url (str): The base URL of the server hosting the training state.
    
    Returns:
        dict: The server's state as a JSON object, providing details about training progress. Returns None if a server error (status code 500) is encountered.
    """
    url_ = url.split("http://")[1]
    resp = requests.get("http://"+url_.split('/')[0]+"/check_state")
    if resp.status_code ==500:
        print(f"Server error:{resp.status_code}")
        return
    if case in json.loads(resp.content)['state']:
        print("Case already trained!")
    state = json.loads(resp.content)
    return state

def train_ml_with_data(
    case = "Brain_cancer",
    data_path = "automl/data/data_4j1r.csv",#path to client data folder
    feature_column=['canonical_smiles'],
    target_column=['docking_score','QED','Synthetic Accessibility','PAINS','SureChEMBL','Glaxo','Brenk','IC50'], #All propreties from dataframe you want to calculate in the end
    regression_props = ['docking_score'], #Column name with data for regression tasks (That not include in calculcateble propreties)
    classification_props = ['IC50'], #Column name with data for classification tasks (That not include in calculcateble propreties)
    description = 'Case for Brain canser',
    timeout = 5, # min
    url: str = "http://10.64.4.247:81/train_ml",
    **kwargs,
):
    """
    Initiates a machine learning model training process on a remote server.
    
    This method prepares data from a CSV file and sends it to a specified server endpoint to train a machine learning model. It structures the data, including features and target properties, and handles the communication with the server using a separate process to avoid blocking.  The method also retrieves and prints the current state of the training process from the server.
    
    Args:
        case (str, optional): The name or identifier for the case/experiment. Defaults to "Brain_cancer".
        data_path (str, optional): The path to the CSV file containing the training data. Defaults to "automl/data/data_4j1r.csv".
        feature_column (list, optional): A list of column names to use as features for the model. Defaults to ['canonical_smiles'].
        target_column (list, optional): A list of column names representing the target properties. Defaults to ['docking_score','QED','Synthetic Accessibility','PAINS','SureChEMBL','Glaxo','Brenk','IC50'].
        regression_props (list, optional): A list of column names for regression tasks. Defaults to ['docking_score'].
        classification_props (list, optional): A list of column names for classification tasks. Defaults to ['IC50'].
        description (str, optional): A description of the case/experiment. Defaults to 'Case for Brain canser'.
        timeout (int, optional): The maximum time (in minutes) allowed for the training process. Defaults to 5.
        url (str, optional): The URL of the server endpoint for training. Defaults to "http://10.64.4.243:81/train_ml".
        kwargs (dict, optional): Additional keyword arguments to be passed to the server. Defaults to {}.
    
    Returns:
        None
    """
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
    """
    Predicts properties for a list of SMILES strings using a remote prediction service.
    
    This method facilitates property prediction for chemical compounds represented as SMILES strings. It packages the input SMILES list and prediction case into a JSON payload and sends it to a specified URL.  This allows leveraging external machine learning models to derive insights from molecular structures without requiring local computation.
    
    Args:
        smiles_list (List[str]): A list of SMILES strings representing the chemical compounds to analyze.
        case (str, optional): The specific prediction model or scenario to use. Defaults to "Brain_cancer".
        timeout (int, optional): The maximum time (in seconds) to wait for a response from the server. Defaults to 10.
        url (str, optional): The URL of the remote prediction service. Defaults to "http://10.64.4.243:81/predict_ml".
        **kwargs: Additional parameters to be included in the request.
    
    Returns:
        tuple: A tuple containing the response object from the `requests` library and the JSON decoded response from the server.  The JSON response contains the predicted properties.
    """
    params = {
        'case': case,
        'smiles_list' : smiles_list,
        'timeout': timeout,
        **kwargs,
    }
    resp = requests.post(url, json.dumps(params))
    return resp, resp.json()



if __name__=='__main__':
    #print(get_state_from_server(case="egfiction",url='http://10.64.4.247:81'))
###############
#Test train with default params
    train_ml_with_data(case="Alzheimer", feature_column=['canonical_smiles'],
    target_column=['docking_score','QED','Synthetic Accessibility','PAINS','SureChEMBL','Glaxo','Brenk','IC50'], #All propreties from dataframe you want to calculate in the end
    regression_props = ['docking_score'], #Column name with data for regression tasks (That not include in calculcateble propreties)
    classification_props = ['IC50'],
    data_path=r'D:\Projects\CoScientist\automl\data\base_cases\alz.csv',
    save_trained_data_to_sync_server = True,
    description = 'Case for Alzheimer with trained FEDOT pipelines for predict "docking_score", and "IC50"',
    timeout=60*20
    )
    print('Process created')

################
#Test predict
    # case = "test_api"
    # smiles = ["Fc1cc(F)c2ccc(Oc3cncc4nnc(-c5ccc(OC(F)F)cc5)n34)cc2c1","Fc1cc(F)c2ccc(Oc3cncc4nnc(-c5ccc(OC(F)F)cc5)n34)cc2c1"]
    
    # print(predict_smiles(smiles_list=smiles,case=case))
    
####################