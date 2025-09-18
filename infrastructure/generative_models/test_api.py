import requests
import json
import socket
import time 
import pandas as pd
from multiprocessing import Process
import sys
print(sys.path)
def get_state_from_server(case:str,url:str):
    url_ = url.split("http://")[1]
    resp = requests.get("http://"+url_.split('/')[0]+"/check_state")
    if resp.status_code ==500:
        print(f"Server error:{resp.status_code}")
        return
    state = json.loads(resp.content)
    print(state['state'])
    return state['state'][case]

def train_gen_with_data(
    case = "Alzheimer",
    data_path = "generative_models/transformer/docked_data_for_train/data_4j1r.csv",#path to client data folder
    feature_column=['canonical_smiles'],
    target_column=['docking_score','QED','Synthetic Accessibility','PAINS','SureChEMBL','Glaxo','Brenk','IC50'], #All propreties from dataframe you want to calculate in the end
    regression_props = ['docking_score'], #Column name with data for regression tasks (That not include in calculcateble propreties)
    classification_props = ['IC50'], #Column name with data for classification tasks (That not include in calculcateble propreties)
    description = 'Case for Brain canser',
    timeout = 5, # min
    url: str = "http://10.32.2.2:81/train_gen_models",
    fine_tune:bool = True,
    n_samples =10,
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
        # 'description' : description,
        'regression_props' : regression_props,
        #'classification_props' : classification_props,
        "fine_tune" :fine_tune,
        'n_samples':n_samples,
        **kwargs,
    }

    #Get state from server
    #state = get_state_from_server(case=case,url=url)

    #print(state)
    #print(state['calc_propreties']) 
    #Get state from server
    resp = requests.post(url,json.dumps(params))
    print(resp)


    # p = Process(target=requests.post,args=[url,json.dumps(params)])
    # p.start()
    # time.sleep(4)


    #p.terminate()
    #resp = requests.post(url, data=json.dumps(params))
    print("--- %s seconds ---" % (time.time() - start_time))


def train_gan_gen_with_data(
    case = "Alzheimer",
    data_path = "generative_models/transformer/docked_data_for_train/data_4j1r.csv",#path to client data folder
    feature_column=['canonical_smiles'],
    target_column=['docking_score','QED','Synthetic Accessibility','PAINS','SureChEMBL','Glaxo','Brenk','IC50'], #All propreties from dataframe you want to calculate in the end
    regression_props = ['docking_score'], #Column name with data for regression tasks (That not include in calculcateble propreties)
    classification_props = ['IC50'], #Column name with data for classification tasks (That not include in calculcateble propreties)
    description = 'Case for Brain canser',
    timeout = 5, # min
    url: str = "http://10.32.2.2:81/train_gan",
    fine_tune:bool = True,
    n_samples =10,
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
        # 'description' : description,
        'regression_props' : regression_props,
        #'classification_props' : classification_props,
        "fine_tune" :fine_tune,
        'n_samples':n_samples,
        **kwargs,
    }

    #Get state from server
    #state = get_state_from_server(case=case,url=url)

    #print(state)
    #print(state['calc_propreties']) 
    #Get state from server
    #resp = requests.post(url,json.dumps(params))
    p = Process(target=requests.post,args=[url,json.dumps(params)])
    p.start()

    time.sleep(4)
    #p.terminate()
    #resp = requests.post(url, data=json.dumps(params))
    print("--- %s seconds ---" % (time.time() - start_time))


def generate_mol_by_case(case = "Alzheimer",
                         url: str = "http://10.32.2.2:81/generate_gen_models_by_case",
                         n_samples =10,
                         **kwargs):
    params = {
        'case': case,
        'n_samples':n_samples,
        **kwargs,
    }
    start_time = time.time()
    resp = requests.post(url, data=json.dumps(params))
    print("--- %s seconds ---" % (time.time() - start_time))
    return json.loads(resp.json())


def call_for_generation(
    numb_mol: int = 3,
    cuda: bool = True,
    mean_: float = 0,
    std_: float = 1,
    url: str = "http://10.32.2.2:81/case_generator",
    case_: str = 'Dslpdm',
    **kwargs,
):
    """Function that call Chem server API for generate molecules with properties by choosen case. By default it call random generation case.

    Args:
        numb_mol (int, optional): Number of moluecules that need to generate. Defaults to 1.
        cuda (bool, optional): Cuda usage mode. Defaults to True.
        mean_ (float, optional): mean of noise distibution. ONLY FOR EXPERIMENTS. Defaults to 0.0.
        std_ (float, optional): std of noise distibution. ONLY FOR EXPERIMENTS. Defaults to 1.0.
        url (_type_, optional): URL to API srver. Defaults to 'http://10.32.2.4:80/case_generator'.
        case_ (str, optional): Key for api, that define what case you choose for. Can be choose from: 'Alzhmr','Sklrz','Prkns','Cnsr','Dslpdm','TBLET', 'RNDM'.
                               Where: 'Alzhmr' - Alzheimer,
                                        'Sklrz' - Skleroz,
                                        'Prkns' - Parkinson,
                                        'Cnsr' - Canser,
                                        'Dslpdm' - Dyslipidemia,
                                        'TBLET' - Drug resistance,
                                        'RNDM' - random generation.
                                        Defaults to RNDM.
    Returns:
        Tuple[requests.models.Response, dict]: Return full respones, or just dict with molecules and properties list.
        Tuple[requests.models.Response, dict]: Return full respones, or just dict with molecules and properties list.

    Example:
        numbs = 4
        params = {'numb_mol': numbs, 'cuda': False, 'mean_': 0, case_ = 'RNDM
                'std_': 1}
        resp_mol, mols = call_for_generation(**params,hello='world')
        print(mols)
        >> {'Molecules': ['Cc1cc(C(=O)OCC(=O)NCC2CCCO2)nn1C', 'CSC1=CC=C(C(=O)O)C(C(=O)c2ccc(C(F)(F)F)cc2)S1', 'CSc1cc(-c2ccc(-c3ccccc3)cc2)nc(C(C)=O)c1O', 'CC(C)N(CC(=O)NCc1cn[nH]c1)Cc1ccccc1'],
          'Docking score': [-6.707, -7.517, -8.541, -7.47],
            'QED': [0.7785404162969669, 0.8150693008303525, 0.5355361484098266, 0.8174264075095671],
              'SA': [2.731063371805302, 3.558887012627684, 2.2174895913203354, 2.2083851588937087],
                'PAINS': [0, 0, 0, 0],
                  'SureChEMBL': [0, 0, 0, 0],
                    'Glaxo': [0, 0, 0, 0]}
    """
    start_time = time.time()
    
    params = {
        "numb_mol": numb_mol,
        "cuda": cuda,
        "mean_": mean_,
        "std_": std_,
        'case_' : case_,
        **kwargs,
    }
    resp = requests.post(url, data=json.dumps(params))
    print("--- %s seconds ---" % (time.time() - start_time))
    return resp, json.loads(resp.json())


def call_for_gan_generation(
    numb_mol: int = 3,
    cuda: bool = True,
    mean_: float = 0,
    std_: float = 1,
    url: str = "http://10.32.2.2:94/gan_case_generator",
    case_: str = 'Dslpdm',
    **kwargs,
):
    """Function that call Chem server API for generate molecules with properties by choosen case. By default it call random generation case.

    Args:
        numb_mol (int, optional): Number of moluecules that need to generate. Defaults to 1.
        cuda (bool, optional): Cuda usage mode. Defaults to True.
        mean_ (float, optional): mean of noise distibution. ONLY FOR EXPERIMENTS. Defaults to 0.0.
        std_ (float, optional): std of noise distibution. ONLY FOR EXPERIMENTS. Defaults to 1.0.
        url (_type_, optional): URL to API srver. Defaults to 'http://10.32.2.4:80/case_generator'.
        case_ (str, optional): Key for api, that define what case you choose for. Can be choose from: 'Alzhmr','Sklrz','Prkns','Cnsr','Dslpdm','TBLET', 'RNDM'.
                               Where: 'Alzhmr' - Alzheimer,
                                        'Sklrz' - Skleroz,
                                        'Prkns' - Parkinson,
                                        'Cnsr' - Canser,
                                        'Dslpdm' - Dyslipidemia,
                                        'TBLET' - Drug resistance,
                                        'RNDM' - random generation.
                                        Defaults to RNDM.
    Returns:
        Tuple[requests.models.Response, dict]: Return full respones, or just dict with molecules and properties list.
        Tuple[requests.models.Response, dict]: Return full respones, or just dict with molecules and properties list.

    Example:
        numbs = 4
        params = {'numb_mol': numbs, 'cuda': False, 'mean_': 0, case_ = 'RNDM
                'std_': 1}
        resp_mol, mols = call_for_generation(**params,hello='world')
        print(mols)
        >> {'Molecules': ['Cc1cc(C(=O)OCC(=O)NCC2CCCO2)nn1C', 'CSC1=CC=C(C(=O)O)C(C(=O)c2ccc(C(F)(F)F)cc2)S1', 'CSc1cc(-c2ccc(-c3ccccc3)cc2)nc(C(C)=O)c1O', 'CC(C)N(CC(=O)NCc1cn[nH]c1)Cc1ccccc1'],
          'Docking score': [-6.707, -7.517, -8.541, -7.47],
            'QED': [0.7785404162969669, 0.8150693008303525, 0.5355361484098266, 0.8174264075095671],
              'SA': [2.731063371805302, 3.558887012627684, 2.2174895913203354, 2.2083851588937087],
                'PAINS': [0, 0, 0, 0],
                  'SureChEMBL': [0, 0, 0, 0],
                    'Glaxo': [0, 0, 0, 0]}
    """
    start_time = time.time()
    
    params = {
        "numb_mol": numb_mol,
        "cuda": cuda,
        "mean_": mean_,
        "std_": std_,
        'case_' : case_,
        **kwargs,
    }
    resp = requests.post(url, data=json.dumps(params))
    print("--- %s seconds ---" % (time.time() - start_time))
    print(resp)
    print(json.loads(resp.json()))
    return resp, json.loads(resp.json())
#ret = call_for_generation()


if __name__=='__main__':
    # train_gen_with_data(case = "QED",
    #                     url = "http://10.32.2.2:81/train_gen_models",
    #                     target_column=['QED'],
    #                      regression_props = ['QED'], #All propreties from dataframe you want to calculate in the end
    #                     fine_tune=True,
    #                     data_path='/projects/generative_models_data/generative_models/transformer/docked_data_for_train/data_cyk_short.csv',
    #                     epochs=3)
    # train_gen_with_data(case = "QED_4version",
    #                     url = "http://10.32.2.2:81/train_gen_models",
    #                     target_column=["QED"],
    #                      regression_props = ["QED"],
    #                     #classification_props = ['IC50'], #All propreties from dataframe you want to calculate in the end
    #                     fine_tune=True,
    #                     data_path='/projects/generative_models_data/generative_models/transformer/docked_data_for_train/data_cyk_short.csv',
    #                     epochs=2)


    # train_gan_gen_with_data(case = "Alzheimer_regression_Minimum_Energy_Vitctor",
    #                     url = "http://10.32.2.2:293/train_gan",
    #                     feature_column=["Smiles"],
    #                     #classification_props = ['IC50'], #All propreties from dataframe you want to calculate in the end
    #                     fine_tune=True,
    #                     data_path='infrastructure/automl/data/base_cases/docked_all_kinase_inhib.csv',
    #                     epochs=2)
    #call_for_gan_generation(numb_mol=100,case_='Alzheimer_regression_Minimum_Energy_Vitctor',url = "http://10.32.2.2:293/gan_case_generator")
    print(call_for_generation(url="http://10.32.2.2:293/case_generator",case_='RNDM'))#'Alzhmr','Sklrz','Prkns','Cnsr','Dslpdm','TBLET', 'RNDM'
    
    #print(get_state_from_server(url = "http://10.32.2.2:193",case = "Alzheimer_regression"))


    # print(generate_mol_by_case(case = "QED_4version",
    #                     url = "http://10.32.2.2:81/generate_gen_models_by_case",
    #                     n_samples=5))