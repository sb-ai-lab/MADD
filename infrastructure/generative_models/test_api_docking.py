import requests
import json
import socket
import time 

def call_for_docking(
    mol_list: list,
    case_: str = 'Dslpdm',
    url:str =' http://10.32.2.4:81/eval_docking',
    **kwargs,
):
    """Function that call Chem server API for generate molecules with properties by choosen case. By default it call random generation case.

    Args:
        
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
        "mol_list": ['CCCCCCCCCCCCCCCCCC(=O)N[C@@H](CO)[C@H](O)CCCCCCCCCCCCCCC'],
        'case_' : case_,
        **kwargs,
    }
    resp = requests.post(url, data=json.dumps(params))
    print("--- %s seconds ---" % (time.time() - start_time))
    return resp, json.loads(resp.json())

print(call_for_docking)