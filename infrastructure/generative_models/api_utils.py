from typing import List 
from fastapi import Body
import os
import sys
from pydantic import BaseModel
from inference import predict_smiles
from autotrain.utils.base_state import TrainState
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(import_path)
from GAN.gan_lstm_refactoring.gen import generate
from generate.config import parsing
from train_data.utils.config_generate import configurate_parser
from train_data.generate import generator_for_agent_multi_prop as multi_generator
import pandas as pd
from utils.validation import check_chem_valid, eval_P_S_G, eval_qed, eval_sa, check_brenk
import lightgbm
from utils.ic_50_models.alzheimer.predict_ic50_clf import eval_ic_50_alzheimer
from utils.ic_50_models.skleroz_ic50_clf.scripts.predict_ic50_btk_clf import eval_ic_50_sklrz
from utils.ic_50_models.kras_ic50_prediction.predict_ic50_clf import eval_ic_50_cancer
from utils.ic_50_models.citrate_classif_inference.inference_citrate_clf import predict as parkenson_predict_ic50
from utils.ic_50_models.drug_resis_classif_inference.inference_drug_clf import predict as drug_res_predict_ic50
from utils.ic_50_models.tyrosine_classif_inference.inference_tyrosine_clf import predict as dyslip_predict_ic50
from utils.ki_models.tyrosine_regression_inference.tyrosine_inference_regr import predict as dyslip_predict_ki
from utils.inference_BB_clf.BB_inference import predict as eval_bbb
from autotrain.auto_train import main, main_generate
###Docking
from autodock_vina_python3.src.docking_score import docking_list
from utils.check_novelty import check_novelty_chembl
import pickle
from GAN.gan_lstm_refactoring.train_gan import auto_train

class GenData(BaseModel):
        numb_mol: int =1
        model:str = None
        cuda:bool=False
        mean_:float=0
        std_:float=1
        case_ : str = 'RNMD'
        url:str = "http://10.64.4.247:81"

class TrainData(BaseModel):
        data:dict = None
        case:str = None
        data_path:str = None
        target_column:list = None
        #smiles_list: list = None
        timeout:int = 30 #30 min
        feature_column:list = ['Smiles']
        path_to_save:str = 'automl/trained_data'
        description:str = 'Unknown case.'
        url:str = "http://10.64.4.247:81"
        n_samples:int = 1000
        fine_tune:bool = True
        new_vocab:bool = False
        epochs:int = 10
        batchsize:int = 2048
        # regression_props:list= None
        # classification_props:list = None

class Molecules(BaseModel):
    mol_list:List[str]

class Docking_config(BaseModel):
    mol_list:List[str]
    receptor_case : str = 'Alzhmr'

def condition_enchance():
     pass

def case_trainer(data:TrainData=Body()):
    # #####FOR TEST####
    # if data.numb_mol>100:
    #     data.numb_mol=100
    # ##################
    #Docking score
    is_evalute_docking = False
    """Main API func for train generative models.

    Args:
        numb_mol (int): Number of molecules to generating.
        model (str, optional): What model need to use.Choose from [lstm,CVAE,TVAE]. Defaults to 'lstm'.
        cuda (bool, optional): Choose cuda usage option. Defaults to False.
        case_ (str): Choose what disease u want to generate molecules for. 

    Returns:
        _type_: _description_
    """
    state = TrainState(state_path='infrastructure/generative_models/autotrain/utils/state.json')
    try:
        if data.data is not None:
                    df = pd.DataFrame(data.data)
                    data.data_path = f"autotrain/data/{data.case}"
                    if not os.path.isdir(data.data_path):
                        os.mkdir(data.data_path)
                    data.data_path = data.data_path + '/data.csv'
                    df = df.dropna()
                    df = df[df[data.feature_column[0]].str.len()<200]
                    df.to_csv(data.data_path)
                    state.gen_model_upd_data(case=data.case,data_path=data.data_path)
        #CASE = 'CYK'
        # train_data = '/projects/generative_models_data/generative_models/docked_data_for_train/data_cyk_short.csv'
        # conditions = ['docking_score','QED','Synthetic Accessibility','PAINS','SureChEMBL','Glaxo','Brenk','IC50']
        test_mode = False
        
    
        if data.fine_tune==True:
            load_weights = 'autotrain/train_Alzheimer_1_prop/Alzheimer_1_prop'
            load_weights_fields = 'autotrain/train_Alzheimer_1_prop/Alzheimer_1_prop'
            data.new_vocab = False
        else:
            load_weights=None
            load_weights_fields = None
            data.new_vocab = True
        # if state(CASE) is None:#Check if case exist
        #     state.add_new_case(CASE,rewrite=False)
        
        if state(data.case,'ml') is None:
            print(f"{data.case} is not exist! Train ML model before")
            state.gen_model_upd_status(case=data.case,status=3)
            return 0

        use_cond2dec = False
        main(epochs=data.epochs,
            conditions = state(data.case,'ml')['target_column'],
            case=data.case, 
            server_dir = f'autotrain/train_{data.case}',
            data_path_with_conds = data.data_path,
            test_mode=test_mode,
            state=state,
            url=data.url,
            n_samples = data.n_samples,
            load_weights=load_weights,
            load_weights_fields = load_weights_fields,
            use_cond2dec=use_cond2dec,
            new_vocab= data.new_vocab,
            ml_model_url=os.getenv('ML_MODEL_URL'))
    except Exception as e:
        print(e)
        state.gen_model_upd_status(case=data.case,error=str(e))

def gan_case_trainer(data:TrainData=Body()):
    # #####FOR TEST####
    # if data.numb_mol>100:
    #     data.numb_mol=100
    # ##################
    #Docking score
    is_evalute_docking = False
    """Main API func for train generative models.

    Args:
        numb_mol (int): Number of molecules to generating.
        model (str, optional): What model need to use.Choose from [lstm,CVAE,TVAE]. Defaults to 'lstm'.
        cuda (bool, optional): Choose cuda usage option. Defaults to False.
        case_ (str): Choose what disease u want to generate molecules for. 

    Returns:
        _type_: _description_
    """
    state = TrainState(state_path='infrastructure/generative_models/autotrain/utils/state.json')
    try:
        if data.data is not None:
                    df = pd.DataFrame(data.data)
                    data.data_path = f"infrastructure/generative_models/autotrain/data/{data.case}"
                    if not os.path.isdir(data.data_path):
                        os.mkdir(data.data_path)
                    data.data_path = data.data_path + '/data.csv'
                    df = df.dropna()
                    #df = df[df[data.feature_column[0]].str.len()<200]
                    df.to_csv(data.data_path)
                    state.gen_model_upd_data(case=data.case,data_path=data.data_path,feature_column=data.feature_column)
        #CASE = 'CYK'
        # train_data = '/projects/generative_models_data/generative_models/docked_data_for_train/data_cyk_short.csv'
        # conditions = ['docking_score','QED','Synthetic Accessibility','PAINS','SureChEMBL','Glaxo','Brenk','IC50']
        test_mode = False
        
    
        # if data.fine_tune==True:
        #     load_weights = '/projects/CoScientist/ChemCoScientist/generative_models/GAN/gan_lstm_refactoring/weights/v4_gan_mol_124_0.0003_8k.pkl'
        #     load_weights_fields = '/projects/CoScientist/ChemCoScientist/generative_models/GAN/gan_lstm_refactoring/weights/v4_gan_mol_124_0.0003_8k.pkl'
        #     data.new_vocab = False
        # else:
        #     load_weights=None
        #     load_weights_fields = None
        #     data.new_vocab = True
        # if state(CASE) is None:#Check if case exist
        #     state.add_new_case(CASE,rewrite=False)
        
        if state(data.case,'ml') is None:
            print(f"{data.case} is not exist! Train ML model before")
            state.gen_model_upd_status(case=data.case,status=3)
            return 0

        auto_train(data.case,
                   path_ds=data.data_path,
                   fine_tune=data.fine_tune,
                   state=state,
                   feature_column=data.feature_column,
                   steps=data.epochs)

    except Exception as e:
        print(e)
        state.gen_model_upd_status(case=data.case,error=str(e))

def gan_auto_generator(data:GenData=Body()):
    print(sys.path)
    state = TrainState(state_path='infrastructure/generative_models/autotrain/utils/state.json')

    with open(state(data.case_,'gen')['weights_path']+'/gan_weights.pkl', "rb") as f:
        gan_mol = pickle.load(f)
    gan_mol.eval()
    samples = gan_mol.generate_n(data.numb_mol)
    valid_mols = state()["Calculateble properties"]['Validity'](samples)
    unique_mols = set(valid_mols)
    DYPLICATES = 1-len(unique_mols)/len(valid_mols)
    VALID = len(valid_mols)/len(samples)
    props_for_calc = [i for i in state.show_calculateble_propreties() if i !="Validity"]
    print(props_for_calc)
    props = {key:state()["Calculateble properties"][key](valid_mols) for key in props_for_calc}
    props['Validity'] = VALID
    props["Duplicates"] = DYPLICATES
    print(props)
    if state(data.case_,'ml')['status'] == 'Trained':
        ml_props = predict_smiles(valid_mols,data.case_,url=os.getenv('ML_MODEL_URL'))
        for key,value in ml_props.items():
            props[key]=value 
    df = pd.DataFrame(data = {'Smiles':valid_mols,**props})
    return df.to_dict()
    #return samples


def auto_generator(data:TrainData=Body()):
     
    state = TrainState(state_path='infrastructure/generative_models/autotrain/utils/state.json')
    if state(data.case,'gen')["status"] == "Trained":
        use_cond2dec = False
        gen_dict = main_generate(epochs=data.epochs,
                conditions = state(data.case,'ml')['target_column'],
                case=data.case, 
                server_dir = f'autotrain/train_{data.case}',
                test_mode=False,
                state=state,
                url=data.url,
                n_samples = data.n_samples,
                load_weights=state(data.case,'gen')['weights_path'],
                load_weights_fields = state(data.case,'gen')['weights_path'],
                use_cond2dec=use_cond2dec,
                new_vocab= data.new_vocab,
                batchsize=data.batchsize)
        return gen_dict
    else:
        print('Case is not trained!')
        return 0

def case_generator(data:GenData=Body()):
    #####FOR TEST####
    if data.numb_mol>100:
        data.numb_mol=100
    ##################
    #Docking score
    is_evalute_docking = False
    """Main API func for generation any chem case.

    Args:
        numb_mol (int): Number of molecules to generating.
        model (str, optional): What model need to use.Choose from [lstm,CVAE,TVAE]. Defaults to 'lstm'.
        cuda (bool, optional): Choose cuda usage option. Defaults to False.
        case_ (str): Choose what disease u want to generate molecules for. 

    Returns:
        _type_: _description_
    """
    dis_case = cases[data.case_]
    print(data)
    init_mol_numb = data.numb_mol
    #data = data
    mol_list = []
    mol_list = _gen_n(data)
    mol_list = [i for i in mol_list if '.' not in i]
    mol_list = check_chem_valid(mol_list)
    mol_list,diversity = check_novelty_chembl(mol_list,train_data_path=dis_case['train_data_path'])
    
    #Check if generated not enouth molecules 
    while len(mol_list)<init_mol_numb:

        #init_mol_numb = data.numb_mol-len(mol_list)
        data.numb_mol = init_mol_numb-len(mol_list)
        mol_list_temp = _gen_n(data)
        mol_list = [i for i in mol_list if '.' not in i]
        mol_list += check_chem_valid(mol_list_temp)
        mol_list,diversity = check_novelty_chembl(mol_list,train_data_path=dis_case['train_data_path'])
    
    path = dis_case['docking_path'] #Choose path to docking file for current case

    #Calculate metrics
    df = pd.DataFrame(data=mol_list,columns=['Molecules'])

    #Docking Score
    if is_evalute_docking:
        d_s = docking_list(smiles=mol_list,path_receptor_pdb=path)
        df['Docking score'] = d_s

    if dis_case['anti_docking_path'] is not None:
         for i in dis_case['anti_docking_path'].keys():
              anti_path = dis_case['anti_docking_path'][i]
              d_s_anti_target = docking_list(smiles=mol_list,path_receptor_pdb=anti_path)
              col_name = f'Anti Docking score for {i}'
              df[col_name] = d_s_anti_target

    df['QED'] = df['Molecules'].apply(eval_qed)
    df['Synthetic Accessibility'] = df['Molecules'].apply(eval_sa) # SA
    df['PAINS'] = df['Molecules'].apply(eval_P_S_G,type_n='PAINS')
    df['SureChEMBL']= df['Molecules'].apply(eval_P_S_G,type_n='SureChEMBL')
    df['Glaxo'] = df['Molecules'].apply(eval_P_S_G,type_n='Glaxo')
    #df['Diversity'] = diversity
    df['Brenk'] = df['Molecules'].apply(check_brenk)
    df['BBB'] = eval_bbb(list(df['Molecules']))

    #Choose property IC50 OR KI for cases
    if data.case_ == 'Dslpdm':
        ki_values = dis_case['KI'](mol_list)
        df['KI'] = ki_values
    if dis_case['IC50'] is not None:
        ic50_ki_values = dis_case['IC50'](mol_list)
        df['IC50'] = ic50_ki_values
    df = df.round(2)
    return {i:df[i].to_list() for i in df.columns}
    

def _gen_n(data):
        """Function for generating molecules for choosen case
        """
       
        mol_list = []
        dis_case = cases[data.case_]
        if data.case_ == 'Alzhmr':
            df = dis_case['generative_model'](opt=dis_case['opt'],
                n_samples=data.numb_mol,
                path_to_save='',
                cuda=data.cuda,
                save=False,
                spec_conds = [[-9,-12],[0.8,1],[0,2.99],[0,0],[0,0],[0,0],[0,0],[0,0]],
                mean_=data.mean_,std_=data.std_)
            mol_list += df
        elif data.case_ == 'TBLET':
            df = dis_case['generative_model'](opt=dis_case['opt'],
                n_samples=data.numb_mol,
                path_to_save='',
                cuda=data.cuda,
                save=False,
                spec_conds = [[-7,-11],[0.7,1],[0,2.99],[0,0],[0,0],[0,0],[0,0],[1,1]],
                mean_=data.mean_,std_=data.std_)
            mol_list += df
        elif data.case_ == 'RNDM':
             df = dis_case['generative_model'](data.numb_mol)
             mol_list += df
        else:
            df = dis_case['generative_model'](opt=dis_case['opt'],
                n_samples=data.numb_mol,
                path_to_save='',
                cuda=data.cuda,
                save=False,
                spec_conds = [[-7,-12],[0.6,1],[0,2.99],[0,0],[0,0],[0,0],[0,0],[1,1]],
                mean_=data.mean_,std_=data.std_)
            mol_list += df
        return mol_list

docking_paths = {'Alzhmr' : 'generative_models/autodock_vina_python3/data/4j1r.pdb',
                 'Sklrz':'generative_models/autodock_vina_python3/data/target_BTK.pdb'}

opt = parsing()
parser = configurate_parser(load_weights="infrastructure/generative_models/autotrain/many_prop_CVAE/weights_8p_alzhmr",#weights_33k_trained
                            load_weights_fields = "infrastructure/generative_models/autotrain/many_prop_CVAE/weights_8p_alzhmr",
                            cuda=False,
                            save_folder_name='alzh_gen_mols',
                            new_vocab = False,
                            import_path = import_path,
                            cond_dim=8
                                )
opt_Alz_multi = parser.parse_args()
opt_sklrz = parser.parse_args()
opt_sklrz.load_weights = "infrastructure/generative_models/autotrain/many_prop_CVAE/weights_8p_sklrz"
opt_sklrz.load_weights_fields = "infrastructure/generative_models/autotrain/many_prop_CVAE/weights_8p_sklrz"

opt_cnsr = parser.parse_args()
opt_cnsr.load_weights = 'infrastructure/generative_models/autotrain/many_prop_CVAE/weights_8p_cnsr'
opt_cnsr.load_weights_fields = 'infrastructure/generative_models/autotrain/many_prop_CVAE/weights_8p_cnsr'


opt_tablet = parser.parse_args()
opt_tablet.load_weights = 'infrastructure/generative_models/autotrain/many_prop_CVAE/weights_8p_tablet'
opt_tablet.load_weights_fields = 'infrastructure/generative_models/autotrain/many_prop_CVAE/weights_8p_tablet'

opt_park = parser.parse_args()
opt_park.load_weights = 'infrastructure/generative_models/autotrain/many_prop_CVAE/weights_parkinson'
opt_park.load_weights_fields = 'infrastructure/generative_models/autotrain/many_prop_CVAE/weights_parkinson'

opt_dislip = parser.parse_args()
opt_dislip.load_weights = 'infrastructure/generative_models/autotrain/many_prop_CVAE/weights_dislip'
opt_dislip.load_weights_fields = 'infrastructure/generative_models/autotrain/many_prop_CVAE/weights_dislip'

# Case information
cases = {'Alzhmr' : 
         {'docking_path' : 'infrastructure/generative_models/autodock_vina_python3/data/4j1r.pdb',
        'generative_model':multi_generator,
        'opt':opt_Alz_multi,
        'IC50':eval_ic_50_alzheimer,
         'KI':None,
         'anti_docking_path':None,
         'train_data_path':'infrastructure/generative_models/docked_data_for_train/data_4j1r.csv'
                     },
        #TODO update next cases
        'Sklrz':
        {'docking_path' :'infrastructure/generative_models/autodock_vina_python3/data/skleroz/target_BTK.pdb',
         'generative_model':multi_generator,
         'opt':opt_sklrz,
         'IC50':eval_ic_50_sklrz,
         'KI':None,
         'anti_docking_path':None,
         'train_data_path':'infrastructure/generative_models/docked_data_for_train/data_5vfi.csv'#{'BMX':'generative_models/autodock_vina_python3/data/skleroz/BMX_8x2a_protein.pdb'}
         },

         'Prkns':
        {'docking_path' :'infrastructure/generative_models/autodock_vina_python3/data/parkinson/tyrosine_protein_kinase_ABL.pdb',
         'generative_model':multi_generator,
         'opt':opt_park,
         'IC50':parkenson_predict_ic50,
         'KI':None,
         'anti_docking_path':None,
         'train_data_path':'infrastructure/generative_models/docked_data_for_train/data_ABL.csv'
         },

         'Cnsr':
        {'docking_path' :'infrastructure/generative_models/autodock_vina_python3/data/Canser/8afb_protein.pdb',
         'generative_model':multi_generator,
         'opt':opt_cnsr,
         'IC50':eval_ic_50_cancer,
         'KI':None,
         'anti_docking_path':None,
         'train_data_path':'infrastructure/generative_models/docked_data_for_train/data_8afb.csv'#{'NRAS':'generative_models/autodock_vina_python3/data/Canser/NRAS_3con_protein.pdb',
                              #'HRAS':'generative_models/autodock_vina_python3/data/Canser/HRAS_3k8y_protein.pdb'}
         },

         'Dslpdm':
        {'docking_path' :'infrastructure/generative_models/autodock_vina_python3/data/dislipidemia/ATP_citrate_synthase.pdb',
         'generative_model':multi_generator,
         'opt':opt_dislip,
         'IC50':dyslip_predict_ic50,
         'KI':dyslip_predict_ki,
         'anti_docking_path':None,
         'train_data_path':'infrastructure/generative_models/docked_data_for_train/data_ATP.csv'
         },

         'TBLET':
        {'docking_path' :'infrastructure/generative_models/autodock_vina_python3/data/Signal_Transducer_and_Activator_of_Transcription_3.pdb',
         'generative_model':multi_generator,
         'opt':opt_tablet,
         'IC50':drug_res_predict_ic50,
         'KI':None,
         'anti_docking_path':None,
         'train_data_path':'infrastructure/generative_models/docked_data_for_train/data_stat3.csv'
         },
         'RNDM':
         {'docking_path' :'infrastructure/generative_models/autodock_vina_python3/data/Signal_Transducer_and_Activator_of_Transcription_3.pdb',
         'generative_model':generate,
         'opt':opt,
         'IC50':None,
         'KI':None,
         'anti_docking_path':None,
         'train_data_path':'infrastructure/generative_models/docked_data_for_train/data_stat3.csv'
         },
        }

