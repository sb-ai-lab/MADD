import random
import sys
import os
import_path = os.getcwd()
# sys.path.append(import_path)
from inference import generate_auto
from train_data.utils.auto_train_loop import train_model,train_model_auto
from train_data.utils.config import configurate_parser
from Models import get_model
from Process import *
from Optim import CosineWithRestarts
import dill as pickle
import pandas as pd
from ic50_classifire_model.read_ic50 import Ic50
import warnings
from typing import List
warnings.filterwarnings('ignore')
from autotrain.utils.base_state import TrainState
from autotrain.utils.config import Config


def main(server_dir = 'generative_models/train_dislip',
         conditions : List[str] = ['ic50'],
         case:str = 'Alzhmr',
         data_path_with_conds = 'generative_models/docked_data_for_train/data_5vfi.csv',
         test_mode = False,
         state = None,
         ml_model_url= 'http://10.64.4.247:81',
         *args,
         **kwargs):
    """Main function to start training.

    Args:
        server_dir (str, optional): Path to save weights. Defaults to 'generative_models/transformer/train_cVAE_transformer_altz'.
        conditions (List[str], optional): Names of condition values, that match with dataset conditions columns names. Defaults to ['ic50'].
    """
    if not os.path.isdir(server_dir):
        os.mkdir(server_dir)
    if not os.path.isdir(server_dir+f'/{case}'):
        os.mkdir(server_dir+f'/{case}')

    opt = Config(import_path = import_path,
                 data_path_with_conds=data_path_with_conds,
                                save_folder_name = server_dir+f'/{case}',
                                batchsize = 2048,
                                dropout = 0.1,
                                cuda = True,
                                cond_dim = len(conditions),
                                
                                **kwargs
                                )
    opt.conditions = conditions
    state.gen_model_upd_status(case=case,model_weight_path=server_dir+f'/{case}')
    opt.url_ml_model = ml_model_url
    opt.feature_col = state(case,'ml')['feature_column']
    
    opt.device = 'cuda' if opt.cuda else 'cpu'
    if opt.historyevery % opt.printevery != 0:
        raise ValueError("historyevery must be a multiple of printevery: {} % {} != 0".format(opt.historyevery, opt.printevery))

    if opt.device == 'cuda':
        print('Cuda is using...\t','cuda is_available:', torch.cuda.is_available())
        assert torch.cuda.is_available()

    read_data(opt,state(case,'ml')['feature_column'])
    PROP = pd.read_csv(opt.cond_train_path)[opt.conditions]
    opt.min_prop = PROP.min().to_dict()
    opt.max_prop = PROP.max().to_dict()
    SRC, TRG = create_fields(opt)

    if not os.path.isdir(opt.save_folder_name+f'/weights'):
            os.mkdir(opt.save_folder_name+f'/weights')

    opt.train = create_dataset_auto(opt, SRC, TRG, PROP, tr_te='tr',test_mode=test_mode)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("# of trainable parameters: {}".format(total_trainable_params))

    opt.optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(opt.lr_beta1, opt.lr_beta2), eps=opt.lr_eps)
    if opt.lr_scheduler == "SGDR":
        if opt.train_len==0:
            opt.sched = CosineWithRestarts(opt.optimizer, T_max=1)
        else:
            opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)
    opt.TRG = TRG
    opt.SRC = SRC
    opt.state = state
    train_model_auto(model,opt,case=case,state=state)



def main_generate(server_dir = 'generative_models/transformer/train_dislip',
         conditions : List[str] = ['ic50'],
         case:str = 'Alzhmr',
         test_mode = False,
         state = None,
         ml_model_url = 'http://10.64.4.247:81',
         *args,
         **kwargs):
    """Main function to start training.

    Args:
        server_dir (str, optional): Path to save weights. Defaults to 'generative_models/transformer/train_cVAE_transformer_altz'.
        conditions (List[str], optional): Names of condition values, that match with dataset conditions columns names. Defaults to ['ic50'].
    """
    if not os.path.isdir(server_dir):
        os.mkdir(server_dir)
    if not os.path.isdir(server_dir+f'/{case}'):
        os.mkdir(server_dir+f'/{case}')

    opt = Config(import_path = import_path,
                 data_path_with_conds=state(case,'gen')['data_path'],
                                save_folder_name = server_dir+f'/{case}',
                                dropout = 0.1,
                                cuda = True,
                                cond_dim = len(conditions),
                                
                                **kwargs
                                )
    opt.conditions = conditions
    state.gen_model_upd_status(case=case,model_weight_path=server_dir+f'/{case}')
    opt.url_ml_model = ml_model_url
    opt.feature_col = state(case,'ml')['feature_column']
    
    opt.device = 'cuda' if opt.cuda else 'cpu'
    if opt.historyevery % opt.printevery != 0:
        raise ValueError("historyevery must be a multiple of printevery: {} % {} != 0".format(opt.historyevery, opt.printevery))

    if opt.device == 'cuda':
        print('Cuda is using...\t','cuda is_available:', torch.cuda.is_available())
        assert torch.cuda.is_available()

    read_data(opt,state(case,'ml')['feature_column'])
    PROP = pd.read_csv(opt.cond_train_path)[opt.conditions]
    opt.min_prop = PROP.min().to_dict()
    opt.max_prop = PROP.max().to_dict()
    SRC, TRG = create_fields(opt)

    if not os.path.isdir(opt.save_folder_name+f'/weights'):
            os.mkdir(opt.save_folder_name+f'/weights')

    opt.train = create_dataset_auto(opt, SRC, TRG, PROP, tr_te='tr',test_mode=test_mode)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("# of trainable parameters: {}".format(total_trainable_params))

    opt.optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(opt.lr_beta1, opt.lr_beta2), eps=opt.lr_eps)
    if opt.lr_scheduler == "SGDR":
        if opt.train_len==0:
            opt.sched = CosineWithRestarts(opt.optimizer, T_max=1)
        else:
            opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)
    opt.TRG = TRG
    opt.SRC = SRC
    opt.state = state
    spec_conds = [np.random.uniform(opt.min_prop[key]+(opt.max_prop[key]-opt.min_prop[key])/2,opt.max_prop[key]) for key in opt.min_prop.keys()]
    df = generate_auto(
            opt=opt,
            model=model,
            SRC=opt.SRC,
            TRG=opt.TRG,
            n_samples=opt.n_samples,
            spec_conds=spec_conds,
            case=case,
            state=state)
    print(df)
    #df.to_csv('generated_alzheimer_2_2_second_epoch.csv')
    return df

# def train_ml_with_data(data:MLData=Body()):
#         state = TrainState()
#         state.add_new_case(case_name=data.case,
#                            rewrite=True,
#                            description=data.description)
#         if data.data is not None:
#                 df = pd.DataFrame(data.data)
#                 data.data_path = f"automl/data/{data.case}"
#                 if not os.path.isdir(data.data_path):
#                     os.mkdir(data.data_path)
#                 data.data_path = data.data_path + '/data.csv'
#                 df.to_csv(data.data_path) 
                      
#         state.ml_model_upd_data(case=data.case,
#                                 data_path=data.data_path,
#                                 feature_column=data.feature_column,
#                                 target_column=data.target_column,
#                                  predictable_properties={"regression":data.regression_props, "classification":data.classification_props})
#         run_train_automl(case=data.case,
#                          path_to_save=data.path_to_save,
#                          timeout=data.timeout)
        
if __name__ == "__main__":
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id="SoloWayG/Molecule_transformer",
                         filename="state.json",
                         local_dir='autotrain/utils',
                         force_download=True)
    state = TrainState(state_path='autotrain/utils/state.json')
    CASE = 'MEK1'
    train_data = '/projects/CoScientist/ChemCoScientist/generative_models/autotrain/data/data_mek.csv'
    conditions = ['IC50']
    test_mode = False
    url = "http://10.64.4.247:81"
    n_samples = 10
    load_weights = '/projects/CoScientist/ChemCoScientist/generative_models/autotrain/train_Alzheimer_1_prop/Alzheimer_1_prop/weights/epo6'#'/projects/generative_models_data/generative_models/autotrain/train_ALZHEIMER_2/ALZHEIMER/weights/epo2'
    load_weights_fields = '/projects/CoScientist/ChemCoScientist/generative_models/autotrain/train_Alzheimer_1_prop/Alzheimer_1_prop/weights/epo6'#'/projects/generative_models_data/generative_models/autotrain/train_ALZHEIMER_2/ALZHEIMER/weights/epo2'
    # if state(CASE) is None:#Check if case exist
    #     state.add_new_case(CASE,rewrite=False)
    use_cond2dec = False
    new_vocab= False
    main(conditions = state(CASE,'ml')['target_column'],
         case=CASE, 
         server_dir = f'/projects/CoScientist/ChemCoScientist/generative_models/autotrain/train_{CASE}',
         data_path_with_conds = train_data,
         test_mode=test_mode,
         state=state,
         url=url,
         n_samples = n_samples,
         load_weights=load_weights,
         load_weights_fields = load_weights_fields,
         use_cond2dec=use_cond2dec,
        new_vocab= new_vocab)

