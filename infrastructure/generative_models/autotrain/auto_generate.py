import random
import sys
import os
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(import_path)
from generative_models.inference import generate_auto
from generative_models.train_data.utils.auto_train_loop import train_model,train_model_auto
from generative_models.train_data.utils.config import configurate_parser
from generative_models.Models import get_model
from generative_models.Process import *
from generative_models.Optim import CosineWithRestarts
import dill as pickle
import pandas as pd
from generative_models.ic50_classifire_model.read_ic50 import Ic50
import warnings
from typing import List
warnings.filterwarnings('ignore')
from utils.base_state import TrainState
from utils.config import Config


def main(server_dir = 'generative_models/train_dislip',
         conditions : List[str] = ['ic50'],
         case:str = 'Alzhmr',
         data_path_with_conds = 'generative_models/docked_data_for_train/data_5vfi.csv',
         test_mode = False,
         state = None,
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
                                new_vocab= False,
                                **kwargs
                                )
    opt.conditions = conditions
    state.gen_model_upd_status(case=case,model_weight_path=server_dir+f'/{case}')
    opt.url_ml_model = url
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

    #opt.train = create_dataset_auto(opt, SRC, TRG, PROP, tr_te='tr',test_mode=test_mode)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("# of trainable parameters: {}".format(total_trainable_params))

    # opt.optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(opt.lr_beta1, opt.lr_beta2), eps=opt.lr_eps)
    # if opt.lr_scheduler == "SGDR":
    #     opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)
    opt.TRG = TRG
    opt.SRC = SRC
    opt.state = state
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
    df.to_csv('generated_alzheimer_2_2_second_epoch.csv')


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
    state = TrainState(state_path='generative_models/autotrain/utils/state.json')
    CASE = 'CYK'
    train_data = 'generative_models/docked_data_for_train/data_4j1r.csv'
    conditions = ['docking_score','QED','Synthetic Accessibility','PAINS','SureChEMBL','Glaxo','Brenk','IC50']
    test_mode = False
    url = "http://10.64.4.247:81"
    n_samples = 10000
    load_weights = '/projects/generative_models_data/generative_models/autotrain/train_ALZHEIMER_2/ALZHEIMER/weights/epo2'
    load_weights_fields = '/projects/generative_models_data/generative_models/autotrain/train_ALZHEIMER_2/ALZHEIMER/weights/epo2'
    # if state(CASE) is None:#Check if case exist
    #     state.add_new_case(CASE,rewrite=False)
    spec_conds = [[-8,-12],[0.8,1],[0,2.99],[0,0],[0,0],[0,0],[0,0],[1,1]]
    spec_conds = np.array([[random.uniform(n[0],n[1]) for n in spec_conds] for _ in range(n_samples)])
    use_cond2dec = False
    main(conditions = state(CASE,'ml')['target_column'],
         case=CASE, 
         server_dir = f'generative_models/autotrain/train_{CASE}',
         data_path_with_conds = train_data,
         test_mode=test_mode,
         state=state,
         url=url,
         n_samples = n_samples,
         spec_conds=spec_conds,
         load_weights=load_weights,
         load_weights_fields = load_weights_fields,
         use_cond2dec=use_cond2dec)

