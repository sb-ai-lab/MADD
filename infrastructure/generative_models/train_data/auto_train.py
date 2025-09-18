import sys
import os
import_path = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(import_path)
from train_data.utils.auto_train_loop import train_model
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
from infrastructure.generative_models.autotrain.utils.base_state import TrainState
def main(server_dir = 'generative_models/transformer/train_dislip',
         conditions : List[str] = ['ic50'],
         case:str = 'Alzhmr',
         data_path_with_conds = 'generative_models/transformer/docked_data_for_train/data_5vfi.csv',
         test_mode = False):
    """Main function to start training.

    Args:
        server_dir (str, optional): Path to save weights. Defaults to 'generative_models/transformer/train_cVAE_transformer_altz'.
        conditions (List[str], optional): Names of condition values, that match with dataset conditions columns names. Defaults to ['ic50'].
    """

    parser = configurate_parser(import_path = import_path,
                                save_folder_name = f'{server_dir}',
                                batchsize = 2048,
                                dropout = 0.1,
                                cuda = True,
                                cond_dim = len(conditions),
                                n_samples = 100,
                                load_weights=None,
                                new_vocab= True,
                                data_path_with_conds = data_path_with_conds)
    mol_column = 'canonical_smiles'
    opt = parser.parse_args()
    opt.conditions = conditions
    opt.device = 'cuda' if opt.cuda else 'cpu'
    if opt.historyevery % opt.printevery != 0:
        raise ValueError("historyevery must be a multiple of printevery: {} % {} != 0".format(opt.historyevery, opt.printevery))

    if opt.device == 'cuda':
        print('Cuda is using...\t','cuda is_available:', torch.cuda.is_available())
        assert torch.cuda.is_available()

    read_data(opt,mol_column)
    PROP = pd.read_csv(opt.cond_train_path)[opt.conditions]
    SRC, TRG = create_fields(opt)

    if not os.path.isdir('{}'.format(opt.save_folder_name)):
        os.mkdir('{}'.format(opt.save_folder_name))
    if not os.path.isdir('{}'.format(f'{server_dir}')):
        os.mkdir('{}'.format(f'{server_dir}'))
    if not os.path.isdir('{}'.format(f'{server_dir}/weights')):
        os.mkdir('{}'.format(f'{server_dir}/weights'))

    opt.path_script = import_path 
    opt.train = create_dataset_auto(opt, SRC, TRG, PROP, tr_te='tr',test_mode=test_mode)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("# of trainable parameters: {}".format(total_trainable_params))

    opt.optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(opt.lr_beta1, opt.lr_beta2), eps=opt.lr_eps)
    if opt.lr_scheduler == "SGDR":
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)
    opt.classifire = Ic50
    opt.TRG=TRG
    opt.SRC =SRC

    train_model(model, opt,case=case)


def train_ml_with_data(data:MLData=Body()):
        state = TrainState()
        state.add_new_case(case_name=data.case,
                           rewrite=True,
                           description=data.description)
        if data.data is not None:
                df = pd.DataFrame(data.data)
                data.data_path = f"automl/data/{data.case}"
                if not os.path.isdir(data.data_path):
                    os.mkdir(data.data_path)
                data.data_path = data.data_path + '/data.csv'
                df.to_csv(data.data_path) 
                      
        state.ml_model_upd_data(case=data.case,
                                data_path=data.data_path,
                                feature_column=data.feature_column,
                                target_column=data.target_column,
                                 predictable_properties={"regression":data.regression_props, "classification":data.classification_props})
        run_train_automl(case=data.case,
                         path_to_save=data.path_to_save,
                         timeout=data.timeout)
        
if __name__ == "__main__":
    state = TrainState(state_path='/projects/generative_models_data/generative_models/transformer/train_data/utils/state.json')
    CASE = 'Dslpdm'
    train_data = 'generative_models/transformer/docked_data_for_train/data_5vfi.csv'
    conditions = ['docking_score','QED','Synthetic Accessibility','PAINS','SureChEMBL','Glaxo','Brenk','IC50']
    test_mode = True

    if state(CASE) is None:#Check if case exist
        state.add_new_case(CASE,rewrite=False)
    
    # main(conditions = conditions,
    #      case=CASE, 
    #      server_dir = f'generative_models/transformer/train_{CASE}',
    #      data_path_with_conds = train_data,
    #      test_mode=test_mode)

