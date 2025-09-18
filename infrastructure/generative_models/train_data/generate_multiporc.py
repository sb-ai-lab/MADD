import sys
import os
import shutil
import time
import_path = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(import_path)
# sys.path.append(os.getcwd())
import lightgbm
import torch.multiprocessing as mp
from ic50_classifire_model.read_ic50 import Ic50
from inference import generate_alzh
from Process import *
import argparse
from Models import get_model
from rdkit import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
#from pipeline.classifier import Classifier
from utils.config_generate import configurate_parser

def generator(opt,
              n_samples=10000,spec_conds=[-10],
              return_dict=None,
              rank=0,
              path_to_save:str='alzh_gen_mols',
              save:bool=True,
              cuda:bool=True,
              path_to_val_model:str='generative_models/Sber_Alzheimer/ic50_classifire_model/kinase_inhib.pkl',
              index = '0',
              mean_=0,std_=1,
              
              ):
    '''
    The generator function generates the specified number of molecules.
        n_samples - number of molecules to be generated.
        path_to_save - suffix to the file path to save the molecules.
    It is necessary to give a name to the file for generation.
        save - whether to save the generated molecules? True/False
        spec_conds - None for random assignment of physical properties/
    list for specifying molecules of interest. Example: [1,1,0].
    '''
    
    opt = opt
    print(opt.load_weights)
    opt.device = 'cuda' if cuda else 'cpu'
    opt.path_script = import_path+f'/{path_to_save}/'
    if not os.path.isdir(opt.path_script):
        os.mkdir(opt.path_script )
    assert opt.k > 0
    assert opt.max_strlen > 10

    SRC, TRG = create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab)).share_memory()
    opt.classifire = Ic50
    df = generate_alzh(opt=opt,
             model=model,
             SRC=SRC,
             TRG=TRG,
             n_samples=n_samples,
             spec_conds=spec_conds,
             save=save,
             shift_path=index,
             mean_=mean_,std_=std_,
             path_to_val_model=path_to_val_model)
    return_dict[rank] = df
    return df


def shut_copy(src,trg):
    shutil.copy(src+'/SRC.pkl',trg+'/SRC.pkl')
    shutil.copy(src+'/TRG.pkl',trg+'/TRG.pkl')
    shutil.copy(src+'/toklen_list.csv',trg+'/toklen_list.csv')

if __name__ == '__main__':
    parser = configurate_parser(import_path=import_path,save_folder_name='alzh_gen_mols',)
    opt_Alz = parser.parse_args()
    mp.set_start_method('spawn')
    #generator(opt = opt_Alz, n_samples=100,spec_conds=[-9])

    manager = mp.Manager()
    return_dict = manager.dict()
    num_processes = 5
    # NOTE: this is required for the ``fork`` method to work
    
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=generator, args=(opt_Alz,100,[-14],return_dict,rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    df_total = pd.concat([i for i in return_dict.values()])
    df_total.to_csv(import_path+f'/alzh_gen_mols/valid_mols_wo_dupls0.csv')