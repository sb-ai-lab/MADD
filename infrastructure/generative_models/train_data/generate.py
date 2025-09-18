import sys
import os
import shutil
import_path = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(import_path)
# sys.path.append(os.getcwd())
import lightgbm
from ic50_classifire_model.read_ic50 import Ic50
from inference import generate_alzh,generate_alzh_for_agent
from Process import *
import argparse
from Models import get_model
from rdkit import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
#from pipeline.classifier import Classifier
from train_data.utils.config_generate import configurate_parser

def generator(opt,
              n_samples=10000,
              path_to_save:str='alzh_gen_mols',
              spec_conds=[-10],
              save:bool=True,
              cuda:bool=True,
              path_to_val_model:str='generative_models/transformer/ic50_classifire_model/kinase_inhib.pkl',
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
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
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
    return df

def generator_for_agent_multi_prop(opt,
              n_samples=10000,
              path_to_save:str='alzh_gen_mols',
              spec_conds=[[-9,-12],[0.8,1],[0,2.99],[0,0],[0,0],[0,0],[0,0],[1,1]],
              save:bool=False,
              cuda:bool=True,
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
    print(opt.device)
    opt.path_script = import_path+f'/{path_to_save}/'
    if not os.path.isdir(opt.path_script):
        os.mkdir(opt.path_script )
    assert opt.k > 0
    assert opt.max_strlen > 10

    SRC, TRG = create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    #opt.classifire = Ic50
    df = generate_alzh_for_agent(opt=opt,
             model=model,
             SRC=SRC,
             TRG=TRG,
             n_samples=n_samples,
             spec_conds=spec_conds,
             save=save,
             shift_path=index,
             mean_=mean_,std_=std_,
             )
    return df

def generator_for_agent(opt,
              n_samples=10000,
              path_to_save:str='alzh_gen_mols',
              spec_conds=[-9],
              save:bool=False,
              cuda:bool=True,
              path_to_val_model:str='generative_models/transformer/ic50_classifire_model/kinase_inhib.pkl',
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
    print(opt.device)
    opt.path_script = import_path+f'/{path_to_save}/'
    if not os.path.isdir(opt.path_script):
        os.mkdir(opt.path_script )
    assert opt.k > 0
    assert opt.max_strlen > 10

    SRC, TRG = create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    opt.classifire = Ic50
    df = generate_alzh_for_agent(opt=opt,
             model=model,
             SRC=SRC,
             TRG=TRG,
             n_samples=n_samples,
             spec_conds=spec_conds,
             save=save,
             shift_path=index,
             mean_=mean_,std_=std_,
             path_to_val_model=path_to_val_model)
    return df


def shut_copy(src,trg):
    shutil.copy(src+'/SRC.pkl',trg+'/SRC.pkl')
    shutil.copy(src+'/TRG.pkl',trg+'/TRG.pkl')
    shutil.copy(src+'/toklen_list.csv',trg+'/toklen_list.csv')

if __name__ == '__main__':
    parser = configurate_parser(import_path=import_path,save_folder_name='alzh_gen_mols',)
    opt_Alz = parser.parse_args()

    generator(opt = opt_Alz, n_samples=1000,spec_conds=[-9])