import argparse
from pydantic import BaseModel,dataclasses
from typing import Union,List,Optional
from generative_models.Batch import MyIterator


class Config_P(BaseModel):
    import_path : str
    save_folder_name : str
    imp_test:bool = False
    data_path_with_conds:str = 'generative_models/Sber_Alzheimer/train_cVAE_sber_skleroz_docking/docked_data_4j1r.csv'
    epochs : int = 15
    cuda : bool = True
    lr_scheduler : str = 'SGDR'
    lr_max : float = 0.005
    lr_min : float = 0.001
    use_cond2lat : bool = True
    latent_dim:int = 128
    d_model:int = 516
    n_layers : int = 12
    heads : int = 12
    dropout : float = 0.3
    batchsize : int = 2048
    max_strlen : int = 100
    load_weights:Union[str,None] = 'generative_models/Sber_Alzheimer/train_cVAE_sber_altz_docking/weights'
    load_weights_fields:str = None
    use_KLA:bool = True
    shift:str = '/../../../GAN'
    cond_dim : int = 1
    n_samples : int = 1000
    new_vocab : bool = False
    lang_format:str ='SMILES'
    calProp:bool = False
    lr_WarmUpSteps: int = 6000
    lr_beta1: float = 0.9
    lr_beta2: float = 0.98
    lr_eps: float = 1e-9
    conditions: Union[List[str],None] = None
    device:str = 'cuda'
    # KL Annealing
    use_KLA: bool
    KLA_ini_beta : float = 0.02
    KLA_inc_beta : float = 0.02
    KLA_max_beta : float = 1.0
    KLA_beg_epoch : int = 1
    lr : float = 0.0001
    checkpoint :int = 20
    k : int = 4
    create_valset:bool = True
    floyd:bool=True
    print_model:bool =False
    printevery:int=5
    historyevery:int=5
    verbose:bool=True
    path_script : Optional[str]
    src_pad:Optional[int]
    trg_pad:Optional[int]
    train_len:Optional[int]
    test_len:Optional[int]
    use_cond2dec:bool = False
    #train:Optional[MyIterator]

class Config:
    def __init__(self,*args,**kwargs):
        n = Config_P(**kwargs)
        for key, value in dict(n).items():
            setattr(self, key, value) 
        self.cond_train_path = self.data_path_with_conds
        self.src_data = self.data_path_with_conds
        self.src_data_te = self.data_path_with_conds
        self.trg_data = self.data_path_with_conds
        self.trg_data_te = self.data_path_with_conds

def configurate_parser(import_path : str,
                       save_folder_name : str,
                       imp_test = False,
                       data_path_with_conds = 'generative_models/Sber_Alzheimer/train_cVAE_sber_skleroz_docking/docked_data_4j1r.csv',
                       epochs : int = 15,
                       cuda : bool = True,
                       lr_scheduler : str = 'SGDR',
                       lr_max : float = 0.005,
                       lr_min : float = 0.001,
                       use_cond2lat : bool = True,
                       latent_dim:int = 128,
                       d_model:int = 516,
                       n_layers : int = 12,
                       heads : int = 12,
                       dropout : float = 0.3,
                       batchsize : int = 2048,
                       max_strlen : int = 100,
                       load_weights = 'generative_models/Sber_Alzheimer/train_cVAE_sber_altz_docking/weights',
                       load_weights_fields = None,
                       use_KLA = True,
                       shift = '/../../../GAN',
                       cond_dim : int = 1,
                       n_samples : int = 1000,
                       new_vocab : bool = False):
    """This is a configuration  parser function. This function create object with NameSpaces of function arguments. 
        That NameSpaces use later in code to initialize models, dataloaders, train parameters and etc. 

    Args:
        import_path (str): Path to training script runing directory.
        save_folder_name (str): Path to folder where saving model weigts.
        imp_test (bool, optional): Suppressed in current version. Flag to start test loop. Defaults to False.
            
            For src_data, src_data_te, trg_data and trg_data_te necessary to choose path to the one file!
            It is a rudiment after model archotecture modernisation from seq2seq translater to seq2seq generator, where
                sorce and target are same.
        src_data (str, optional): Train data with smiles path. Defaults to '/generative_models/data/ChEMBL_ic50.csv'.
        src_data_te (str, optional): Train data with smiles path. Defaults to '/generative_models/data/ChEMBL_ic50.csv'.
        trg_data (str, optional): Train data with smiles path. Defaults to '/generative_models/data/ChEMBL_ic50.csv'.
        trg_data_te (str, optional): Train data with smiles path. Defaults to '/generative_models/data/ChEMBL_ic50.csv'.

        epochs (int, optional): Number of epochs for trainig process. Defaults to 60.
        cuda (bool, optional): Choose whether use cuda. Defaults to True.
        lr_scheduler (str, optional): LR scheduler type. Defaults to 'SGDR'.
        lr_max (float, optional): lr_max. Defaults to 0.005.
        lr_min (float, optional): lr_min. Defaults to 0.00001.
        use_cond2lat (bool, optional):Choose whether use conditional case. If False - that mean you will train a VAE model. Defaults to True.
        latent_dim (int, optional): Size of latent noise vector dimension inside model. Defaults to 128.
        d_model (int, optional): Size of model. Defaults to 516.
        n_layers (int, optional): Numbers of transformer layers. Defaults to 12.
        heads (int, optional): Numbers of transformer heads. Defaults to 12.
        dropout (float, optional): Dropout rate. Defaults to 0.3.
        batchsize (int, optional): Batchsize. Defaults to 2048.
        max_strlen (int, optional): Max length of molecules SMILES in train DATA. Defaults to 100.
        load_weights (_type_, optional): Path to model weihts to load (Inside this direcotory you need to paste SRC.pkl,TRG.pkl and toklen_list.csv). Defaults to None.
        use_KLA (bool, optional): Whether to use KL divergence annealing. Defaults to True.
        cond_test_path (str, optional): Train data with smiles and conditions path. Defaults to '/nfs/home/gsololvyev/AAAI_code/generative_models/GAN/ChEMBL_training/data/database_ChEMBL_ic50.csv'.
        cond_train_path (str, optional): Train data with smiles and conditions path. Defaults to '/nfs/home/gsololvyev/AAAI_code/generative_models/GAN/ChEMBL_training/data/database_ChEMBL_ic50.csv'.
        shift (str, optional): Will be suppressed later. Defaults to '/../../../GAN'.
        cond_dim (int, optional): Number of contions values.
        n_samples (int, optional): Number of molecules samples for validational generation in validation part.

    Returns:
        NameSpace: NameSpace with training parameters.
    """
    parser = argparse.ArgumentParser()
    # Data settings
    parser.add_argument('-imp_test', type=bool, default=imp_test)
    data_path = import_path + shift
    parser.add_argument('-src_data', type=str, default=f'{data_path_with_conds}') # Path to train data csv
    parser.add_argument('-src_data_te', type=str, default=f'{data_path_with_conds}') # Path to train data csv
    parser.add_argument('-trg_data', type=str, default=f'{data_path_with_conds}') # Path to train data csv
    parser.add_argument('-trg_data_te', type=str, default=f'{data_path_with_conds}') # Path to train data csv
    parser.add_argument('-lang_format', type=str, default='SMILES')
    parser.add_argument('-calProp', type=bool, default=False) #if prop_temp.csv and prop_temp_te.csv exist, set False

    # Learning hyperparameters
    parser.add_argument('-epochs', type=int, default=epochs) 
    parser.add_argument('-cuda', type=str, default=cuda) 
    # parser.add_argument('-lr_scheduler', type=str, default="SGDR", help="WarmUpDefault, SGDR")
    parser.add_argument('-lr_scheduler', type=str, default=lr_scheduler, help="WarmUpDefault, SGDR")
    parser.add_argument('-lr_WarmUpSteps', type=int, default=6000, help="only for WarmUpDefault")
    parser.add_argument('-lr_max', type=float, default=lr_max)
    parser.add_argument('-lr_min', type=float, default=lr_min)
    parser.add_argument('-lr_beta1', type=float, default=0.9)
    parser.add_argument('-lr_beta2', type=float, default=0.98)
    parser.add_argument('-lr_eps', type=float, default=1e-9)

    # KL Annealing
    parser.add_argument('-use_KLA', type=bool, default=use_KLA)
    parser.add_argument('-KLA_ini_beta', type=float, default=0.02)
    parser.add_argument('-KLA_inc_beta', type=float, default=0.02)
    parser.add_argument('-KLA_max_beta', type=float, default=1.0)
    parser.add_argument('-KLA_beg_epoch', type=int, default=1) #KL annealing begin

    # Network sturucture
    parser.add_argument('-use_cond2dec', type=bool, default=False)
    parser.add_argument('-use_cond2lat', type=bool, default=use_cond2lat) # Choose whether use conditional case. If False - that mean you will train a VAE model
    parser.add_argument('-latent_dim', type=int, default=latent_dim)
    parser.add_argument('-cond_dim', type=int, default=cond_dim) # Number of condition values
    parser.add_argument('-d_model', type=int, default=d_model)
    parser.add_argument('-n_layers', type=int, default=n_layers)
    parser.add_argument('-heads', type=int, default=heads)
    parser.add_argument('-dropout', type=int, default=dropout)
    parser.add_argument('-lr', type=float, default=0.0001)
    #parser.add_argument('-batchsize', type=int, default=1024)
    parser.add_argument('-batchsize', type=int, default=batchsize)
    parser.add_argument('-max_strlen', type=int, default=max_strlen)  # Set max length of molecules SMILES in train DATA/

    # History
    parser.add_argument('-verbose', type=bool, default=True)
    parser.add_argument('-save_folder_name', type=str, default=save_folder_name)
    parser.add_argument('-print_model', type=bool, default=False)
    parser.add_argument('-printevery', type=int, default=5)
    parser.add_argument('-historyevery', type=int, default=5) # must be a multiple of printevery
    parser.add_argument('-load_weights',default=load_weights)
    parser.add_argument('-new_vocab',default=new_vocab)
    
    parser.add_argument('-load_weights_fields',default=load_weights_fields)
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=20)

    # For validate
    parser.add_argument('-n_samples', type=int, default=n_samples)
    parser.add_argument('-cond_test_path', type=str, default=data_path_with_conds)
    parser.add_argument('-cond_train_path', type=str, default=data_path_with_conds)
    parser.add_argument('-train_data_csv', type=str,
                        default=data_path_with_conds)# Need to check novelty of generated mols from train data
    parser.add_argument('-k', default=4) # Numb of beams inside beamsearch function.
    return parser