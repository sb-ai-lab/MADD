import pickle as pi

from huggingface_hub import HfApi
from autotrain.utils.base_state import TrainState
from scripts.model import MolGen
from typing import List
import os 
import pandas as pd
# steps = 8 or 10 good for ds size 1700000 samples
def run(path_ds: str, lr: float = 0.0003, bs: int = 256, steps: int = 10, hidden: int = 256,feature_column='Smiles'):
# data = []
    # with open(path_ds, "r") as f:
    #     for line in f.readlines()[1:]:
    #         data.append(line.split("\n")[0])
    df_train = pd.read_csv(path_ds)
    data = df_train[feature_column[0]].to_list()

    # create model
    gan_mol = MolGen(data, hidden_dim=hidden, lr=lr, device="cuda")
    # create dataloader
    loader = gan_mol.create_dataloader(data, batch_size=bs, shuffle=True, num_workers=0)
    # train model for 10000 steps
    gan_mol.train_n_steps(loader, max_step=steps, evaluate_every=150)
    # save model
    pi.dump(gan_mol, open(f'/projects/CoScientist/ChemCoScientist/generative_models/autotrain/train_GAN_121.pkl', 'wb'))

def auto_train(case,state,path_ds: str, lr: float = 0.0003, bs: int = 256, steps: int = 10, hidden: int = 256,fine_tune=False,feature_column=['Smiles']):

    df_train = pd.read_csv(path_ds)
    data = df_train[feature_column[0]].to_list()

    # create model
    gan_mol = MolGen(data, hidden_dim=hidden, lr=lr, device="cuda")
    # create dataloader
    loader = gan_mol.create_dataloader(data, batch_size=bs, shuffle=True, num_workers=0)
    # train model for 10000 steps
    if fine_tune:
        with open('infrastructure/generative_models/GAN/gan_lstm_refactoring/weights/v4_gan_mol_124_0.0003_8k.pkl', "rb") as f:
            gan_mol = pi.load(f)
    state.gen_model_upd_status(case=case,model_weight_path=f'infrastructure/generative_models/autotrain/GAN_weights/train_GAN_{case}',status=1)
    gan_mol.train_n_steps(loader, max_step=steps, evaluate_every=150)
 
    if not os.path.isdir(f'infrastructure/generative_models/autotrain/GAN_weights/train_GAN_{case}'):
        os.mkdir(f'infrastructure/generative_models/autotrain/GAN_weights/train_GAN_{case}')
    # save model
    pi.dump(gan_mol, open(f'infrastructure/generative_models/autotrain/GAN_weights/train_GAN_{case}/gan_weights.pkl', 'wb'))
    api = HfApi(token=os.getenv("HF_TOKEN"))
    state.gen_model_upd_status(case=case,model_weight_path=f'infrastructure/generative_models/autotrain/GAN_weights/train_GAN_{case}',status=2)
    api.upload_file(
    path_or_fileobj="infrastructure/generative_models/autotrain/utils/state.json",
    repo_id="SoloWayG/Molecule_transformer",
    repo_type="model",
    path_in_repo = 'state.json'
)   
    api.upload_folder(repo_id='SoloWayG/Molecule_transformer',
                              folder_path=f'infrastructure/generative_models/autotrain/GAN_weights',
                              path_in_repo=f'GAN_weights',
                              commit_message=f'Add GAN model for {case} case',
                              #delete_patterns=True
            )

# def main(server_dir = 'generative_models/train_dislip',
#          conditions : List[str] = ['ic50'],
#          case:str = 'Alzhmr',
#          data_path_with_conds = 'generative_models/docked_data_for_train/data_5vfi.csv',
#          test_mode = False,
#          state = None,
#          ml_model_url= 'http://10.64.4.247:81',
#          *args,
#          **kwargs):
#     """Main function to start training.

#     Args:
#         server_dir (str, optional): Path to save weights. Defaults to 'generative_models/transformer/train_cVAE_transformer_altz'.
#         conditions (List[str], optional): Names of condition values, that match with dataset conditions columns names. Defaults to ['ic50'].
#     """
#     if not os.path.isdir(server_dir):
#         os.mkdir(server_dir)
#     if not os.path.isdir(server_dir+f'/{case}'):
#         os.mkdir(server_dir+f'/{case}')


#     state.gen_model_upd_status(case=case,model_weight_path=server_dir+f'/{case}')

#     train_model_auto(model,opt,case=case,state=state)

# run train LSTM GAN
if __name__ == '__main__':
    # path = 'autotrain/data/data_mek.csv' #change if another
    # run(path,feature_column=['smiles'])

    state = TrainState(state_path='autotrain/utils/state.json')
    auto_train(case = "Alzheimer_regression",state=state,
                            feature_column=["smiles"],
                            #classification_props = ['IC50'], #All propreties from dataframe you want to calculate in the end
                            fine_tune=True,
                            path_ds='/projects/CoScientist/ChemCoScientist/generative_models/autotrain/data/data_mek.csv',
                        )