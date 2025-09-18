import pickle as pi
import lightgbm
import numpy as np
from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem
import pandas as pd
from typing import List

def mol2fp(smiles:List[str]):
    """
    """
    fps = [] 
    clear_mols = [] 
    mols = [Chem.MolFromSmiles(smile) if smile is not None else None for smile in smiles]  
    for mol in mols:
        if mol is not None: 
            clear_mols.append(mol)
            fp = [AllChem.GetHashedMorganFingerprint(mol, 2, nBits=2048)]
            ar = np.zeros((1,), dtype=np.int8) 
            DataStructs.ConvertToNumpyArray(fp[0], ar) 
            fps.append(ar) 
    
    return fps,clear_mols

def Ic50(mol:List[str]):  
    #"""Calculates predicted Ic50"""  
    model =  pi.load(open('generative_models/transformer/ic50_classifire_model/kinase_inhib.pkl','rb')) 
    fp,clear_mols = mol2fp(mol)
    if len(fp)==0:
        return None,None
    ic50_scores = model.predict(fp)  
    return ic50_scores,clear_mols
if __name__ == "__main__":
    model = pi.load(open('utils/ic_50_models/skleroz_ic50_clf/checkpoints/ic50_btk_clf.pkl','rb'))  
    mols = pd.read_csv('generative_models/train_data/alzh_gen_mols/mols_generated_mols0.csv')
    #pd.read_csv('/nfs/home/gsololvyev/AAAI_code/generative_models/TVAE/generate/cvae_gen/valid/valid_mols_wo_dupls_cvae_0.csv')['0']
    #pd.read_csv('/nfs/home/gsololvyev/AAAI_code/sber/altz/kinase_inhib.csv')['Smiles']#
    ic50_scores,clear_mols = Ic50(list(mols['0']),model)
    # for i in list(mols):
    #     target_ic50.append()
    #mols['ic50'] = ic50_scores
    #mols.to_csv('/nfs/home/gsololvyev/AAAI_code/generative_models/GAN/ChEMBL_training/data/database_ChEMBL_ic50.csv')
    print(ic50_scores,clear_mols)