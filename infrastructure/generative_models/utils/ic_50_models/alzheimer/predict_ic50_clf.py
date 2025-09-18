from typing import List
import pandas as pd
import numpy as np 
import pickle
import sklearn
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint, GetMorganFingerprintAsBitVect
from rdkit.Chem import MACCSkeys  
import warnings
warnings.filterwarnings('ignore')

def fingerprints_descriptors(smiles_list, fp_type, size, radius):
  fps = []
  mols = [Chem.MolFromSmiles(molecule) for molecule in smiles_list]
  if fp_type == 'morgan':
     for mol in mols:
       mfps = np.array(GetMorganFingerprintAsBitVect(mol, radius, nBits = size))
       fps.append(mfps)
  elif fp_type == 'maccs':
     for mol in mols:
       mfps = np.array(GetMACCSKeysFingerprint(mol).ToList())
       fps.append(mfps)            
  df_fps = pd.DataFrame(fps, columns=['bit{}'.format(i) for i in range(np.array(fps).shape[1])])
  return df_fps


def smi_to_MACCS(smiles:List[str]):   
  mols = [np.array(list(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smile)).ToBitString())).astype(int) if smile else np.zeros(167) for smile in smiles]
  return mols


def main():
  df = pd.read_csv('generative_models/GAN/ChEMBL_training/data/database_ChEMBL_ic50_filter.csv')
  model = pickle.load(open('utils/ic_50_models/alzheimer/alzheimer_clf.pkl', 'rb'))
  
  fps = smi_to_MACCS(df['0'].to_list()[:2000])
  labels = model.predict(fps)
  for i in labels:
    print(i)

def eval_ic_50_alzheimer(smiles : List[str]):
  model = pickle.load(open('infrastructure/generative_models/utils/ic_50_models/alzheimer/alzheimer_clf.pkl', 'rb'))
  fps = smi_to_MACCS(smiles)
  labels = model.predict(fps)
  return 1 - labels


if __name__ == "__main__":
  #main()
  print(eval_ic_50_alzheimer(['C[C@@H](C(F)(F)F)N1C(=C(C(=N1)C2=CC=C(C=C2)CNC(=O)C3=C(C=CC(=C3)F)OC)C(=O)N)N',
  'C[C@H]1CN(CCN1C2=CN=C(C=C2)NC3=CC(=CN(C3=O)C)C4=C(C(=NC=C4)N5CCN6C7=C(CC(C7)(C)C)C=C6C5=O)CO)C8COC8',
  'C1C[C@H](OC[C@@H]1NC2=NC=NC3=C2C(=CN3)C(=O)C4=C(C=C(C=C4)OC5=CC=CC=C5)Cl)CO',
  'CC1=CN=C(N1)C2=C3C(=C(C=C2)C4=C(C=C(C=C4)NC(=O)NC5=C(C=C(C=C5F)F)F)F)CNC3=O',
  'C1CC1C2=CC(=C3C(=C2)C=CN(C3=O)C4=CC=CC(=C4CO)C5=C6C=C(NC6=NC(=N5)N)C7=CCN(CC7)C8COC8)F',
  'CC1(c2ccccn2)ON=C(c2ccccc2Cl)O1']))