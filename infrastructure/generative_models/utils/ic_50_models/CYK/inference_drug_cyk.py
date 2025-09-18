import os
import pandas as pd
from rdkit.Avalon import pyAvalonTools
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors, AllChem
import numpy as np
import warnings
from rdkit import RDLogger
import joblib
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint, GetMorganFingerprintAsBitVect
RDLogger.DisableLog('rdApp.*')
pd.set_option('display.float_format', '{:.2f}'.format)
warnings.filterwarnings('ignore')
from joblib import load

def predict(smiles_list):
    # with open("/projects/generative_models_data/utils/ic_50_models/CYK/stacking_regressor.pkl", "rb") as f:
    #     model = load(f)
    rename = {'bit'+str(i):'fingerprint_'+str(i) for i in range(0,2048)}
    model = load(r"infrastructure/generative_models/utils/ic_50_models/CYK/stacking_regressor.joblib")
    # df = fingerprints_descriptors(smiles_list, 'morgan', 2048, 2).rename(columns=rename)
    # cols = [rename[i] for i in rename.keys()]
    # df = df[cols]
    fp = [get_fingerprint(i) for i in smiles_list]
    predictions = model.predict(np.float64(fp))#(np.array(df.values.tolist()))
    return predictions

def get_fingerprint(smiles):
        fp_array = np.zeros((0,), dtype=np.int8)
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048)
        ConvertToNumpyArray(fp, fp_array)
        return fp_array

def safe_canon_smiles(smiles):
    try:
        return Chem.CanonSmiles(smiles)
    except Exception as e:
        print(f"Bad Smiles: {smiles}")
        return None
def get_all_descriptors(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()
    mol_descriptors = []
    for mol in mols:
        mol = Chem.AddHs(mol)
        descriptors = calc.CalcDescriptors(mol)
        mol_descriptors.append(descriptors)
    return mol_descriptors, desc_names
def generate_AVfpts(data):
    Avalon_fpts = []
    mols = [Chem.MolFromSmiles(x) for x in data if x is not None]
    for mol in mols:
        avfpts = pyAvalonTools.GetAvalonFP(mol, nBits=2048)
        Avalon_fpts.append(avfpts)
    return pd.DataFrame(np.array(Avalon_fpts))
def create_features_for_smiles(smiles_names):
    df = pd.DataFrame(columns=["Smiles"])
    for i in smiles_names:
        df = df._append({"Smiles": i}, ignore_index=True)
    df['Canonical Smiles'] = df.Smiles.apply(safe_canon_smiles)
    df.drop(['Smiles'], axis=1, inplace=True)
    #df.drop_duplicates(inplace=True, keep='first', subset='Canonical Smiles')
    mol_descriptors, descriptors_names = get_all_descriptors(df['Canonical Smiles'].tolist())
    descriptors_df = pd.DataFrame(mol_descriptors, columns=descriptors_names)
    AVfpts = generate_AVfpts(df['Canonical Smiles'])
    AVfpts.columns = AVfpts.columns.astype(str)
    df.drop(["Canonical Smiles"], axis=1, inplace=True)
    X_test = pd.concat([descriptors_df, AVfpts], axis=1)
    return X_test

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
if __name__=='__main__':
    predictions = predict(["Cc1cc(C)n(n1)-c1nnc2SCC(=Nn12)c1c[nH]c2ccccc12",
                           "Cc1cc(C)n(n1)-c1nnc2SCC(=Nn12)c1c[nH]c2ccccc12",
                           "Cc1cc(C)n(n1)-c1nnc2SCC(=Nn12)c1c[nH]c2ccccc12"])
    print(predictions)