import os
import pandas as pd
from rdkit.Avalon import pyAvalonTools
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import numpy as np
import warnings
from rdkit import RDLogger
import joblib
RDLogger.DisableLog('rdApp.*')
pd.set_option('display.float_format', '{:.2f}'.format)
warnings.filterwarnings('ignore')


#a function that predicts the standard value for the transmitted smiles list
def predict(smiles_list):
    model = joblib.load("infrastructure/generative_models/utils/ic_50_models/catechol_classif_inference/model_catechol_clf.pkl")
    predictions = model.predict(create_features_for_smiles(smiles_list))
    return 1 - predictions

#a function that brings the transmitted smiles to the canonical form
def safe_canon_smiles(smiles):
    try:
        return Chem.CanonSmiles(smiles)
    except Exception as e:
        print(f"Bad Smiles: {smiles}")
        return None

#a function that generates descriptors to describe the smiles molecule
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

#a function that generates fingerprints to describe the structure of a molecule
def generate_AVfpts(data):
    Avalon_fpts = []
    mols = [Chem.MolFromSmiles(x) for x in data if x is not None]
    for mol in mols:
        avfpts = pyAvalonTools.GetAvalonFP(mol, nBits=512)
        Avalon_fpts.append(avfpts)
    return pd.DataFrame(np.array(Avalon_fpts))

#a function that creates a dataframe with all the features for the transmitted smiles
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
if __name__=='__main__':
    predictions = predict(["Nc1ncnc2n(cnc12)[C@@H]1O[C@H](\C=C\CNC(=O)c2cc(cc(O)c2O)N(=O)=O)[C@@H](O)[C@H]1O",
                           "Nc1ncnc2n(cnc12)[C@@H]1O[C@H](\C=C\CNC(=O)c2cc(cc(O)c2O)N(=O)=O)[C@@H](O)[C@H]1O",
                           "CN1C2=C(C(=O)N(C1=O)C)NC=N2"])
    print(predictions)