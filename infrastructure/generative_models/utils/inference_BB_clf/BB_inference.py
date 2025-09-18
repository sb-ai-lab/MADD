import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import QED,Crippen, Lipinski, Descriptors, GraphDescriptors, Fragments, FragmentMatcher,  SaltRemover, MolFromSmiles, Draw, GetFormalCharge, MolToSmiles
from rdkit.Chem.EState.EState_VSA import VSA_EState_
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Avalon import pyAvalonTools
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
import numpy as np
from rdkit import RDLogger
import warnings
import matplotlib.pyplot as plt
RDLogger.DisableLog('rdApp.*')
pd.set_option('display.float_format', '{:.2f}'.format)
warnings.filterwarnings('ignore')


def predict(smiles_list):
    model = joblib.load("infrastructure/generative_models/utils/inference_BB_clf/model_BB_clf.pkl")
    predictions = model.predict(create_features_for_smiles(create_df_without_descriptors(smiles_list)))
    return predictions

def create_features_for_smiles(df_without_descriptors):
    df = df_without_descriptors['SMILES_uncharge']
    df = df.to_frame()
    df = df.rename(columns={'SMILES_uncharge': 'SMILES'})
    df = get_qed_and_crippen_modules(df)
    df = get_lipinski_module(df)
    df = get_descriptors_modules(df)
    df = get_graph_descriptors_module(df)
    df = get_all_descriptors(df)
    kill_inf_error(df)
    AVfpts = generate_AVfpts(df['SMILES'])
    AVfpts.columns = AVfpts.columns.astype(str)
    result = pd.concat([df, AVfpts], axis=1)
    result.drop(columns = ["SMILES"], axis = 1, inplace=True)
    kill_inf_error(result)
    result = prepare_to_predict(result)
    return result
def get_qed_and_crippen_modules(df):
    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(row.SMILES)
        qed_vector = QED.properties(mol)
        df.at[i, 'MW'] = round(qed_vector[0], 2)
        df.at[i, '#HBA'] = qed_vector[2]
        df.at[i, '#HBD'] = qed_vector[3]
        df.at[i, 'PSA'] = qed_vector[4]
        df.at[i, '#ROTB'] = qed_vector[5]
        df.at[i, '#ALERTS'] = qed_vector[7]
        df.at[i, 'MlogP'] = round(Crippen.MolLogP(mol), 2)
        df.at[i, '#MR'] = round(Crippen.MolMR(mol), 2)
    return df

def get_lipinski_module(df):
    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(row.SMILES)
        df.at[i, '#HeavyAtoms'] = Lipinski.HeavyAtomCount(mol)
        df.at[i, '#NHOH'] = Lipinski.NHOHCount(mol)
        df.at[i, '#NO'] = Lipinski.NOCount(mol)
        df.at[i, '#AromaticCarbocycles'] = Lipinski.NumAromaticCarbocycles(mol)
        df.at[i, '#AromaticHeterocycles'] = Lipinski.NumAromaticHeterocycles(mol)
        df.at[i, '#Heteroatoms'] = Lipinski.NumHeteroatoms(mol)
    return df

def get_descriptors_modules(df):
    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(row.SMILES)
        df.at[i, 'Morgan2'] = round(Descriptors.FpDensityMorgan2(mol), 2)
        df.at[i, 'Morgan3'] = round(Descriptors.FpDensityMorgan3(mol), 2)
        df.at[i, 'HeavyAtomMW'] = round(Descriptors.HeavyAtomMolWt(mol), 2)
        df.at[i, 'MaxPartialCharge'] = Descriptors.MaxPartialCharge(mol)
        df.at[i, 'MinPartialCharge'] = Descriptors.MinPartialCharge(mol)
        df.at[i, '#ValenceElectrons'] = Descriptors.NumValenceElectrons(mol)
    return df

def get_graph_descriptors_module(df):
    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(row.SMILES)
        df.at[i, 'BertzCT'] = round(GraphDescriptors.BertzCT(mol), 2)
        df.at[i, 'Kappa1'] = round(GraphDescriptors.Kappa1(mol), 2)
    return df

def get_all_descriptors(df):
    smiles = df["SMILES"].to_list()
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()
    mol_descriptors = []
    for mol in mols:
        mol = Chem.AddHs(mol)
        descriptors = calc.CalcDescriptors(mol)
        mol_descriptors.append(descriptors)
    df[[*desc_names]] = mol_descriptors
    return df
def generate_AVfpts(data):
    Avalon_fpts = []
    mols = [Chem.MolFromSmiles(x) for x in data if x is not None]
    for mol in mols:
        avfpts = pyAvalonTools.GetAvalonFP(mol, nBits=512)
        Avalon_fpts.append(avfpts)
    return pd.DataFrame(np.array(Avalon_fpts))



def create_df_without_descriptors(smiles_list):
    data = pd.DataFrame(columns=["SMILES"])
    for i in smiles_list:
        data = data._append({"SMILES": i}, ignore_index=True)
    data['SMILES'] = data.SMILES.apply(safe_canon_smiles)
    data.reset_index(drop=True, inplace=True)
    data.drop_duplicates()
    data = del_wrong_smiles(data)
    data = do_smiles_preprocessing(data)
    data = data.drop(labels=['SMILES_clear', 'ROMol', 'FORMAL_CHARGE_ch'], axis=1)
    data = data.dropna()
    data = data.reset_index(drop=True)
    data = remove_inorganic(data)
    data = data.reset_index(drop=True)
    kill_inf_error(data)
    return data

def safe_canon_smiles(smiles):
    try:
        return Chem.CanonSmiles(smiles)
    except Exception as e:
        print(f"Bad Smiles: {smiles}")
        return None
def del_wrong_smiles(data):
    for i, row in data.iterrows():
        if 'C' not in row.SMILES and 'c' not in row.SMILES:
            data = data.drop(labels=[i], axis=0)
    return data
def do_smiles_preprocessing(data):
    data['Agglomeration'] = data['SMILES'].apply(lambda x: '.' in x)
    data['SMILES_clear'] = data['SMILES'].apply(lambda x: max(x.split('.'), key=len))
    data['ROMol'] = data['SMILES_clear'].apply(lambda x: Chem.MolFromSmiles(x))
    un = rdMolStandardize.Uncharger()
    try:
        data['FORMAL_CHARGE_ch'] = data['SMILES_clear'].apply(lambda x: GetFormalCharge(Chem.MolFromSmiles(x)))
        data['SMILES_uncharge'] = data['ROMol'].apply(lambda x: Chem.MolToSmiles(un.uncharge(x), kekuleSmiles=True))
        data['FORMAL_CHARGE_unch'] = data['SMILES_uncharge'].apply(lambda x: GetFormalCharge(Chem.MolFromSmiles(x)))
    except:
        pass
    return data
def remove_inorganic(data):
    index_sm = data.index.tolist()
    sm = data['SMILES_uncharge'].values
    cl = data[data['SMILES_uncharge'] == '[O-][Cl+3]([O-])([O-])O']
    list_inorg_el = ['Al', 'Au', 'Ar,''Ba', 'Be', 'Bi', 'Ca', 'Cd', 'Co', 'Cr', 'Cu', 'Fe', 'Gd', 'Ge', 'Hf',
                     'Hg', 'In', 'K', 'Kr' 'La', 'Mg', 'Mn', 'Na', 'Ni', 'Pb', 'Pt', 'Sb', 'Sn', 'Sr', 'Te',
                     'V', 'Zn', 'Li', 'Xe', 'Rn', 'Ne']
    index_drop = []
    uniq_el = set([])
    for sm_i in range(len(sm)):
        mol_sm = MolFromSmiles(sm[sm_i])
        try:
            list_at = [str(x.GetSymbol()) for x in mol_sm.GetAtoms()]
            intersec = list(set(list_at) & set(list_inorg_el))
            T_C = 'C' in list_at
            if T_C == False:
                index_drop.append(index_sm[sm_i])
            elif len(intersec) > 0:
                index_drop.append(index_sm[sm_i])
            uniq_el = uniq_el | set(list_at)
        except:
            index_drop.append(index_sm[sm_i])
    data = data.drop(index=index_drop)
    return data
def prepare_to_predict(result):
    result.fillna(0, inplace=True)
    scaler = joblib.load("infrastructure/generative_models/utils/inference_BB_clf/scaler_BB_clf.pkl")
    features = joblib.load("infrastructure/generative_models/utils/inference_BB_clf/most_importance_features.pkl")
    data = result[features]
    data = scaler.transform(data)
    return data
def kill_inf_error(data):
    return data.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

if __name__=='__main__':
    predictions = predict(["NC(N)=Nc1nc(-c2ccc(O)c(O)c2)cs1"])
    results = pd.DataFrame(predictions, columns =["predictions"])
    print(results)
