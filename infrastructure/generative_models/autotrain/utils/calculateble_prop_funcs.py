#from ctypes import Union
from typing import List, Union
import pandas as pd
from rdkit.Contrib.SA_Score import sascorer
from rdkit import Chem,DataStructs
from rdkit.Chem import Descriptors, QED, rdDepictor, AllChem, Draw, Crippen
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

#Functions for evaluting molecules properties. Take List[str] of smiles, returns list of properties values.

def check_brenk(smiles:List[str]):
    params_brenk = FilterCatalogParams()
    params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog_brenk = FilterCatalog(params_brenk)
    if type(smiles) == str:
        mol = Chem.MolFromSmiles(smiles)
        return 1*catalog_brenk.HasMatch(mol)
    return [1*catalog_brenk.HasMatch(Chem.MolFromSmiles(smile)) for smile in smiles]
    
def check_chem_valid(smiles:List[str])->List[str]:
    """Check smiles for chemical validity and return only valid molecules.

    Args:
        smiles (List[str]): Molecules Smiles strings

    Returns:
        List[str]: Molecules Smiles strings
    """
    generated_coformers_clear = []
    for smiles_mol in smiles:
            if smiles_mol=='':
                continue
            if Chem.MolFromSmiles(str(smiles_mol)) is None:
                continue
            generated_coformers_clear.append(smiles_mol)
    generated_coformers_clear = [Chem.MolToSmiles(Chem.MolFromSmiles(str(i))) for i in generated_coformers_clear]
    return generated_coformers_clear

def eval_P_S_G(smile:str,type_n:str='all'):
    """Evalute PAINS,SureChEMBL and Glaxo metrics for smile.

    Args:
        smile (str): SMILE molecule describe
        type_n (str): What function need to evalute, if "all" - all functions will be evalute and return
    Returns:
        List: List with metrics of PAINS,SureChEMBL and Glaxo metrics
    """
    alert_table = pd.read_csv('infrastructure/generative_models/autotrain/utils/alert_collections.csv')
    patterns = dict()
    if type_n == 'all':
        for name in ['PAINS', 'SureChEMBL', 'Glaxo']:
            patterns[name] = alert_table[alert_table['rule_set_name'] == name]['smarts']
        return [PatternFilter(patterns[i], smile) for i in ['PAINS','Glaxo','SureChEMBL']]  
    else:
          patterns[type_n] = alert_table[alert_table['rule_set_name'] == type_n]['smarts']
          return PatternFilter(patterns[type_n], smile)

def eval_pains(smiles:Union[List[str],str]):
    return [eval_P_S_G(smile,"PAINS") for smile in smiles]

def eval_sure_chembl(smiles:Union[List[str],str]):
    return [eval_P_S_G(smile,"SureChEMBL") for smile in smiles]

def eval_glaxo(smiles:Union[List[str],str]):
    return [eval_P_S_G(smile,"Glaxo") for smile in smiles]

def eval_qed(smiles:Union[List[str],str]):
    if type(smiles) == str:
        return Chem.QED.qed(Chem.MolFromSmiles(smiles))
    return [Chem.QED.qed(Chem.MolFromSmiles(smile)) for smile in smiles]

def eval_sa(smiles:Union[List[str],str]):
    if type(smiles) == str:
          return sascorer.calculateScore(Chem.MolFromSmiles(smiles))
    return [sascorer.calculateScore(Chem.MolFromSmiles(smile)) for smile in smiles]

def PatternFilter(patterns, smiles):
    structures = list(filter(None, map(Chem.MolFromSmarts, patterns)))
    molecule = Chem.MolFromSmiles(smiles)
    return int(any(molecule.HasSubstructMatch(struct) for struct in structures))

def logP(smiles:Union[List[str],str]):
     
    return [Crippen.MolLogP(Chem.MolFromSmiles(smile)) for smile in smiles]

def polar_surf_area(smiles:Union[List[str],str]):
     
    return [Descriptors.TPSA(Chem.MolFromSmiles(smile)) for smile in smiles]

def h_bound_donors(smiles:Union[List[str],str]):
     
    return [Descriptors.NumHDonors(Chem.MolFromSmiles(smile)) for smile in smiles]

def aromatic_rings(smiles:Union[List[str],str]):
     
    return [sum(1 for ring in Chem.GetSSSR(Chem.MolFromSmiles(smile)) if all(atom.GetIsAromatic() for atom in [Chem.MolFromSmiles(smile).GetAtomWithIdx(i) for i in ring])) for smile in smiles]

def rotatable_bonds(smiles:Union[List[str],str]):
     
    return [Descriptors.NumRotatableBonds(Chem.MolFromSmiles(smile)) for smile in smiles]

def h_bound_acceptors(smiles:Union[List[str],str]):
     
    return [Descriptors.NumHAcceptors(Chem.MolFromSmiles(smile)) for smile in smiles]


config = { 
    #Functions for evaluting molecules properties. Take List[str] of smiles, returns list of properties values.
    "Validity" : check_chem_valid,
    "Brenk" : check_brenk,
    "QED" : eval_qed,
    "Synthetic Accessibility" : eval_sa,
    "LogP" : logP,
    "Polar Surface Area": polar_surf_area,
    "H-bond Donors": h_bound_donors,
    "H-bond Acceptors": aromatic_rings,
    "Rotatable Bonds": rotatable_bonds,
    "Aromatic Rings": h_bound_acceptors,
    "Glaxo" : eval_glaxo,
    "SureChEMBL" : eval_sure_chembl,
    "PAINS" : eval_pains
}

if __name__=='__main__':
    print(eval_pains(['CC(C)(C)C1=CC=C2C=CC13CCCC2=C(c1ccc(N)cc1)C(=O)CC3','CC(C)(C)C1=CC=C2C=CC13CCCC2=C(c1ccc(N)cc1)C(=O)CC3']))
    print(h_bound_acceptors(['CC(C)(C)C1=CC=C2C=CC13CCCC2=C(c1ccc(N)cc1)C(=O)CC3']))
    print(rotatable_bonds(['CC(C)(C)C1=CC=C2C=CC13CCCC2=C(c1ccc(N)cc1)C(=O)CC3']))