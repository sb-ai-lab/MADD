from typing import List
import pandas as pd
from rdkit.Contrib.SA_Score import sascorer
from rdkit import Chem,DataStructs
from rdkit.Chem import Descriptors, QED, rdDepictor, AllChem, Draw
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams



def check_brenk(smiles):
    params_brenk = FilterCatalogParams()
    params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog_brenk = FilterCatalog(params_brenk)
    mol = Chem.MolFromSmiles(smiles)
    return 1*catalog_brenk.HasMatch(mol)

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

def eval_qed(smile:str):
     score = Chem.QED.qed(Chem.MolFromSmiles(smile))
     return score

def eval_sa(smile:str):
     score = sascorer.calculateScore(Chem.MolFromSmiles(smile))
     return score

def PatternFilter(patterns, smiles):
    structures = list(filter(None, map(Chem.MolFromSmarts, patterns)))
    molecule = Chem.MolFromSmiles(smiles)
    return int(any(molecule.HasSubstructMatch(struct) for struct in structures))

if __name__=='__main__':
    print(eval_qed('CC(C)(C)C1=CC=C2C=CC13CCCC2=C(c1ccc(N)cc1)C(=O)CC3'))
    print(eval_P_S_G('CC(C)(C)C1=CC=C2C=CC13CCCC2=C(c1ccc(N)cc1)C(=O)CC3'))
    print(eval_sa('CC(C)(C)C1=CC=C2C=CC13CCCC2=C(c1ccc(N)cc1)C(=O)CC3'))