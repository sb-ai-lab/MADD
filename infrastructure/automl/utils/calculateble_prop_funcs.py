#from ctypes import Union
from typing import List, Union
import pandas as pd
from rdkit.Contrib.SA_Score import sascorer
from rdkit import Chem,DataStructs
from rdkit.Chem import Descriptors, QED, rdDepictor, AllChem, Draw, Crippen
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

#Functions for evaluting molecules properties. Take List[str] of smiles, returns list of properties values.

def check_brenk(smiles:List[str]):
    """
    Checks if a SMILES string or a list of SMILES strings contains a Brenk functional group, which are known to be associated with potential toxicity.
    
    Args:
        smiles (str or list of str): A single SMILES string or a list of SMILES strings to check.
    
    Returns:
        int or list of int: 1 if the SMILES string/molecule contains a Brenk functional group, 0 otherwise.  If a list of SMILES strings is provided, a list of 0s and 1s is returned, corresponding to each SMILES string in the input list.
    """
    params_brenk = FilterCatalogParams()
    params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog_brenk = FilterCatalog(params_brenk)
    if type(smiles) == str:
        mol = Chem.MolFromSmiles(smiles)
        return 1*catalog_brenk.HasMatch(mol)
    return [1*catalog_brenk.HasMatch(Chem.MolFromSmiles(smile)) for smile in smiles]
    
def check_chem_valid(smiles:List[str])->List[str]:
    """
    Filters a list of SMILES strings, retaining only those that represent chemically valid molecules and converting them to canonical SMILES.
    
    Args:
        smiles (List[str]): A list of SMILES strings representing molecules.
    
    Returns:
        List[str]: A list of SMILES strings, where each string represents a chemically valid molecule in canonical form.
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
    """
    Evaluate molecular alerts for a given SMILES string.
    
    This method identifies potentially problematic substructures (molecular alerts)
    within a molecule based on predefined sets of rules.  These alerts can indicate
    properties that might affect a molecule's suitability for certain applications,
    such as drug discovery.
    
    Args:
        smile (str): The SMILES string representing the molecule to evaluate.
        type_n (str, optional):  Specifies which alert set to use. 
            'all' evaluates against all implemented alert sets (PAINS, SureChEMBL, Glaxo). 
            Defaults to 'all'.
    
    Returns:
        List: A list of alert counts. The order of counts corresponds to 
            ['PAINS','Glaxo','SureChEMBL'] when `type_n` is 'all', or a single 
            alert count when a specific `type_n` is provided.
    """
    alert_table = pd.read_csv('infrastructure/automl/utils/alert_collections.csv')
    patterns = dict()
    if type_n == 'all':
        for name in ['PAINS', 'SureChEMBL', 'Glaxo']:
            patterns[name] = alert_table[alert_table['rule_set_name'] == name]['smarts']
        return [PatternFilter(patterns[i], smile) for i in ['PAINS','Glaxo','SureChEMBL']]  
    else:
          patterns[type_n] = alert_table[alert_table['rule_set_name'] == type_n]['smarts']
          return PatternFilter(patterns[type_n], smile)

def eval_pains(smiles:Union[List[str],str]):
    """
    Evaluates a list of SMILES strings to identify potentially problematic compounds that may interfere with biological assays.
    
    Args:
        smiles (Union[List[str], str]): A SMILES string or a list of SMILES strings to evaluate.
    
    Returns:
        list: A list of boolean values. True indicates the corresponding SMILES string is identified as a PAINS compound, False otherwise.
    """
    return [eval_P_S_G(smile,"PAINS") for smile in smiles]

def eval_sure_chembl(smiles:Union[List[str],str]):
    """
    Evaluates the SureChEMBL score for a SMILES string or a list of SMILES strings. This score provides a quantitative measure of chemical safety based on structural alerts.
    
    Args:
        smiles (Union[List[str], str]): A single SMILES string or a list of SMILES strings to evaluate.
    
    Returns:
        list: A list of SureChEMBL scores, corresponding to the input SMILES strings.
    """
    return [eval_P_S_G(smile,"SureChEMBL") for smile in smiles]

def eval_glaxo(smiles:Union[List[str],str]):
    """
    Evaluates a medicinal chemistry scoring function for a SMILES string or a list of SMILES strings.
    
    Args:
        smiles (Union[List[str], str]): A SMILES string or a list of SMILES strings to evaluate.
    
    Returns:
        list: A list of scores corresponding to the input SMILES strings.
    
    This method calculates a score for each input SMILES string using a predefined scoring function ("Glaxo" in this case). The score represents the predicted properties or suitability of the molecule for a specific purpose, aiding in compound prioritization and selection during drug discovery.
    """
    return [eval_P_S_G(smile,"Glaxo") for smile in smiles]

def eval_qed(smiles:Union[List[str],str]):
    """
    Computes the Quantitative Estimate of Drug-likeness (QED) score for a given molecule.
    
    The QED score is a metric used to assess the drug-likeness of a chemical compound, 
    providing an indication of its potential as a viable drug candidate. It considers 
    various physicochemical properties relevant to oral bioavailability.
    
    Args:
        smiles: A single SMILES string representing a molecule, or a list of SMILES strings 
               representing multiple molecules.
    
    Returns:
        float or List[float]: The QED score for the input molecule if a single SMILES string 
                               is provided.  If a list of SMILES strings is provided, 
                               returns a list of QED scores, one for each molecule.
    """
    if type(smiles) == str:
        return Chem.QED.qed(Chem.MolFromSmiles(smiles))
    return [Chem.QED.qed(Chem.MolFromSmiles(smile)) for smile in smiles]

def eval_sa(smiles:Union[List[str],str]):
    """
    Calculates the Synthetic Accessibility (SA) score for a molecule represented by a SMILES string.
    
    The SA score estimates how easily a molecule can be synthesized in a laboratory setting, 
    providing a quantitative measure of its synthetic feasibility. It is based on the 
    frequency of different chemical functionalities in known chemical reactions.
    
    Args:
        smiles (Union[str, List[str]]): A SMILES string representing a single molecule, 
            or a list of SMILES strings representing multiple molecules.
    
    Returns:
        Union[int, List[int]]: The SA score (an integer) for the input molecule if a 
            single SMILES string is provided.  If a list of SMILES strings is given, 
            returns a list of corresponding SA scores.
    """
    if type(smiles) == str:
          return sascorer.calculateScore(Chem.MolFromSmiles(smiles))
    return [sascorer.calculateScore(Chem.MolFromSmiles(smile)) for smile in smiles]

def PatternFilter(patterns, smiles):
    """
    Checks if a molecule, represented by a SMILES string, contains any of the specified substructures defined by SMARTS patterns.
    
    Args:
      patterns: A list of SMARTS strings representing the substructures to search for.
      smiles: A SMILES string representing the molecule to be analyzed.
    
    Returns:
      int: 1 if at least one of the SMARTS patterns is found as a substructure within the molecule, 0 otherwise.
    """
    structures = list(filter(None, map(Chem.MolFromSmarts, patterns)))
    molecule = Chem.MolFromSmiles(smiles)
    return int(any(molecule.HasSubstructMatch(struct) for struct in structures))

def logP(smiles:Union[List[str],str]):
    """
    Calculates the octanol-water partition coefficient (logP) for a given molecule represented by a SMILES string.
    
    Args:
      smiles (Union[List[str], str]): A SMILES string or a list of SMILES strings representing the molecule(s) for which to calculate logP.
    
    Returns:
      list: A list of calculated logP values, one for each input SMILES string.  LogP is a measure of a molecule's lipophilicity and is important for understanding its behavior in biological systems and chemical processes.
    """
     
    return [Crippen.MolLogP(Chem.MolFromSmiles(smile)) for smile in smiles]

def polar_surf_area(smiles:Union[List[str],str]):
    """
    Computes the topological polar surface area (TPSA) for a list of SMILES strings. TPSA is a valuable descriptor in cheminformatics used to estimate the hydrogen-bonding capacity of a molecule, which influences its permeability and overall drug-like properties.
    
    Args:
        smiles (Union[List[str], str]): A single SMILES string or a list of SMILES strings representing the molecules to analyze.
    
    Returns:
        list: A list of TPSA values (in Ångströms²) corresponding to each input SMILES string.
    """
     
    return [Descriptors.TPSA(Chem.MolFromSmiles(smile)) for smile in smiles]

def h_bound_donors(smiles:Union[List[str],str]):
    """
    Calculates the number of hydrogen bond donors for each molecule represented by a SMILES string.
    
    Args:
        smiles (Union[List[str], str]): A SMILES string or a list of SMILES strings representing the molecule(s).
    
    Returns:
        list: A list of integers, where each integer represents the number of hydrogen bond donors for the corresponding input SMILES string.
    
    The method determines the number of hydrogen bond donors because this property is crucial for understanding a molecule's interactions and potential biological activity. Analyzing these donors aids in predicting a molecule's behavior in various chemical and biological systems.
    """
     
    return [Descriptors.NumHDonors(Chem.MolFromSmiles(smile)) for smile in smiles]

def aromatic_rings(smiles:Union[List[str],str]):
    """
    Counts the number of aromatic rings in a SMILES string or a list of SMILES strings.
    
    Args:
      smiles (Union[List[str], str]): A single SMILES string or a list of SMILES strings representing molecular structures.
    
    Returns:
      list: A list containing the number of aromatic rings for each SMILES string in the input.  The method analyzes the molecular structure encoded in each SMILES string to identify and count aromatic ring systems.
    """
     
    return [sum(1 for ring in Chem.GetSSSR(Chem.MolFromSmiles(smile)) if all(atom.GetIsAromatic() for atom in [Chem.MolFromSmiles(smile).GetAtomWithIdx(i) for i in ring])) for smile in smiles]

def rotatable_bonds(smiles:Union[List[str],str]):
    """
    Calculates the number of rotatable bonds for each SMILES string.
    
    Args:
        smiles (Union[List[str], str]): A SMILES string or a list of SMILES strings representing molecules.
    
    Returns:
        list: A list of integers, where each integer represents the number of rotatable bonds 
            in the corresponding molecule.  This metric helps to quantify the flexibility
            of a molecule, which is relevant in understanding its potential properties and interactions.
    """
     
    return [Descriptors.NumRotatableBonds(Chem.MolFromSmiles(smile)) for smile in smiles]

def h_bound_acceptors(smiles:Union[List[str],str]):
    """
    Calculates the number of hydrogen bond acceptors for a list of SMILES strings.
    
    Args:
        smiles (Union[List[str], str]): A list of SMILES strings or a single SMILES string representing chemical structures.
    
    Returns:
        list: A list of integers, where each integer represents the number of hydrogen bond acceptors for the corresponding SMILES string.
    
    This method determines the number of hydrogen bond acceptor atoms within each provided molecular structure. Hydrogen bond acceptors are crucial for understanding molecular interactions and are frequently used in drug discovery and materials science to predict binding affinities and material properties.
    """
     
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