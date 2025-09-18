import statistics
from typing import List
import pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem import  AllChem

def check_novelty_mol_path(
        train_dataset_path: str,
        gen_data: list,
        train_col_name: str,
        gen_col_name: str,
        ):
    """Function for count how many new molecules generated compared with train data.


    :param train_dataset_path: Path to csv train dataset.
    :param gen_data: gen molecules.
    :param train_col_name: Name of column that consist a molecule strings.
    :param gen_col_name: Name of column that consist a molecule strings.
    :return:
    """
    train_d = pd.read_csv(train_dataset_path)[train_col_name]
    gen_d = pd.DataFrame(gen_data,columns=[gen_col_name])
    gen_len = len(gen_d)
    duplicates = gen_d.duplicated(subset=gen_col_name, keep='first').sum()/len(gen_d)
    total_len_gen = len(gen_d[gen_col_name])
    #gen_d = gen_d[gen_d['val_check']==1][gen_col_name]
    #len_train = len(train_d)
    len_gen = len(gen_d.drop_duplicates())
    novelty =( len(gen_d[gen_col_name].drop_duplicates())-gen_d[gen_col_name].drop_duplicates().isin(train_d).sum() )/ gen_len * 100
    print('Generated molecules consist of',novelty, '% unique new examples',
          '\t',
          f'duplicates: {duplicates}')
    return novelty,duplicates

def check_novelty(
        train_dataset_path: str,
        gen_data_path: str,
        train_col_name: str,
        gen_col_name: str) ->str:
    """Function for count how many new molecules generated compared with train data.


    :param train_dataset_path: Path to csv train dataset.
    :param gen_data_path: Path to csv gen dataset.
    :param train_col_name: Name of column that consist a molecule strings.
    :param gen_col_name: Name of column that consist a molecule strings.
    :return:
    """
    train_d = pd.read_csv(train_dataset_path)[train_col_name]
    gen_d = pd.read_csv(gen_data_path)
    duplicates = gen_d.duplicated(subset=gen_col_name, keep='first').sum()
    total_len_gen = len(gen_d[gen_col_name])
    gen_d = gen_d[gen_d['val_check']==1][gen_col_name]
    #len_train = len(train_d)
    len_gen = len(gen_d)

    print('Generated molecules consist of',(len_gen-train_d.isin(gen_d).sum())/len_gen*100, '% new examples',
          '\t',f'{len_gen/total_len_gen*100}% valid molecules generated','\t',
          f'duplicates, {duplicates}')



def check_novelty_chembl(smiles: List[str],
        train_data_path: str = 'generative_models/Sber_Alzheimer/docked_data_4j1r.csv',
        train_col_name:str= 'canonical_smiles',
        ) ->List[str]:
    """Function for count how many new molecules generated compared with train data.


    :param train_dataset_path: Path to csv train dataset.
    :param gen_data_path: Path to csv gen dataset.
    :param train_col_name: Name of column that consist a molecule strings.
    :param gen_col_name: Name of column that consist a molecule strings.
    :return:
    """
    train_d = pd.read_csv(train_data_path)[train_col_name]
    gen_s = pd.Series(smiles, name='smiles')
    gen_s = gen_s[~gen_s.isin(train_d.to_list())].drop_duplicates()
    self_diversity = check_self_diversity(gen_s.to_list())


    #print(gen_s.to_list())
    return gen_s.to_list(),self_diversity

def check_diversity(smiles:List[str],train_mols:List[str]):
    fpgen = AllChem.GetRDKitFPGenerator()
    self_scores = []
    gen_fp = [fpgen.GetFingerprint(mol) for mol in [Chem.MolFromSmiles(i) for i in smiles]]
    train_fp = [fpgen.GetFingerprint(Chem.MolFromSmiles(x)) for x in train_mols]
    for i,mol in enumerate(gen_fp):
        self_scores.append(max(DataStructs.BulkTanimotoSimilarity(mol, gen_fp.pop(i))))


    #self_scores = DataStructs.BulkTanimotoSimilarity(gen_fp, gen_fp)
    train_scores = DataStructs.BulkTanimotoSimilarity(gen_fp, train_fp)
    return self_scores,max(train_scores)

def check_self_diversity(smiles:List[str]):
    fpgen = AllChem.GetRDKitFPGenerator()
    self_scores = []
    gen_fp = [fpgen.GetFingerprint(mol) for mol in [Chem.MolFromSmiles(i) for i in smiles]]
    if len(gen_fp)==1:
        return [0]
    for i,mol in enumerate(gen_fp):
        self_scores.append(1-max(DataStructs.BulkTanimotoSimilarity(mol, gen_fp[:i] + gen_fp[i+1 :])))
    return self_scores


if __name__=='__main__':
    # #rnn
    # check_novelty('D:\Projects\Cocrystal37pip\AAAI_code\GAN\CCDC_fine-tuning\data\database_CCDC_0.csv',
    #               'D:\Projects\Cocrystal37pip\AAAI_code\GAN\ChEMBL_training//rnn_data.csv',
    #               '0',
    #               '0')

    # check_novelty('D:\Projects\Cocrystal37pip\AAAI_code\GAN\CCDC_fine-tuning\data\database_CCDC_0.csv',
    #               'D:\Projects\Cocrystal37pip\molGCT//fine_tune\moses_bench2_lat=128_epo=1111111111111_k=4_20231218.csv',
    #               '0',
    #               'mol')

    # check_novelty('/CCDC_fine-tuning/data/database_CCDC_0.csv',
    #               'D:\Projects\Cocrystal37pip\molGCT//fine_tune//10k_moses_bench2_lat=128_epo=1111111111111_k=4_20231219.csv',
    #               '0',
    #               'mol')



    # check_novelty_mol_path('generative_models/GAN/CCDC_fine-tuning/data/database_CCDC.csv',["CS(=O)C1=CC(F)=CC(C2=CC=CC=C2)=C1","CC1=CC2=C(C=C1O)C(=O)C1=CC=CC=C12","O=C(NC1=CC=C([N+](=O)[O-])C=C1)C1=CC=C(C(F)(F)C2=CC=C(I)C=C2)C=C1","O=S(=O)(C1=CC=C(Cl)C=C1)C1=C(Cl)C=CC2=CC=CC=C12","N#CC1=CC([N+](=O)[O-])=CC([N+](=O)[O-])=C1[N+](=O)[O-]"],
    #               '0',
    #               'mol')
    check_novelty_chembl(smiles = ['CC(=O)O[C@@H]1[C@@H](OC(C)=O)/C(C)=C\[C@@H]2OC(=O)[C@]3(C)O[C@]23[C@H](OC(C)=O)[C@H]2[C@](C)(O)[C@H](O)C=C[C@]2(C)[C@H]1OC(C)=O','Cc1ccc(C(=O)c2ccc3c(c2)C(=O)c2c(O)cccc2C3)c(C)c1', 'Cc1ccccc1C(=O)c1ccccc1C(C)N', 'Cc1ccc(C(c2ccccc2)c2ccccc2)cc1'])