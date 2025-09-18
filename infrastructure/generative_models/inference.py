import json
import sys
import os
import_path = os.path.dirname(os.path.abspath(__file__))
import statistics
import time
from Process import *
import argparse
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, QED, rdDepictor, AllChem, Draw
from Models import get_model
from Beam import beam_search
from torch.autograd import Variable
import joblib
import numpy as np
from rand_gen import rand_gen_from_data_distribution, tokenlen_gen_from_data_distribution
from dataDistibutionCheck import checkdata
from tqdm import tqdm
from utils.check_novelty import check_novelty_mol_path
import warnings
from rdkit import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from rdkit.Contrib.SA_Score import sascorer
import pickle as pi
import lightgbm
from autodock_vina_python3.src.docking_score import docking_list
import random
from utils.prop_calc import all_prop_calc_list
from typing import List
import requests

def gen_mol_val(cond, model, opt, SRC, TRG, toklen, z):
    model.eval()
    cond = Variable(torch.Tensor(cond))
    sentence = beam_search(cond, model, SRC, TRG, toklen, opt, z)
    return sentence

def generate_alzh(opt, model, SRC, TRG,n_samples=100,spec_conds=None,save=False,shift_path='',mean_=0,std_=1,
                  path_to_val_model:str='generative_models/Sber_Alzheimer/sber_altz/kinase_inhib.pkl'):
    #model_val = pi.load(open(path_to_val_model,'rb')) 
    warnings.filterwarnings('ignore')
    molecules, val_check, conds_trg, conds_rdkit, toklen_check, toklen_gen = [], [], [], [], [], []
    #print("\nGenerating molecules for MOSES benchmarking...")
    n_samples = n_samples
    toklen_data = pd.read_csv(f'{opt.load_weights}/toklen_list.csv')

    if spec_conds is None:
        conds = np.array([[np.random.randint(2)] for i in range(n_samples)])# for physics performances
    else:
        conds = np.array([list(spec_conds) for _ in range(n_samples)])

    toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max()-toklen_data.min()), size=n_samples)
    opt.conds = 'm'
    start = time.time()
    matched_cond =[]
    for idx in tqdm(range(n_samples)):
        toklen = int(toklen_data[idx]) + 3  # +3 due to cond2enc
        z = torch.Tensor(np.random.normal(loc=float(mean_),scale=float(std_),size=(1, toklen, opt.latent_dim)))
        molecule_tmp = gen_mol_val(conds[idx], model, opt, SRC, TRG, toklen, z)
        if molecule_tmp=='':
                continue
        if Chem.MolFromSmiles(str(molecule_tmp)) is None:
            continue
        toklen_gen.append(molecule_tmp.count(' ')+1)
        molecule_tmp = ''.join(molecule_tmp).replace(" ", "")
        matched_cond.append(conds[idx])
        molecules.append(molecule_tmp)
    
    conds = matched_cond

    df_molecules = pd.DataFrame(molecules)
    if save:
        df_molecules.drop_duplicates().to_csv(f'{opt.path_script}/mols_valid_mols_wo_dupls{shift_path}.csv')
        df_molecules.to_csv(f'{opt.path_script}/mols_generated_mols{shift_path}.csv')
    molecules = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else None for smi in molecules]
    df_init = pd.DataFrame(np.concatenate((conds, np.array(molecules).reshape(len(molecules), 1)), axis=1),columns= ['ic50', 0])
    generated_coformers_clear =[]
    for smiles_mol in molecules:
            if smiles_mol=='':
                continue
            if Chem.MolFromSmiles(str(smiles_mol)) is None:
                continue
            generated_coformers_clear.append(smiles_mol)
    if len(generated_coformers_clear)==0:
        print('No one valid molecules')
        return 0, 0, 0, 0, 0, 0
    ic50,clear_mols_ic_50 = list(opt.classifire(molecules))
    clf_df = pd.DataFrame(np.concatenate(( np.array(ic50).reshape(len(ic50), 1), np.array(clear_mols_ic_50).reshape(len(clear_mols_ic_50), 1)), axis=1),columns= ['ic50', 0])
    clf_df[0] = clf_df[0].apply(Chem.MolToSmiles)
    sa = []
    for mol in clf_df[0].tolist():
        sa.append(sascorer.calculateScore(Chem.MolFromSmiles(mol)))
    #df_to_save = clf_df#.copy(deep=True)
    clf_df['sa'] = sa
    if save:
        clf_df.drop_duplicates().to_csv(f'{opt.path_script}/valid_mols_wo_dupls{shift_path}.csv')
        df_init.to_csv(f'{opt.path_script}/generated_mols{shift_path}.csv')
    return clf_df.drop_duplicates()


def generate_alzh_for_agent(opt, model, SRC, TRG,n_samples=100,spec_conds=None,save=False,shift_path='',mean_=0,std_=1):
    warnings.filterwarnings('ignore')
    molecules, val_check, conds_trg, conds_rdkit, toklen_check, toklen_gen = [], [], [], [], [], []
    n_samples = n_samples
    toklen_data = pd.read_csv(f'{opt.load_weights}/toklen_list.csv')

    if spec_conds is None:
        conds = np.array([[np.random.randint(2)] for i in range(n_samples)])
    else:
        conds = np.array([[random.uniform(n[0],n[1]) for n in spec_conds] for _ in range(n_samples)])
    print(conds)
    toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max()-toklen_data.min()), size=n_samples)
    opt.conds = 'm'
    start = time.time()
    matched_cond =[]
    for idx in tqdm(range(n_samples)):
        toklen = int(toklen_data[idx]) + 3 
        z = torch.Tensor(np.random.normal(loc=float(mean_),scale=float(std_),size=(1, toklen, opt.latent_dim)))
        molecule_tmp = gen_mol_val(conds[idx], model, opt, SRC, TRG, toklen, z)
        if molecule_tmp=='':
                continue
        if Chem.MolFromSmiles(str(molecule_tmp)) is None:
            continue
        toklen_gen.append(molecule_tmp.count(' ')+1)
        molecule_tmp = ''.join(molecule_tmp).replace(" ", "")
        matched_cond.append(conds[idx])
        molecules.append(molecule_tmp)
    return molecules


def generate_auto(opt,
                   model,
                   SRC,
                   TRG,
                   state,
                   n_samples=100,
                   spec_conds=None,
                   case='Alzhmr'
                   ):
    
    warnings.filterwarnings('ignore')
    molecules, toklen_gen = [], []


    toklen_data = pd.read_csv(f"{opt.load_weights}/toklen_list.csv")
    conds = spec_conds

    toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max()-toklen_data.min()), size=n_samples)


    for idx in tqdm(range(n_samples)):
        toklen = int(toklen_data[idx]) + 3  # +3 due to cond2enc
        z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
        molecule_tmp = gen_mol_val(conds[idx], model, opt, SRC, TRG, toklen, z)
        toklen_gen.append(molecule_tmp.count(' ')+1)
        molecule_tmp = ''.join(molecule_tmp).replace(" ", "")

        molecules.append(molecule_tmp)
    valid_mols = state()["Calculateble properties"]['Validity'](molecules) #with duplicates
    
    unique_mols = set(valid_mols)
    DYPLICATES = 1-len(unique_mols)/len(valid_mols)
    VALID = len(valid_mols)/len(molecules)
    props_for_calc = [i for i in state.show_calculateble_propreties() if i !="Validity"]
    print(props_for_calc)
    props = {key:state()["Calculateble properties"][key](valid_mols) for key in props_for_calc}
    props['Validity'] = VALID
    props["Duplicates"] = DYPLICATES
    print(props)
    if state(case,'ml')['status'] == 'Trained':
        ml_props = predict_smiles(valid_mols,case,url=opt.url_ml_model)
        for key,value in ml_props.items():
            props[key]=value 
    df = pd.DataFrame(data = {'Smiles':valid_mols,**props})
    return df


def generate_for_agent(opt, model, SRC, TRG,n_samples=100,spec_conds=None,save=False,shift_path='',drug='CN1C2=C(C(=O)N(C1=O)C)NC=N2',mean_=0,std_=1):
    warnings.filterwarnings('ignore')
    molecules, val_check, conds_trg, conds_rdkit, toklen_check, toklen_gen = [], [], [], [], [], []
    #print("\nGenerating molecules for MOSES benchmarking...")
    n_samples = n_samples
    toklen_data = pd.read_csv(f'{opt.load_weights}/toklen_list.csv')

    if spec_conds is None:
        conds = np.array([[np.random.randint(2),np.random.randint(2),np.random.randint(2)] for i in range(n_samples)])# for physics performances
    else:
        conds = np.array([list(spec_conds) for _ in range(n_samples)])

    toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max()-toklen_data.min()), size=n_samples)
    opt.conds = 'm'
    start = time.time()
    for idx in tqdm(range(n_samples)):
        toklen = int(toklen_data[idx]) + 3  # +3 due to cond2enc
        z = torch.Tensor(np.random.normal(loc=float(mean_),scale=float(std_),size=(1, toklen, opt.latent_dim)))
        molecule_tmp = gen_mol_val(conds[idx], model, opt, SRC, TRG, toklen, z)
        toklen_gen.append(molecule_tmp.count(' ')+1)
        molecule_tmp = ''.join(molecule_tmp).replace(" ", "")

        molecules.append(molecule_tmp)
    
    return molecules


def validate_docking(opt, model, SRC, TRG,n_samples=100,spec_conds=None,save=False,shift_path='',
                          is_molecules=True,case='Alzhmr',
                          path_receptor_pdb='generative_models/Sber_Alzheimer/autodock_vina_python3/data/4j1r.pdb'):
    
    warnings.filterwarnings('ignore')
    molecules, val_check, conds_trg, conds_rdkit, toklen_check, toklen_gen = [], [], [], [], [], []
    #print("\nGenerating molecules for MOSES benchmarking...")
    n_samples = n_samples
    #nBins = [100, 100, 100]

    #data = pd.read_csv(opt.cond_train_path)
    toklen_data = pd.read_csv(f"{opt.path_script}/weights/toklen_list.csv")
    if spec_conds is None:
        conds = np.array([[np.random.randint(2)] for i in range(n_samples)])# for physics performances
    else:
        conds = np.array([list(spec_conds) for _ in range(n_samples)])
    #conds = np.array([[np.random.randint(2),np.random.randint(2),np.random.randint(2)] for i in range(n_samples)])# for physics performances
    #conds = rand_gen_from_data_distribution(data, size=n_samples, nBins=nBins)

    toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max()-toklen_data.min()), size=n_samples)
    opt.conds = 'm'
    start = time.time()
    for idx in tqdm(range(n_samples)):
        toklen = int(toklen_data[idx]) + 3  # +3 due to cond2enc
        z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
        molecule_tmp = gen_mol_val(conds[idx], model, opt, SRC, TRG, toklen, z)
        toklen_gen.append(molecule_tmp.count(' ')+1)
        molecule_tmp = ''.join(molecule_tmp).replace(" ", "")

        molecules.append(molecule_tmp)
    molecules = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else None for smi in molecules]
    df_init = pd.DataFrame(np.concatenate((conds, np.array(molecules).reshape(len(molecules), 1)), axis=1),columns= opt.conditions + [0])
    generated_coformers_clear =[]
    for smiles_mol in molecules:
            if smiles_mol=='':
                continue
            if Chem.MolFromSmiles(str(smiles_mol)) is None:
                continue

            generated_coformers_clear.append(smiles_mol)
    if len(generated_coformers_clear)==0:
        print('No one valid molecules')
        return 0, 0, 0, 0, 0, 0,0
    
    ic50, clear_mols_ic_50 = list(opt.classifire(generated_coformers_clear))
    if ic50 is None:
        print('No one valid molecules')
        return 0, 0, 0, 0, 0, 0,0
    
    docking = docking_list([Chem.MolToSmiles(i) for i in clear_mols_ic_50],path_receptor_pdb=path_receptor_pdb)
    print('docking scores: ',docking)
    mean_docking = sum(docking)/len(docking)
    max_docking, min_docking = max(docking),min(docking)
    clf_df = pd.DataFrame(np.concatenate(( np.array(ic50).reshape(len(ic50), 1), np.array(clear_mols_ic_50).reshape(len(clear_mols_ic_50), 1)), axis=1),columns= ['IC50', 0])
    clf_df[0] = clf_df[0].apply(Chem.MolToSmiles)
    sa = []
    for mol in clf_df[0].tolist():
        if mol != '' and mol is not None:
            sa.append(sascorer.calculateScore(Chem.MolFromSmiles(mol)))
        #sa.append(sascorer.calculateScore(Chem.MolFromSmiles(mol)))

    if save:
        df_to_save = clf_df.copy(deep=True)
        df_to_save['sa'] = sa
        df_to_save.drop_duplicates().to_csv(f'{opt.path_script}/valid_mols_wo_dupls{shift_path}.csv')
        df_init.to_csv(f'{opt.path_script}/generated_mols{shift_path}.csv')

    sa_total =pd.DataFrame(sa,columns=['mol'])
    df_total = df_init.merge(clf_df, right_on=0, left_on=0, how='right')
    if is_molecules:
        valid = sum([Chem.MolFromSmiles(i) is not None for i in df_init[0] if i is not None]) / n_samples
    else:
        reacts = []
        for i in df_init[0]:
            try:
                react_temp = AllChem.ReactionFromSmarts(i)
                reacts.append(True)
            except:
                reacts.append(False)
        valid = sum(reacts)/ n_samples


    novelty, duplicates = check_novelty_mol_path(train_dataset_path=opt.train_data_csv, gen_data=df_total[0].to_list(),
                                                 train_col_name='canonical_smiles', gen_col_name='mol')
    df_total = df_total.drop_duplicates()
    df_total = df_total[~df_total[0].isin(pd.read_csv(opt.train_data_csv)['canonical_smiles'])]

    #init_cond = np.int_(df_total['ic50_x'].to_numpy())
    generated_cond = np.float_(df_total['IC50_y'].to_numpy())

    df_total['IC50_y'] = df_total['IC50_y'].astype(float)
    #df_cond_comp - DF, that consist of molecules, with matched conditions performances
    mean_ic50 = generated_cond.mean()
    print('mean ic_50',mean_ic50)
    df_cond_comp = df_total[df_total['IC50_y']<=4]
    correctness = [abs(np.float32(i[0])-np.int_(i[1])) for i in zip(df_total['IC50_y'],df_total['IC50_x'])]#without duplicates and only valids
    sa_unique = []
    if len([i for i in df_total[df_total['IC50_y']<=4][0]]):
        for mol in [i for i in df_total[df_total['IC50_y']<=4][0]]:
            sa_unique.append(sascorer.calculateScore(Chem.MolFromSmiles(mol)))

    sa_cond_compared = pd.DataFrame(sa_unique, columns=['mol'])
    sa_comp_3 = (sa_cond_compared['mol']<=3).sum()
    of_ineterested_cond = sa_comp_3/len(df_init) #How many molecules have SA<=3 of mol-s,
                                                            # that have ineterested physics performance
    of_total_generated = sa_comp_3/n_samples
    total_sa_3 = (sa_total['mol'] <= 3).sum()
    if correctness == []:
        cond_score = 0
    else:
        cond_score = mean_ic50

    ###
    list_smi = df_cond_comp[0].tolist()
    fpgen = AllChem.GetRDKitFPGenerator()
    df_cond_comp['mol'] = df_cond_comp[0].apply(Chem.MolFromSmiles)
    fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(x)) for x in list_smi]

    def check_diversity(mol):
        fp = fpgen.GetFingerprint(mol)
        scores = DataStructs.BulkTanimotoSimilarity(fp, fps)
        return statistics.mean(scores)

    df_cond_comp['diversity'] = df_cond_comp.mol.apply(check_diversity)
    ###
    props_8 = [list(i) for i in all_prop_calc_list(list(df_total[0]),case=case)]
    props_8 = [[sum(list(i))/len(list(i)),max(list(i)),min(list(i))] for i in props_8]
    print('Condition correct score (how many conditions matched):', cond_score, 'Total valid mol:', valid,
          'Molecules of spec cond with sa<=3:', of_ineterested_cond, ';Matched conds', len(df_cond_comp),
          'Spec cond:', spec_conds, 'Mean_deversity:', df_cond_comp['diversity'].mean(), ';Sum intersting molecules:',
          sa_comp_3,'Docking score is (mean,max,min): ',mean_docking,max_docking,min_docking)
    print('Special conditions is: ','QED',props_8[0],'Synthetic Accessibility',props_8[1],'PAINS',props_8[2],'SureChEMBL',props_8[3],'Glaxo',props_8[4],'Brenk',props_8[5],'IC50',props_8[6])
    if props_8==0:
        return 0, valid, novelty, duplicates, of_ineterested_cond, df_cond_comp['diversity'].mean(), {'QED':0,'Synthetic Accessibility':0,'PAINS':0,'SureChEMBL':0,'Glaxo':0,'Brenk':0,'IC50':0}
    else:
        return 0, valid, novelty, duplicates, of_ineterested_cond, df_cond_comp['diversity'].mean(), {'QED':props_8[0],'Synthetic Accessibility':props_8[1],'PAINS':props_8[2],'SureChEMBL':props_8[3],'Glaxo':props_8[4],'Brenk':props_8[5],'IC50':props_8[6]}

def predict_smiles(smiles_list : List[str],
                   case : str = "Brain_cancer",
                   timeout : int = 10, #30 min
                   url: str = "http://10.64.4.243:81",
                   **kwargs,):
    url = url+"/predict_ml"
    params = {
        'case': case,
        'smiles_list' : smiles_list,
        'timeout': timeout,
        **kwargs,
    }
    resp = requests.post(url, json.dumps(params))
    return resp.json()

def generate_auto(opt,
                   model,
                   SRC,
                   TRG,
                   state,
                   n_samples=100,
                   spec_conds=None,
                   case='Alzhmr'
                   ):
    
    warnings.filterwarnings('ignore')
    molecules, toklen_gen = [], []

    if spec_conds is None:
        conds = np.array([[np.random.randint(2),np.random.randint(2),np.random.randint(2)] for i in range(n_samples)])# for physics performances
    else:
        conds = np.array([list(spec_conds) for _ in range(n_samples)])
    toklen_data = pd.read_csv(f"{opt.load_weights}/toklen_list.csv")

    toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max()-toklen_data.min()), size=n_samples)


    for idx in tqdm(range(n_samples)):
        toklen = int(toklen_data[idx])  # +3 due to cond2enc
        z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
        molecule_tmp = gen_mol_val(conds[idx], model, opt, SRC, TRG, toklen, z)
        toklen_gen.append(molecule_tmp.count(' ')+1)
        molecule_tmp = ''.join(molecule_tmp).replace(" ", "")

        molecules.append(molecule_tmp)
    valid_mols = state()["Calculateble properties"]['Validity'](molecules)
    break_point =3 #with duplicates
    while len(valid_mols)==0:
        for idx in tqdm(range(n_samples)):
            toklen = int(toklen_data[idx])  # +3 due to cond2enc
            z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
            molecule_tmp = gen_mol_val(conds[idx], model, opt, SRC, TRG, toklen, z)
            toklen_gen.append(molecule_tmp.count(' ')+1)
            molecule_tmp = ''.join(molecule_tmp).replace(" ", "")
            molecules.append(molecule_tmp)
        break_point-=1
        valid_mols = state()["Calculateble properties"]['Validity'](molecules)
        if break_point<1:
            print('Model Cant generate valid molecules! Retrain it for this case!')
            return {'Smiles':'NO VALID'}

    unique_mols = set(valid_mols)
    DYPLICATES = 1-len(unique_mols)/len(valid_mols)
    VALID = len(valid_mols)/len(molecules)
    props_for_calc = [i for i in state.show_calculateble_propreties() if i !="Validity"]
    print(props_for_calc)
    props = {key:state()["Calculateble properties"][key](valid_mols) for key in props_for_calc}
    props['Validity'] = VALID
    props["Duplicates"] = DYPLICATES
    print(props)
    if state(case,'ml')['status'] == 'Trained':
        ml_props = predict_smiles(valid_mols,case,url=opt.url_ml_model)
        for key,value in ml_props.items():
            props[key]=value 
    df = pd.DataFrame(data = {'Smiles':valid_mols,**props})
    return df.to_dict()

def validate_props(opt,
                   model,
                   SRC,
                   TRG,
                   state,
                   n_samples=100,
                   spec_conds=None,
                   case='Alzhmr'
                   ):
    
    warnings.filterwarnings('ignore')
    molecules, toklen_gen = [], []


    toklen_data = pd.read_csv(f"{opt.save_folder_name}/weights/toklen_list.csv")
    conds = np.array([list(spec_conds) for _ in range(n_samples)])

    toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max()-toklen_data.min()), size=n_samples)


    for idx in tqdm(range(n_samples)):
        toklen = int(toklen_data[idx]) + opt.cond_dim  # +3 due to cond2enc
        z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
        molecule_tmp = gen_mol_val(conds[idx], model, opt, SRC, TRG, toklen, z)
        toklen_gen.append(molecule_tmp.count(' ')+1)
        molecule_tmp = ''.join(molecule_tmp).replace(" ", "")

        molecules.append(molecule_tmp)
        #molecules = ["CCCCCCNc1n[n+]([O-])c2cc(C)ccc2[n+]1[O-]",'COc1nc(OC)nc(Oc2cccc(Br)c2)n1']

    valid_mols = state()["Calculateble properties"]['Validity'](molecules) #with duplicates
    
    if len(valid_mols)==0:
        print('No one valid molecules')
        return {'Validity':0,"Duplicates":100}
    unique_mols = set(valid_mols)
    DYPLICATES = 1-len(unique_mols)/len(valid_mols)
    VALID = len(valid_mols)/len(molecules)
    props_for_calc = [i for i in state.show_calculateble_propreties() if i !="Validity"]
    #props_for_calc = [i for i in state(case,'ml')['target_column'] if i !="Validity"]
    props = {key:state()["Calculateble properties"][key](valid_mols) for key in props_for_calc}
    props['Validity'] = VALID
    props["Duplicates"] = DYPLICATES

    if state(case,'ml')['status'] == 'Trained':
        ml_props = predict_smiles(valid_mols,case,url=opt.url_ml_model)
        for key,value in ml_props.items():
            props[key]=value 
    return props

    # molecules = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else None for smi in molecules]
    # df_init = pd.DataFrame(np.concatenate((conds, np.array(molecules).reshape(len(molecules), 1)), axis=1),columns= opt.conditions + [0])
    # generated_coformers_clear =[]
    # for smiles_mol in molecules:
    #         if smiles_mol=='':
    #             continue
    #         if Chem.MolFromSmiles(str(smiles_mol)) is None:
    #             continue

    #         generated_coformers_clear.append(smiles_mol)
    # if len(generated_coformers_clear)==0:
    #     print('No one valid molecules')
    #     return 0, 0, 0, 0, 0, 0,0
    
    # ic50, clear_mols_ic_50 = list(opt.classifire(generated_coformers_clear))
    # if ic50 is None:
    #     print('No one valid molecules')
    #     return 0, 0, 0, 0, 0, 0,0
    
    # # docking = docking_list([Chem.MolToSmiles(i) for i in clear_mols_ic_50],path_receptor_pdb=path_receptor_pdb)
    # # print('docking scores: ',docking)
    # # mean_docking = sum(docking)/len(docking)
    # # max_docking, min_docking = max(docking),min(docking)
    # clf_df = pd.DataFrame(np.concatenate(( np.array(ic50).reshape(len(ic50), 1), np.array(clear_mols_ic_50).reshape(len(clear_mols_ic_50), 1)), axis=1),columns= ['IC50', 0])
    # clf_df[0] = clf_df[0].apply(Chem.MolToSmiles)
    # sa = []
    # for mol in clf_df[0].tolist():
    #     if mol != '' and mol is not None:
    #         sa.append(sascorer.calculateScore(Chem.MolFromSmiles(mol)))
    #     #sa.append(sascorer.calculateScore(Chem.MolFromSmiles(mol)))

    # if save:
    #     df_to_save = clf_df.copy(deep=True)
    #     df_to_save['sa'] = sa
    #     df_to_save.drop_duplicates().to_csv(f'{opt.path_script}/valid_mols_wo_dupls{shift_path}.csv')
    #     df_init.to_csv(f'{opt.path_script}/generated_mols{shift_path}.csv')

    # sa_total =pd.DataFrame(sa,columns=['mol'])
    # df_total = df_init.merge(clf_df, right_on=0, left_on=0, how='right')
    # if is_molecules:
    #     valid = sum([Chem.MolFromSmiles(i) is not None for i in df_init[0] if i is not None]) / n_samples
    # else:
    #     reacts = []
    #     for i in df_init[0]:
    #         try:
    #             react_temp = AllChem.ReactionFromSmarts(i)
    #             reacts.append(True)
    #         except:
    #             reacts.append(False)
    #     valid = sum(reacts)/ n_samples


    # novelty, duplicates = check_novelty_mol_path(train_dataset_path=opt.train_data_csv, gen_data=df_total[0].to_list(),
    #                                              train_col_name='canonical_smiles', gen_col_name='mol')
    # df_total = df_total.drop_duplicates()
    # df_total = df_total[~df_total[0].isin(pd.read_csv(opt.train_data_csv)['canonical_smiles'])]

    # #init_cond = np.int_(df_total['ic50_x'].to_numpy())
    # generated_cond = np.float_(df_total['IC50_y'].to_numpy())

    # df_total['IC50_y'] = df_total['IC50_y'].astype(float)
    # #df_cond_comp - DF, that consist of molecules, with matched conditions performances
    # mean_ic50 = generated_cond.mean()
    # print('mean ic_50',mean_ic50)
    # df_cond_comp = df_total[df_total['IC50_y']<=4]
    # correctness = [abs(np.float32(i[0])-np.int_(i[1])) for i in zip(df_total['IC50_y'],df_total['IC50_x'])]#without duplicates and only valids
    # sa_unique = []
    # if len([i for i in df_total[df_total['IC50_y']<=4][0]]):
    #     for mol in [i for i in df_total[df_total['IC50_y']<=4][0]]:
    #         sa_unique.append(sascorer.calculateScore(Chem.MolFromSmiles(mol)))

    # sa_cond_compared = pd.DataFrame(sa_unique, columns=['mol'])
    # sa_comp_3 = (sa_cond_compared['mol']<=3).sum()
    # of_ineterested_cond = sa_comp_3/len(df_init) #How many molecules have SA<=3 of mol-s,
    #                                                         # that have ineterested physics performance
    # of_total_generated = sa_comp_3/n_samples
    # total_sa_3 = (sa_total['mol'] <= 3).sum()
    # if correctness == []:
    #     cond_score = 0
    # else:
    #     cond_score = mean_ic50

    # ###
    # list_smi = df_cond_comp[0].tolist()
    # fpgen = AllChem.GetRDKitFPGenerator()
    # df_cond_comp['mol'] = df_cond_comp[0].apply(Chem.MolFromSmiles)
    # fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(x)) for x in list_smi]

    # def check_diversity(mol):
    #     fp = fpgen.GetFingerprint(mol)
    #     scores = DataStructs.BulkTanimotoSimilarity(fp, fps)
    #     return statistics.mean(scores)

    # df_cond_comp['diversity'] = df_cond_comp.mol.apply(check_diversity)
    # ###
    # props_8 = [list(i) for i in all_prop_calc_list(list(df_total[0]),case=case)]
    # props_8 = [[sum(list(i))/len(list(i)),max(list(i)),min(list(i))] for i in props_8]
    # print('Condition correct score (how many conditions matched):', cond_score, 'Total valid mol:', valid,
    #       'Molecules of spec cond with sa<=3:', of_ineterested_cond, ';Matched conds', len(df_cond_comp),
    #       'Spec cond:', spec_conds, 'Mean_deversity:', df_cond_comp['diversity'].mean(), ';Sum intersting molecules:',
    #       sa_comp_3,'Docking score is (mean,max,min): ',mean_docking,max_docking,min_docking)
    # print('Special conditions is: ','QED',props_8[0],'Synthetic Accessibility',props_8[1],'PAINS',props_8[2],'SureChEMBL',props_8[3],'Glaxo',props_8[4],'Brenk',props_8[5],'IC50',props_8[6])
    # if props_8==0:
    #     return 0, valid, novelty, duplicates, of_ineterested_cond, df_cond_comp['diversity'].mean(), {'QED':0,'Synthetic Accessibility':0,'PAINS':0,'SureChEMBL':0,'Glaxo':0,'Brenk':0,'IC50':0}
    # else:
    #     return 0, valid, novelty, duplicates, of_ineterested_cond, df_cond_comp['diversity'].mean(), {'QED':props_8[0],'Synthetic Accessibility':props_8[1],'PAINS':props_8[2],'SureChEMBL':props_8[3],'Glaxo':props_8[4],'Brenk':props_8[5],'IC50':props_8[6]}