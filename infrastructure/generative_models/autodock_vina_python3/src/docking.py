import enum
from vina import Vina
import os
import sys

from generative_models.transformer.autodock_vina_python3.src.utils.paths import get_project_path
import numpy as np 
from generative_models.transformer.autodock_vina_python3.src.utils.prepare_receptor import prepare_pdb, get_center_of_protein, get_bounding_box
from generative_models.transformer.autodock_vina_python3.src.utils.prepare_ligand import preprocess_ligand
import pandas as pd
from tqdm import tqdm
# Ligand SMILES (Apigenin)
ligand_smiles = 'C1=CC(=CC=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O'
# Path to PDB for receptor (A2B1) 
path_receptor_pdb = os.path.join(get_project_path(), 'data', '4j1r.pdb')



def get_docking_score(pdb_fie: str, smiles_ligand: str, n_poses=100, max_steps=100, box_size=None, padding=None):

    preprocess_ligand(smiles_ligand)
    prepare_pdb(path_to_receptor=pdb_fie)

    v = Vina(sf_name='vina')

    center = get_center_of_protein(path_receptor_pdb)


    path_receptor = os.path.join(os.getcwd(), 'receptor.pdbqt')
    path_ligand = os.path.join(os.getcwd(), 'ligand.pdbqt')
    v.set_receptor(path_receptor)
    v.set_ligand_from_file(path_ligand)

    
    if box_size is None:
        if padding is None:
            padding = 10
        box_size = get_bounding_box(path_receptor_pdb)
        box_size = [size + padding for size in box_size]


    v.compute_vina_maps(center=center, box_size=box_size)

    v.dock(exhaustiveness=8, n_poses=n_poses)

    energy_minimized = v.optimize(max_steps=max_steps)
    print('Score after minimization : %.3f (kcal/mol)' % energy_minimized[0])
    os.system(f"rm {path_ligand} {path_receptor}")
    print('end end' % energy_minimized[0])
    return energy_minimized[0]
path = '/projects/AAAI_code/generative_models/Sber_Alzheimer/train_cVAE_sber_altz_docking/alzh_gen_mols/valid_mols_wo_dupls0_mp_ic50_6.csv'

db = pd.read_csv(path)
smiles = list(db['0'])[::-1]

docking_scores = []

for i,smile in enumerate(tqdm(smiles)):
    try:
        docking_scores.append(get_docking_score(pdb_fie=path_receptor_pdb, smiles_ligand=smile, n_poses=20, max_steps=20))
    except:
        docking_scores.append(1000)
    if i%10==0:
        db_dock = pd.DataFrame(data=docking_scores,columns=['docking_scores'])
        db_dock.to_csv('/projects/AAAI_code/generative_models/Sber_Alzheimer/train_cVAE_sber_altz_docking/alzh_gen_mols/valid_mols_wo_dupls0_mp_ic50_6_docking.csv')
db['docking_scores'] = docking_scores
db.to_csv(path)




# print('dock_score:', get_docking_score(pdb_fie=path_receptor_pdb, smiles_ligand=ligand_smiles, n_poses=20, max_steps=20))
# print('dock_score:', get_docking_score(pdb_fie=path_receptor_pdb, smiles_ligand='C=CCSS/C=C\C[S+]([O-])CCC', n_poses=20, max_steps=20))