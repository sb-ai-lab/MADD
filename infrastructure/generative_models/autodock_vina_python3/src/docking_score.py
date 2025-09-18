import enum
from typing import List
from vina import Vina
import os
import sys
from generative_models.autodock_vina_python3.src.utils.paths import get_project_path
import numpy as np 
from generative_models.autodock_vina_python3.src.utils.prepare_receptor import prepare_pdb, get_center_of_protein, get_bounding_box
from generative_models.autodock_vina_python3.src.utils.prepare_ligand import preprocess_ligand
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

    center = get_center_of_protein(pdb_fie)


    path_receptor = os.path.join(os.getcwd(), 'receptor.pdbqt')
    path_ligand = os.path.join(os.getcwd(), 'ligand.pdbqt')
    v.set_receptor(path_receptor)
    v.set_ligand_from_file(path_ligand)

    
    if box_size is None:
        if padding is None:
            padding = 10
        box_size = get_bounding_box(pdb_fie)
        box_size = [size + padding for size in box_size]


    v.compute_vina_maps(center=center, box_size=box_size)

    v.dock(exhaustiveness=8, n_poses=n_poses)

    energy_minimized = v.optimize(max_steps=max_steps)
    print('Score after minimization : %.3f (kcal/mol)' % energy_minimized[0])
    os.system(f"rm {path_ligand} {path_receptor}")
    print('end end' % energy_minimized[0])
    return energy_minimized[0]


def docking_list(smiles:List[str],path_receptor_pdb:str):
    docking_scores = []

    for i,smile in enumerate(tqdm(smiles)):
        try:
            docking_scores.append(get_docking_score(pdb_fie=path_receptor_pdb, smiles_ligand=smile, n_poses=20, max_steps=20))
        except:
            docking_scores.append(1)
        # if i%10==0:
        #     db_dock = pd.DataFrame(data=docking_scores,columns=['docking_scores'])
        #     db_dock.to_csv('/projects/AAAI_code/generative_models/Sber_Alzheimer/autodock_vina_python3/chembl_34_chemreps_filtered_400w_150s_docking.csv')
    return docking_scores