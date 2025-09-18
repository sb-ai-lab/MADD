from openbabel import pybel
import subprocess
from Bio.PDB import PDBParser
import numpy as np
import os

def prepare_pdb(path_to_receptor):
    cmd = f"obabel {path_to_receptor} -xr -O {os.path.join(os.getcwd(), 'receptor.pdbqt')}"
    subprocess.run(cmd, shell=True, text=True, capture_output=True)

def get_center_of_protein(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)

    atoms = [atom for atom in structure.get_atoms()]
    atom_coords = np.array([atom.coord for atom in atoms])
    
    center = atom_coords.mean(axis=0).tolist()
    
    return center

def get_bounding_box(pdb_path):
    """
    Calculates the bounding box for a PDB structure.
    
    Args:
        pdb_path (str): Path to the PDB file.
    
    Returns:
        box_size (list): The box size as [x_size, y_size, z_size].
        center (tuple): The geometric center of the box as (x_center, y_center, z_center).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    all_coords = []
    
    for atom in structure.get_atoms():
        all_coords.append(atom.get_coord())
    
    all_coords = np.array(all_coords)
    
    min_coords = np.min(all_coords, axis=0)
    max_coords = np.max(all_coords, axis=0)
    
    box_size = max_coords - min_coords
    
    return box_size.tolist()
