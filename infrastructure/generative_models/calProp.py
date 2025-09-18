import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import pandas as pd


def printProgressBar(i,max,postText):
    n_bar = 20 #size of progress bar
    j= i/max
    sys.stdout.write('\r')
    sys.stdout.write(f"  [{'#' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}")
    sys.stdout.flush()

def calcProperty(opt):
    data = [opt.src_data, opt.src_data_te]
    for data_kind in data:
        if data_kind == opt.src_data:
            print("Calculating properties for {} train molecules: logP, tPSA, QED".format(len(opt.src_data)))
        if data_kind == opt.src_data_te:
            print("Calculating properties for {} test molecules: logP, tPSA, QED".format(len(opt.src_data_te)))
        count = 0
        mol_list, logP_list, tPSA_list, QED_list = [], [], [], []

        for smi in opt.src_data:
            count += 1
            printProgressBar(int(count / len(opt.src_data) * 100), 100, 'completed!')
            mol = Chem.MolFromSmiles(smi)
            mol_list.append(smi), logP_list.append(Descriptors.MolLogP(mol)), tPSA_list.append(Descriptors.TPSA(mol)), QED_list.append(QED.qed(mol))

        if data_kind == opt.src_data:
            prop_df = pd.DataFrame({'logP': logP_list, 'tPSA': tPSA_list, 'QED': QED_list})
            prop_df.to_csv("data/moses/prop_temp.csv", index=False)
        if data_kind == opt.src_data_te:
            prop_df_te = pd.DataFrame({'logP': logP_list, 'tPSA': tPSA_list, 'QED': QED_list})
            prop_df_te.to_csv("data/moses/prop_temp_te.csv", index=False)

    return prop_df, prop_df_te