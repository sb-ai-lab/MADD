import os

import_path = os.path.dirname(os.path.abspath(__file__))
import argparse
def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights_fields', type=str, default=True)
    parser.add_argument('-new_vocab', type=str, default=False)
    parser.add_argument('-load_weights', type=str, default='')
    #parser.add_argument('-load_traindata', type=str, default="data/moses/prop_temp.csv")
    parser.add_argument('-load_toklendata', type=str, default='toklen_list.csv')
    parser.add_argument('-k', type=int, default=4)
    parser.add_argument('-lang_format', type=str, default='SMILES')
    parser.add_argument('-max_strlen', type=int, default=100)  # max 80
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)

    parser.add_argument('-use_cond2dec', type=bool, default=False)
    parser.add_argument('-use_cond2lat', type=bool, default=True)
    parser.add_argument('-cond_dim', type=int, default=3)
    parser.add_argument('-latent_dim', type=int, default=128)

    parser.add_argument('-epochs', type=int, default=30)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-lr_beta1', type=int, default=0.9)
    parser.add_argument('-lr_beta2', type=int, default=0.98)

    parser.add_argument('-print_model', type=bool, default=False)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-cuda', type=str, default=False)
    parser.add_argument('-floyd', action='store_true')
    data_path = import_path + '/../../GAN'
    parser.add_argument('-cond_test_path', type=str,
                        default=f'{data_path}/CCDC_fine-tuning/data/physic/conditions/database_CCDC_cond_test.csv')
    parser.add_argument('-cond_train_path', type=str,
                        default=f'{data_path}/CCDC_fine-tuning/data/physic/conditions/database_CCDC_cond_train.csv')
    parser.add_argument('-train_data_csv', type=str,
                        default=f'{data_path}/CCDC_fine-tuning/data/physic/molecules/database_CCDC_train.csv')  # Need to check novelty of generated mols from train data
    opt = parser.parse_args()
    return opt