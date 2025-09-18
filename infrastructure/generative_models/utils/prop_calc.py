from autodock_vina_python3.src.docking_score import docking_list
from utils.validation import check_chem_valid, eval_P_S_G, eval_qed, eval_sa, check_brenk
import lightgbm
from utils.check_novelty import check_novelty_chembl
from utils.ic_50_models.alzheimer.predict_ic50_clf import eval_ic_50_alzheimer
from utils.ic_50_models.skleroz_ic50_clf.scripts.predict_ic50_btk_clf import eval_ic_50_sklrz
from utils.ic_50_models.kras_ic50_prediction.predict_ic50_clf import eval_ic_50_cancer
from utils.ic_50_models.catechol_classif_inference.inference_catechol_clf import predict as parkenson_predict_ic50
from utils.ic_50_models.drug_resis_classif_inference.inference_drug_clf import predict as drug_res_predict_ic50
from utils.ic_50_models.tyrosine_classif_inference.inference_tyrosine_clf import predict as dyslip_predict_ic50
from utils.ki_models.tyrosine_regression_inference.tyrosine_inference_regr import predict as dyslip_predict_ki
import pandas as pd
from utils.inference_BB_clf.BB_inference import predict as eval_bbb
from tqdm import tqdm
import ast

cases = {'Alzhmr' : 
         {'docking_path' : 'generative_models/transformer/autodock_vina_python3/data/4j1r.pdb',
        'IC50':eval_ic_50_alzheimer,
         'KI':None,
         'anti_docking_path':None
                     },
        #TODO update next cases
        'Sklrz':
        {'docking_path' :'generative_models/transformer/autodock_vina_python3/data/skleroz/target_BTK.pdb',

         'IC50':eval_ic_50_sklrz,
         'KI':None,
         'anti_docking_path':None#{'BMX':'generative_models/Transformer/autodock_vina_python3/data/skleroz/BMX_8x2a_protein.pdb'}
         },

         'Prkns':
        {'docking_path' :'generative_models/transformer/autodock_vina_python3/data/parkinson/Catechol_O_methyltransferase.pdb',

         'IC50':parkenson_predict_ic50,
         'KI':None,
         'anti_docking_path':None
         },

         'Cnsr':
        {'docking_path' :'generative_models/transformer/autodock_vina_python3/data/Canser/8afb_protein.pdb',

         'IC50':eval_ic_50_cancer,
         'KI':None,
         'anti_docking_path':None#{'NRAS':'generative_models/Transformer/autodock_vina_python3/data/Canser/NRAS_3con_protein.pdb',
                              #'HRAS':'generative_models/Transformer/autodock_vina_python3/data/Canser/HRAS_3k8y_protein.pdb'}
         },

         'Dslpdm':
        {'docking_path' :'generative_models/transformer/autodock_vina_python3/data/dislipidemia/Proprotein_Convertase_SubtilisinKexin_Type_9.pdb',

         'IC50':dyslip_predict_ic50,
         'KI':dyslip_predict_ki,
         'anti_docking_path':None
         },

         'TBLET':
        {'docking_path' :'generative_models/transformer/autodock_vina_python3/data/Signal_Transducer_and_Activator_of_Transcription_3.pdb',

         'IC50':drug_res_predict_ic50,
         'KI':None,
         'anti_docking_path':None
         },
         'RNDM':
         {'docking_path' :'generative_models/transformer/autodock_vina_python3/data/target_BTK.pdb',

         'IC50':None,
         'KI':None,
         'anti_docking_path':None
         },
        }

def join_tables(path_to_first:str,
                path_to_second:str,
                columns:list):
    df_first = pd.read_csv(path_to_first)
    df_second = pd.read_csv(path_to_second)
    df_total = df_second.join(df_first,how='left',rsuffix='_').drop(f'{columns}_',axis=1)
    new_path = path_to_second.split(sep='.')[0]
    df_total.to_csv(f'{new_path}_props_total.csv')

def calc_metrics(path:str,
                 c_props = ['DockingScoreProperty', 'IC50', 'Synthetic Accessibility','Brenk',   'PAINS',
       'SureChEMBL', 'Glaxo','QED',  ]):
    df = pd.read_csv(path)
    df = df.dropna()
    length = len(df)
    try:
        diversity = df['Diversity'].mean()
        print('diversity',diversity)
    except:
        pass
    try:
        size = sum(df['size'])
        print('size',size)
    except:
        pass
    #GR1
    df = df[df[c_props[0]]<=-7]
    df = df[df[c_props[1]]==1]
    try:
        GR1 = sum(df['size'])
    except:
        GR1 = len(df)
    #GR2
    df = df[df[c_props[2]]<=3]
    try:
        GR2 = sum(df['size'])
    except:
        GR2 = len(df)
    #GR3
    df = df[df[c_props[3]]==0]
    try:
        GR3 = sum(df['size'])
    except:
        GR3 = len(df)
    #GR4
    df = df[(df[c_props[4]]==0) & (df[c_props[5]]==0) & (df[c_props[6]]==0)]
    try:
        GR4 = sum(df['size'])
    except:
        GR4 = len(df)
    #GR5

    qed_075_tot = df[df[c_props[7]]>0.6]
    try:
        GR5 = sum(df['size'])
    except:
        GR5 = len(df)

    
    print('GR1:',GR1,'GR2:',GR2,'GR3:',GR3,'GR4:',GR4,'GR5:',GR5,'length',length)

def all_prop_calc_list(data:list,
                   case:str='Alzhmr'):
    data_mol_col_name = 'Smiles'
    df = pd.DataFrame(data=data,columns=[data_mol_col_name])
    
    dis_case = cases[case]
    docking_path = dis_case['docking_path']
    QED,Synthetic,PAINS,SureChEMBL,Glaxo,Brenk,BBB,IC50,d_s = [],[],[],[],[],[],[],[],[]

    QED+=list(df[data_mol_col_name].apply(eval_qed))
    Synthetic+=list(df[data_mol_col_name].apply(eval_sa)) # SA
    PAINS+=list(df[data_mol_col_name].apply(eval_P_S_G,type_n='PAINS'))
    SureChEMBL+=list(df[data_mol_col_name].apply(eval_P_S_G,type_n='SureChEMBL'))
    Glaxo+=list(df[data_mol_col_name].apply(eval_P_S_G,type_n='Glaxo'))
    #df['Diversity'] = diversity
    Brenk+=list(df[data_mol_col_name].apply(check_brenk))
    #BBB+=list(eval_bbb(list(df[data_mol_col_name])))
    d_s=docking_list(smiles=df[data_mol_col_name],path_receptor_pdb=docking_path)
    ic50_ki_values = dis_case['IC50'](df[data_mol_col_name])
    IC50+=list(ic50_ki_values)
    
    return QED,Synthetic,PAINS,SureChEMBL,Glaxo,Brenk,IC50,d_s

def all_prop_calc(path_to_data:str,
                   data_mol_col_name:str,
                   case:str='Alzhmr',
                   part = 50,
                   calc_docking = False,
                   gen_data = True,
                   train_data_path:str='generative_models/transformer/docked_data_4j1r.csv',
                   calc_ic50 = True
                   ):
    
    df_init = pd.read_csv(path_to_data)

    if gen_data:
        valid_mols = check_chem_valid(df_init[data_mol_col_name])
        valid_mols,diversity = check_novelty_chembl(valid_mols,train_data_path=train_data_path)
    else:
        valid_mols = df_init[data_mol_col_name]
    df_init = pd.DataFrame(data=valid_mols,columns=[data_mol_col_name])
    df_init['Diversity'] = diversity
    print(len(df_init[data_mol_col_name])//part*part)
    df_init= df_init[:len(df_init[data_mol_col_name])//part*part]
    dis_case = cases[case]
    docking_path = dis_case['docking_path']
    QED,Synthetic,PAINS,SureChEMBL,Glaxo,Brenk,BBB,IC50,d_s = [],[],[],[],[],[],[],[],[]
    for i in tqdm(range(len(df_init[data_mol_col_name])//part)):
       
        df = df_init[i*part:(i+1)*part]
        if calc_docking:
            d_s+=docking_list(smiles=df[data_mol_col_name],path_receptor_pdb=docking_path)
        QED+=list(df[data_mol_col_name].apply(eval_qed))
        Synthetic+=list(df[data_mol_col_name].apply(eval_sa)) # SA
        PAINS+=list(df[data_mol_col_name].apply(eval_P_S_G,type_n='PAINS'))
        SureChEMBL+=list(df[data_mol_col_name].apply(eval_P_S_G,type_n='SureChEMBL'))
        Glaxo+=list(df[data_mol_col_name].apply(eval_P_S_G,type_n='Glaxo'))
        
        Brenk+=list(df[data_mol_col_name].apply(check_brenk))
        #BBB+=list(eval_bbb(list(df[data_mol_col_name])))
        if calc_ic50:
            ic50_ki_values = dis_case['IC50'](df[data_mol_col_name])
            IC50+=list(ic50_ki_values)
    
    df_init['QED'],df_init['Synthetic Accessibility'],df_init['PAINS'],df_init['SureChEMBL'], df_init['Glaxo'],df_init['Brenk']=QED,Synthetic,PAINS,SureChEMBL,Glaxo,Brenk
    if calc_ic50:
        df_init['IC50']  = IC50
    if calc_docking:
        df_init['Docking Score'] = d_s 
    new_path = path_to_data.split(sep='.')[0]
    df_init.to_csv(f'{new_path}_props.csv')

def all_prop_calc_filtered(path_to_data:str,
                   data_mol_col_name:str,
                   case:str='Alzhmr',
                   part = 50,
                   calc_docking = False,
                   gen_data = True,
                   train_data_path:str='generative_models/Transformer/docked_data_4j1r.csv'
                   ):
    
    df_init = pd.read_csv(path_to_data)

    if gen_data:
        valid_mols = check_chem_valid(df_init[data_mol_col_name])
        valid_mols,diversity = check_novelty_chembl(valid_mols,train_data_path=train_data_path)
    else:
        valid_mols = df_init[data_mol_col_name]
    
    df_init = pd.DataFrame(data=valid_mols,columns=[data_mol_col_name])
    df_init['Diversity'] = diversity
    print(len(df_init[data_mol_col_name])//part*part)
    df_init= df_init[:len(df_init[data_mol_col_name])//part*part]
    dis_case = cases[case]
    docking_path = dis_case['docking_path']
    QED,Synthetic,PAINS,SureChEMBL,Glaxo,Brenk,BBB,IC50,d_s = [],[],[],[],[],[],[],[],[]
    
    for i in tqdm(range(len(df_init[data_mol_col_name])//part)):
       
        df = df_init[i*part:(i+1)*part]
        ic50_ki_values = dis_case['IC50'](df[data_mol_col_name])
        IC50+=list(ic50_ki_values)

    df_init['IC50'] = IC50
    df_init = df_init[df_init['IC50']==1]
    df_init= df_init[:len(df_init[data_mol_col_name])//part*part]
    for i in tqdm(range(len(df_init[data_mol_col_name])//part)):
       
        df = df_init[i*part:(i+1)*part]
        QED+=list(df[data_mol_col_name].apply(eval_qed))
        Synthetic+=list(df[data_mol_col_name].apply(eval_sa)) # SA
        PAINS+=list(df[data_mol_col_name].apply(eval_P_S_G,type_n='PAINS'))
        SureChEMBL+=list(df[data_mol_col_name].apply(eval_P_S_G,type_n='SureChEMBL'))
        Glaxo+=list(df[data_mol_col_name].apply(eval_P_S_G,type_n='Glaxo'))
        Brenk+=list(df[data_mol_col_name].apply(check_brenk))

    df_init['QED'],df_init['Synthetic Accessibility'],df_init['PAINS'],df_init['SureChEMBL'], df_init['Glaxo'],df_init['Brenk'] =QED,Synthetic,PAINS,SureChEMBL,Glaxo,Brenk
    #Filetring
    #df_init = df_init[(df_init['IC50']==1)&(df_init['Synthetic Accessibility']<=3)&(df_init['Brenk']==0)]
    df_init= df_init[:len(df_init[data_mol_col_name])//part*part]
    for i in tqdm(range(len(df_init[data_mol_col_name])//part)):
        df = df_init[i*part:(i+1)*part]
        if calc_docking:
            d_s+=docking_list(smiles=df[data_mol_col_name],path_receptor_pdb=docking_path)
    if calc_docking:
        df_init['Docking Score'] = d_s 
    new_path = path_to_data.split(sep='.')[0]
    df_init.to_csv(f'{new_path}_props.csv')

def all_docking(path_to_data:str,
                   data_mol_col_name:str,
                   case:str='Alzhmr',
                   part = 50,
                   calc_docking = False,
                   gen_data = True,
                   train_data_path:str='generative_models/Transformer/docked_data_4j1r.csv'
                   ):
    
    df_init = pd.read_csv(path_to_data)


    df_init= df_init[:len(df_init[data_mol_col_name])//part*part]
    dis_case = cases[case]
    docking_path = dis_case['docking_path']
    d_s = []
    
    for i in tqdm(range(len(df_init[data_mol_col_name])//part)):
        df = df_init[i*part:(i+1)*part]
        if calc_docking:
            d_s+=docking_list(smiles=df[data_mol_col_name],path_receptor_pdb=docking_path)
    if calc_docking:
        df_init['Docking Score'] = d_s 
    new_path = path_to_data.split(sep='.')[0]
    df_init.to_csv(f'{new_path}_props.csv')


def calc_ic_50(path_to_data:str,
                   data_mol_col_name:str,
                   case:str='Alzhmr',
                   part = 50):
                   
    
    df_init = pd.read_csv(path_to_data)


    df_init= df_init[:len(df_init[data_mol_col_name])//part*part]
    dis_case = cases[case]
    docking_path = dis_case['docking_path']
    IC50 = []
    
    for i in tqdm(range(len(df_init[data_mol_col_name])//part)):
       
        df = df_init[i*part:(i+1)*part]
        ic50_ki_values = dis_case['IC50'](df[data_mol_col_name])
        IC50+=list(ic50_ki_values)

    df_init['IC50'] = IC50
    df_init = df_init[df_init['IC50']==1]
    new_path = path_to_data.split(sep='.')[0]
    df_init.to_csv(f'{new_path}_props_ic50_{case}.csv')


if __name__ == "__main__":
    cases_ = {'alzheimer':'Alzhmr', 'dyslipidemia':'Dslpdm', 'lung cancer':'Cnsr', 'sclerosis':'Sklrz',
       'Drug_Resistance':'TBLET', 'Parkinson':'Prkns'}

    df = pd.read_csv('/projects/generative_models_data/utils/result_context.csv')
    df_t = df.dropna()
    #df_t = df_t.drop_duplicates(subset='extracted molecules')
    propert = [[],[],[],[],[],[],[],[]]
    for i in tqdm(range(len(df_t))):
        data = df_t.iloc[i]
        if len(list(ast.literal_eval(data['ChemDFM_out'])))==0:
            for p in propert:
                p.append(0)
        else:
            props = all_prop_calc_list(case=cases_[data['case']],data=list(ast.literal_eval(data['ChemDFM_out'])))
            for i in range(len(props)):
                propert[i].append(props[i])
    
    df_t['QED'],df_t['Synthetic'],df_t['PAINS'],df_t['SureChEMBL'],df_t['Glaxo'],df_t['Brenk'],df_t['IC50'],df_t['Docking_score'] = propert[0],propert[1],propert[2],propert[3],propert[4],propert[5],propert[6],propert[7] #'Docking score'
    df_t.to_csv('/projects/generative_models_data/utils/result_context2.csv')

    # df = pd.read_csv('/projects/generative_models_data/utils/result_context.csv')
    # df_t = df.dropna()
    # df_t['case'] = df_t['case'].apply(lambda x: x.split(', '))
    # #df_t = df_t.drop_duplicates(subset='extracted molecules')
    # propert = [[],[],[],[],[],[],[],[]]
    # for i in tqdm(range(len(df_t))):
    #     data = df_t.iloc[i]
    #     if len(ast.literal_eval(data['extracted molecules']))==0:
    #         for p in propert:
    #             p.append(0)
    #     else:
    #         #interm_cases = [[] for i in data['case']]
    #         curr_cases = [[],[],[],[],[],[],[],[]]
    #         for ca_i in range(len(data['case'])):
    #             props = all_prop_calc_list(case=cases_[data['case'][ca_i]],data=ast.literal_eval(data['extracted molecules']))
    #             for pr in range(len(props)):
    #                 curr_cases[pr].append(props[pr])
    #         for pr in range(len(curr_cases)):
    #             propert[pr].append(curr_cases[pr])
        
    
    # df_t['QED'],df_t['Synthetic'],df_t['PAINS'],df_t['SureChEMBL'],df_t['Glaxo'],df_t['Brenk'],df_t['IC50'],df_t['Docking_score'] = propert[0],propert[1],propert[2],propert[3],propert[4],propert[5],propert[6],propert[7] #'Docking score'
    # df_t.to_csv('utils/llasmol_answers_1_3tasks_all2.csv')
    # for case in cases:
        
    #     calc_ic_50(path_to_data='experiments/pure_rndm/counted_props_props_total.csv',
    #               data_mol_col_name='Smiles',
    #               case=case)
    
    # all_prop_calc(path_to_data='Vepreva_cases/tio/sklrz.csv',
    #               data_mol_col_name='Smiles',
    #               case='Alzhmr',
    #               calc_docking=True,
    #               gen_data=True,
    #               train_data_path = 'generative_models/GAN/ChEMBL_training/chembl_filtered_400w_150s.csv')
    # join_tables('Vepreva_cases/tio/sklrz.csv',
    #             'Vepreva_cases/tio/sklrz_props.csv',
    #             'Smiles')
    #calc_metrics('Vepreva_cases/tio/filters/Dslpdm/dyslipid_props_props_total.csv')