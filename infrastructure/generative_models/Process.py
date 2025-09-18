import pandas as pd
import torch
import torchtext
from torchtext import data
from Tokenize import moltokenize
from Batch import MyIterator, batch_size_fn
import os
import dill as pickle
import numpy as np
import sys
import generative_models as KOSTIL_FOR_PICKL
sys.modules['generative_models.Sber_Alzheimer'] = KOSTIL_FOR_PICKL

sys.path.append(os.path.dirname(__file__))

def read_data(opt,mol_column):
    if opt.src_data is not None:
       opt.src_data = pd.read_csv(opt.src_data)[mol_column]
    if opt.trg_data is not None:
       opt.trg_data = pd.read_csv(opt.trg_data)[mol_column]

def create_fields(opt):
    lang_formats = ['SMILES', 'SELFIES']
    if opt.lang_format not in lang_formats:
        print('invalid src language: ' + opt.lang_forma + 'supported languages : ' + lang_formats)

    print("loading molecule tokenizers...")

    t_src = moltokenize()
    t_trg = moltokenize()
    if opt.load_weights_fields is None:
        SRC = data.Field(tokenize=t_src.tokenizer)
        TRG = data.Field(tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')

    if opt.load_weights_fields is not None:
        try:
            print("loading presaved fields...")
            SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))

        except:
            print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
            quit()

    return (SRC, TRG)


def create_dataset(opt, SRC, TRG, PROP, tr_te,test_mode:bool = False, shuffle=True,if_cond=True):
    """

    :param opt:
    :param SRC:
    :param TRG:
    :param PROP:
    :param tr_te:
    :param test_mode: Slice dataset to df[:3000] and df[:300] parts for accelerate data preparing in debug/cod-test mode
    :return:
    """
    # masking data longer than max_strlen
    if tr_te == "tr":
        print("\n* creating [train] dataset and iterator... ")
        raw_data = {'src': [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
    if tr_te == "te":
        print("\n* creating [test] dataset and iterator... ")
        raw_data = {'src': [line for line in opt.src_data_te], 'trg': [line for line in opt.trg_data_te]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    if if_cond:
        df = pd.concat([df, PROP], axis=1)
    if test_mode:
        if tr_te == "tr":  #for code test
            df = df[:3000]
        if tr_te == "te":
            df = df[:300]

    # if opt.lang_format == 'SMILES':
    #     mask = (df['src'].str.len() + opt.cond_dim < opt.max_strlen) & (df['trg'].str.len() + opt.cond_dim < opt.max_strlen)
    # if opt.lang_format == 'SELFIES':
    #     mask = (df['src'].str.count('][') + opt.cond_dim < opt.max_strlen) & (df['trg'].str.count('][') + opt.cond_dim < opt.max_strlen)

    #df = df.loc[mask]
    if tr_te == "tr":
        print("     - # of training samples:", len(df.index))
        df.to_csv(f"{opt.save_folder_name}/DB_transformer_temp.csv", index=False)
    if tr_te == "te":
        print("     - # of test samples:", len(df.index))
        df.to_csv(f"{opt.save_folder_name}/DB_transformer_temp_te.csv", index=False)

    logP = data.Field(use_vocab=False, sequential=False, dtype=torch.float)
    tPSA = data.Field(use_vocab=False, sequential=False, dtype=torch.float)
    QED = data.Field(use_vocab=False, sequential=False, dtype=torch.float)
    if if_cond:
        data_fields = [('src', SRC), ('trg', TRG), ('logP', logP), ('tPSA', tPSA), ('QED', QED)]
    else:
        data_fields = [('src', SRC), ('trg', TRG)]
    if tr_te == "tr":
        toklenList = []
        train = data.TabularDataset(f'{opt.save_folder_name}/DB_transformer_temp.csv', format='csv', fields=data_fields, skip_header=True)
        for i in range(len(train)):
            toklenList.append(len(vars(train[i])['src']))
        df_toklenList = pd.DataFrame(toklenList, columns=["toklen"])
        df_toklenList.to_csv(f"{opt.save_folder_name}/weights/toklen_list.csv", index=False)
        if opt.verbose == True:
            print("     - tokenized training sample 0:", vars(train[0]))
    if tr_te == "te":
        train = data.TabularDataset(f"{opt.save_folder_name}/DB_transformer_temp_te.csv", format='csv', fields=data_fields, skip_header=True)
        if opt.verbose == True:
            print("     - tokenized testing sample 0:", vars(train[0]))

    train_iter = MyIterator(train, batch_size=opt.batchsize, device=opt.device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True, shuffle=shuffle)
    try:
        os.remove(f'{opt.save_folder_name}/DB_transformer_temp.csv')
    except:
        pass
    try:
        os.remove(f'{opt.save_folder_name}/DB_transformer_temp_te.csv')
    except:
        pass

    if tr_te == "tr":
        if opt.load_weights is None:
            print("     - building vocab from train data...")
            SRC.build_vocab(train)
            if opt.verbose == True:
                print('     - vocab size of SRC: {}\n        -> {}'.format(len(SRC.vocab), SRC.vocab.stoi))
            TRG.build_vocab(train)
            if opt.verbose == True:
                print('     - vocab size of TRG: {}\n        -> {}'.format(len(TRG.vocab), TRG.vocab.stoi))
            if opt.checkpoint > 0:
                try:
                    if not os.path.isdir(f"{opt.save_folder_name}/weights"):
                        os.mkdir("weights")
                except:
                    print("weights folder already exists, run program with -load_weights weights to load them")
                    quit()
                pickle.dump(SRC, open(f'{opt.save_folder_name}/weights/SRC.pkl', 'wb'))
                pickle.dump(TRG, open(f'{opt.save_folder_name}/weights/TRG.pkl', 'wb'))

        opt.src_pad = SRC.vocab.stoi['<pad>']
        opt.trg_pad = TRG.vocab.stoi['<pad>']

        opt.train_len = get_len(train_iter)

    if tr_te == "te":
        opt.test_len = get_len(train_iter)

    return train_iter

def create_dataset_auto(opt, SRC, TRG, PROP, tr_te,test_mode:bool = False, shuffle=True,if_cond=True,is_train=True):
    """

    :param opt:
    :param SRC:
    :param TRG:
    :param PROP:
    :param tr_te:
    :param test_mode: Slice dataset to df[:3000] and df[:300] parts for accelerate data preparing in debug/cod-test mode
    :return:
    """
    # masking data longer than max_strlen
    if tr_te == "tr":
        print("\n* creating [train] dataset and iterator... ")
        raw_data = {'src': [line for line in opt.src_data[opt.feature_col[0]]], 'trg': [line for line in opt.trg_data[opt.feature_col[0]]]}
    if tr_te == "te":
        print("\n* creating [test] dataset and iterator... ")
        raw_data = {'src': [line for line in opt.src_data_te], 'trg': [line for line in opt.trg_data_te]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    if if_cond:
        df = pd.concat([df, PROP], axis=1)
    if test_mode:
        if tr_te == "tr":  #for code test
            df = df[:30000]
        if tr_te == "te":
            df = df[:300]

    # if opt.lang_format == 'SMILES':
    #     mask = (df['src'].str.len() + opt.cond_dim < opt.max_strlen) & (df['trg'].str.len() + opt.cond_dim < opt.max_strlen)
    # if opt.lang_format == 'SELFIES':
    #     mask = (df['src'].str.count('][') + opt.cond_dim < opt.max_strlen) & (df['trg'].str.count('][') + opt.cond_dim < opt.max_strlen)

    #df = df.loc[mask]
    if tr_te == "tr":
        print("     - # of training samples:", len(df.index))
        df.to_csv(f"{opt.save_folder_name}/DB_transformer_temp.csv", index=False)
    if tr_te == "te":
        print("     - # of test samples:", len(df.index))
        df.to_csv(f"{opt.save_folder_name}/DB_transformer_temp_te.csv", index=False)

    #ic50 = data.Field(use_vocab=False, sequential=False, dtype=torch.float)
    # tPSA = data.Field(use_vocab=False, sequential=False, dtype=torch.float)
    # QED = data.Field(use_vocab=False, sequential=False, dtype=torch.float)
    if if_cond:
        data_fields = [('src', SRC), ('trg', TRG)] + [(i, data.Field(use_vocab=False, sequential=False, dtype=torch.float)) for i in opt.conditions]
    else:
        data_fields = [('src', SRC), ('trg', TRG)]
    if tr_te == "tr":
        toklenList = []
        train = data.TabularDataset(f'{opt.save_folder_name}/DB_transformer_temp.csv', format='csv', fields=data_fields, skip_header=True)
        for i in range(len(train)):
            toklenList.append(len(vars(train[i])['src']))
        df_toklenList = pd.DataFrame(toklenList, columns=["toklen"])
        if not os.path.isdir(f"{opt.save_folder_name}/weights"):
            os.mkdir(f"{opt.save_folder_name}/weights")
        df_toklenList.to_csv(f"{opt.save_folder_name}/weights/toklen_list.csv", index=False)
        if opt.verbose == True:
            print("     - tokenized training sample 0:", vars(train[0]))
    if tr_te == "te":
        train = data.TabularDataset(f"{opt.save_folder_name}/DB_transformer_temp_te.csv", format='csv', fields=data_fields, skip_header=True)
        if opt.verbose == True:
            print("     - tokenized testing sample 0:", vars(train[0]))

    train_iter = MyIterator(train, batch_size=opt.batchsize, device=opt.device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=is_train, shuffle=shuffle)
    try:
        os.remove(f'{opt.save_folder_name}/DB_transformer_temp.csv')
    except:
        pass

    pickle.dump(SRC, open(f'{opt.save_folder_name}/weights/SRC.pkl', 'wb'))
    pickle.dump(TRG, open(f'{opt.save_folder_name}/weights/TRG.pkl', 'wb'))
    if tr_te == "tr":
        if opt.load_weights_fields is None:
            print("     - building vocab from train data...")
            SRC.build_vocab(train)
            if opt.verbose == True:
                print('     - vocab size of SRC: {}\n        -> {}'.format(len(SRC.vocab), SRC.vocab.stoi))
            TRG.build_vocab(train)
            if opt.verbose == True:
                print('     - vocab size of TRG: {}\n        -> {}'.format(len(TRG.vocab), TRG.vocab.stoi))
            if opt.checkpoint > 0:
                try:
                    if not os.path.isdir(f"{opt.save_folder_name}/weights"):
                        os.mkdir("weights")
                except:
                    print("weights folder already exists, run program with -load_weights weights to load them")
                    quit()
                pickle.dump(SRC, open(f'{opt.save_folder_name}/weights/SRC.pkl', 'wb'))
                pickle.dump(TRG, open(f'{opt.save_folder_name}/weights/TRG.pkl', 'wb'))

        opt.src_pad = SRC.vocab.stoi['<pad>']
        opt.trg_pad = TRG.vocab.stoi['<pad>']

        opt.train_len = get_len(train_iter)

    if tr_te == "te":
        opt.test_len = get_len(train_iter)

    return train_iter


def create_dataset_vectors(opt, SRC, TRG, PROP, tr_te,test_mode:bool = False, shuffle=True,if_cond=True):
    """

    :param opt:
    :param SRC:
    :param TRG:
    :param PROP:
    :param tr_te:
    :param test_mode: Slice dataset to df[:3000] and df[:300] parts for accelerate data preparing in debug/cod-test mode
    :return:
    """
    # masking data longer than max_strlen
    if tr_te == "tr":
        print("\n* creating [train] dataset and iterator... ")
        raw_data = {'src': [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
    if tr_te == "te":
        print("\n* creating [test] dataset and iterator... ")
        raw_data = {'src': [line for line in opt.src_data_te], 'trg': [line for line in opt.trg_data_te]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    if if_cond:
        df = pd.concat([df, PROP], axis=1)
    if test_mode:
        if tr_te == "tr":  #for code test
            df = df[:3000]
        if tr_te == "te":
            df = df[:300]

    # if opt.lang_format == 'SMILES':
    #     mask = (df['src'].str.len() + opt.cond_dim < opt.max_strlen) & (df['trg'].str.len() + opt.cond_dim < opt.max_strlen)
    # # if opt.lang_format == 'SELFIES':
    # #     mask = (df['src'].str.count('][') + opt.cond_dim < opt.max_strlen) & (df['trg'].str.count('][') + opt.cond_dim < opt.max_strlen)

    # df = df.loc[mask]
    if tr_te == "tr":
        print("     - # of training samples:", len(df.index))
        df.to_csv(f"{opt.save_folder_name}/DB_transformer_temp.csv", index=False)
    if tr_te == "te":
        print("     - # of test samples:", len(df.index))
        df.to_csv(f"{opt.save_folder_name}/DB_transformer_temp_te.csv", index=False)

    logP = data.Field(use_vocab=False, sequential=False, dtype=torch.float)
    tPSA = data.Field(use_vocab=False, sequential=False, dtype=torch.float)
    QED = data.Field(use_vocab=False, sequential=False, dtype=torch.float)
    if if_cond:
        data_fields = [('src', SRC), ('trg', TRG), ('logP', logP), ('tPSA', tPSA), ('QED', QED)]
    else:
        data_fields = [('src', SRC), ('trg', TRG)]
    if tr_te == "tr":
        toklenList = []
        train = data.TabularDataset(f'{opt.save_folder_name}/DB_transformer_temp.csv', format='csv', fields=data_fields, skip_header=True)
        for i in range(len(train)):
            toklenList.append(len(vars(train[i])['src']))
        df_toklenList = pd.DataFrame(toklenList, columns=["toklen"])
        df_toklenList.to_csv(f"{opt.save_folder_name}/weights/toklen_list.csv", index=False)
        if opt.verbose == True:
            print("     - tokenized training sample 0:", vars(train[0]))
    if tr_te == "te":
        train = data.TabularDataset(f"{opt.save_folder_name}/DB_transformer_temp_te.csv", format='csv', fields=data_fields, skip_header=True)
        if opt.verbose == True:
            print("     - tokenized testing sample 0:", vars(train[0]))
    
    train_iter = data.Iterator(train, batch_size=opt.batchsize, device=opt.device,
                            repeat=False, sort_key=None,
                            batch_size_fn=None, train=False, shuffle=shuffle,sort=False)
    try:
        os.remove(f'{opt.save_folder_name}/DB_transformer_temp.csv')
    except:
        pass
    try:
        os.remove(f'{opt.save_folder_name}/DB_transformer_temp_te.csv')
    except:
        pass

    if tr_te == "tr":
        if opt.load_weights is None:
            print("     - building vocab from train data...")
            SRC.build_vocab(train)
            if opt.verbose == True:
                print('     - vocab size of SRC: {}\n        -> {}'.format(len(SRC.vocab), SRC.vocab.stoi))
            TRG.build_vocab(train)
            if opt.verbose == True:
                print('     - vocab size of TRG: {}\n        -> {}'.format(len(TRG.vocab), TRG.vocab.stoi))
            if opt.checkpoint > 0:
                try:
                    if not os.path.isdir(f"{opt.save_folder_name}/weights"):
                        os.mkdir("weights")
                except:
                    print("weights folder already exists, run program with -load_weights weights to load them")
                    quit()
                pickle.dump(SRC, open(f'{opt.save_folder_name}/weights/SRC.pkl', 'wb'))
                pickle.dump(TRG, open(f'{opt.save_folder_name}/weights/TRG.pkl', 'wb'))

        opt.src_pad = SRC.vocab.stoi['<pad>']
        opt.trg_pad = TRG.vocab.stoi['<pad>']

        opt.train_len = get_len(train_iter)

    if tr_te == "te":
        opt.test_len = get_len(train_iter)

    return train_iter


def get_len(train):
    for i, b in enumerate(train):
        pass
    return i

