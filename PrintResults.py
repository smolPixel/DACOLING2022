
import subprocess
import argparse
import os
import shutil
import json

# import run_glue_xgboost
# import run_glue_lstm
# import run_glue_bert
from process_data import *
# from SentenceVAE.process_data import process_data_for_SVAE
# from SentenceVAE.processingText import createFoldersSVAE, add_data, checkTrained
import random
import numpy as np
import torch
from scipy.stats import ttest_ind, ttest_rel


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def getArray(argdict):
    results = []
    for seed in random_seeds:
        if argdict['vary_starting_set']:
            argdict['starting_set_seed'] = seed
        argdict['random_seed'] = seed
        argdict['retrain'] = False
        result, i=checkFolders(argdict, v=False)
        # results['acc_train'], results['acc_dev'], results['f1_train'], results['f1_dev']
        # print("BITCH")
        # print(result, i)
        # print(argdict)
        results.append({'acc_train': result[0], 'acc_dev':result[1], 'f1_train':result[2], 'f1_dev':result[3]})
    return results

def getAllResult(argdict):
    # retrainOG=argdict['retrain']
    results=getArray(argdict)
    print(results)
    return getResult(argdict, results)


def t_test(argdict):
    resultsArray=getArray(argdict)
    argdict['split']=0
    # if argdict['classifier']=='bert' and argdict['dataset']=="TREC6":
    #     argdict['nb_epochs_lstm']=8
    baselineArray=getArray(argdict)
    results=extractResults(resultsArray, argdict, metric='acc')
    baseline=extractResults(baselineArray, argdict, metric='acc')
    tt=ttest_rel(baseline, results)
    print("T test for accuracy")
    print(tt)
    results = extractResults(resultsArray, argdict, metric='f1')
    baseline = extractResults(baselineArray, argdict, metric='f1')
    tt = ttest_rel(baseline, results)
    print("T test for f1")
    print(tt)
#
# def findFile(argdict):
#     path=f"{argdict['pathDataAdd']}/Experiments/Record/{argdict['dataset']}/{argdict['algo']}/{argdict['classifier']}"
#     for folder in os.listdir(path):
#         param = json.load(open(f"{path}/{folder}/param.json", 'r'))
#         results = json.load(open(f"{path}/{folder}/out.json", 'r'))
#         same=compareFiles(argdict, param, "exp")
#         if same:
#             return param, results

def extractResults(results, argdict, metric='acc'):
    Acc = []
    # print(metric)
    # print(results)
    for rr in results:
        if argdict['classifier'] == 'lstm':
            Acc.append(max(rr[f'{metric}_dev']))
            # MaxAcc = max([MaxAcc, max(rr['acc_dev'])])
        elif argdict['classifier'] == 'dan':
            Acc.append(max(rr[f'{metric}_dev']))
            # MaxAcc = max([MaxAcc, max(rr['acc_dev'])])
        elif argdict['classifier'] == 'xgboost':
            # print(f'{metric}_dev')
            # print(rr)
            # print(rr[f'{metric}_dev'])
            Acc.append(rr[f'{metric}_dev'])
            # MaxAcc = max([MaxAcc, rr['acc_dev']])
        elif argdict['classifier'] in ['bert', 'bert2']:
            Acc.append(rr[f'{metric}_dev'])
            # MaxAcc = max([MaxAcc, rr['acc_dev']])
    return Acc

def getResult(argdict, results):
    tot=len(results)
    if tot!=len(random_seeds):
        raise ValueError(f"did not find {len(random_seeds)} files")
    totAcc=[]
    MaxAcc=0
    totf1=[]
    Maxf1=0
    for rr in results:
        if argdict['classifier'] in ['lstm', 'dan']:
            totAcc.append(max(rr['acc_dev']))
            MaxAcc=max([MaxAcc, max(rr['acc_dev'])])
            totf1.append(max(rr['f1_dev']))
            Maxf1 = max([Maxf1, max(rr['f1_dev'])])
        elif argdict['classifier']=='xgboost':
            totAcc.append(rr['acc_dev'])
            MaxAcc=max([MaxAcc, rr['acc_dev']])
            totf1.append(rr['f1_dev'])
            Maxf1 = max([Maxf1, rr['f1_dev']])
        elif argdict['classifier'] in ['bert', 'bert2']:
            totAcc.append(rr['acc_dev'])
            MaxAcc = max([MaxAcc, rr['acc_dev']])
            totf1.append(rr['f1_dev'])
            Maxf1 = max([Maxf1, rr['f1_dev']])
    return (np.mean(totAcc), np.std(totAcc)), MaxAcc, (np.mean(totf1), np.std(totf1)), Maxf1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE for data augmentation')
    #General arguments on training
    parser.add_argument('--dataset', type=str, default='SST-2', help="dataset you want to run the process on. Includes SST2, CoLA, MRPC")
    parser.add_argument('--classifier', type=str, default='lstm', help="classifier you want to use. Includes bert, lstm, jiantLstm or xgboost")
    parser.add_argument('--computer', type=str, default='home', help="Whether you run at home or at iro. Automatically changes the base path")
    parser.add_argument('--split', type=float, default=1.0, help='percent of the dataset added')
    parser.add_argument('--retrain', action='store_true', help='whether to retrain the VAE or not')
    parser.add_argument('--rerun', action='store_true', help='whether to rerun knowing that is has already been ran')
    parser.add_argument('--dataset_size', type=int, default=10, help='number of example in the original dataset. If 0, use the entire dataset')
    parser.add_argument('--algo', type=str, default='EDA', help='data augmentation algorithm to use, includes, EDA, GAN, and VAE')
    parser.add_argument('--random_seed', type=int, default=7, help='Random seed ')

    #VAE specific arguments
    parser.add_argument('--nb_epoch_VAE', type=int, default=10, help="Number of epoch for which to run the VAE")

    #Classifier specific arguments
    parser.add_argument('--nb_epochs_lstm', type=int, default=5, help="Number of epoch to train the lstm with")
    parser.add_argument('--dropout_classifier', type=int, default=0.3, help="dropout parameter of the classifier")
    parser.add_argument('--hidden_size_classifier', type=int, default=128, help="dropout parameter of the classifier")

    #Experiments
    parser.add_argument('--test_dss_vs_split', action='store_true', help='test the influence of the dataset size and ')
    parser.add_argument('--test_epochLstm_vs_split', action='store_true', help='test the influence of the number of epoch for the lstm classifier')
    parser.add_argument('--test_dropoutLstm_vs_split', action='store_true', help='test the influence of the dropout for the lstm classifier')
    parser.add_argument('--test_hsLstm_vs_split', action='store_true', help='test the influence of the dropout for the lstm classifier')
    parser.add_argument('--test_randomSeed', action='store_true', help='test the influence of the random seed for LSTM')
    parser.add_argument('--run_ds_split', action='store_true', help='Run all DS and splits with specified arguments')

    args = parser.parse_args()

    argsdict = args.__dict__
    if argsdict['computer']=='home':
        argsdict['pathDataOG'] = "/media/frederic/DAGlue/data"
        argsdict['pathDataAdd'] = "/media/frederic/DAGlue"
    elif argsdict['computer']=='labo':
        argsdict['pathDataOG'] = "/data/rali5/Tmp/piedboef/data/jiantData/OG/OG"
        argsdict['pathDataAdd'] = "/u/piedboef/Documents/DAGlue"
    # Create directories for the runs

    # json.dump(argsdict, open(f"/media/frederic/DAGlue/SentenceVAE/GeneratedData/{argsdict['dataset']}/{argsdict['dataset_size']}/parameters.json", "w"))


    print("=================================================================================================")
    print(argsdict)
    print("=================================================================================================")

    if argsdict['test_dss_vs_split']:
        dico={'help':'key is dataset size, then key inside is split and value is a tuple of train val accuracy'}
        for ds in [10,50,100,500,5000,10000,20000,30000,40000,50000]:
            dicoTemp={}
            for split in [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                print(f"Dataset size {ds}, split {split}")
                argsdict['dataset_size']=ds
                argsdict['split']=split
                argsdict['retrain']=False
                # createFoldersEDA(argsdict)
                acc_train, acc_val=main(argsdict)
                dicoTemp[split]=(acc_train, acc_val)
            dico[ds]=dicoTemp
            with open(f"/media/frederic/DAGlue/Experiments/DSSVSSplit/EDA.json", "w") as f:
                json.dump(dico, f)
    elif argsdict['test_epochLstm_vs_split']:
        dico = {'help': 'key is number of epoch, then key inside is split and value is a tuple of train val accuracy (lists)'}
        argsdict['dataset_size'] = 5000
        for epochs in [5, 10, 30]:
            dicoTemp={}
            for split in [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                print(f"Number of epochs {epochs}, split {split}")
                argsdict['nb_epochs_lstm']=epochs
                argsdict['split']=split
                argsdict['retrain']=False
                # createFoldersEDA(argsdict)
                acc_train, acc_val=main(argsdict)
                dicoTemp[split]=(acc_train, acc_val)
            dico[epochs]=dicoTemp
            with open(f"/media/frederic/DAGlue/Experiments/epochLstmVsSplit/EDA.json", "w") as f:
                json.dump(dico, f)
    elif argsdict['test_dropoutLstm_vs_split']:
        dico = {'help': 'key is dropout, then key inside is split and value is a tuple of train val accuracy (lists)'}
        argsdict['dataset_size'] = 5000
        argsdict['nb_epoch_lstm']=5
        for dropout in [0.3, 0.6, 0.9]:
            dicoTemp={}
            for split in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                print(f"Dropout {dropout}, split {split}")
                argsdict['dropout_classifier']=dropout
                argsdict['split']=split
                argsdict['retrain']=False
                # createFoldersEDA(argsdict)
                acc_train, acc_val=main(argsdict)
                dicoTemp[split]=(acc_train, acc_val)
            dico[dropout]=dicoTemp
            with open(f"/media/frederic/DAGlue/Experiments/dropoutLstmVsSplit/{argsdict['algo']}.json", "w") as f:
                json.dump(dico, f)
    elif argsdict['test_hsLstm_vs_split']:
        dico = {'help': 'key is dropout, then key inside is split and value is a tuple of train val accuracy (lists)'}
        argsdict['dataset_size'] = 5000
        argsdict['nb_epoch_lstm']=5
        argsdict['dropout_classifier']=0.3
        for hidden_size in [100, 500, 1000, 1500]:
            dicoTemp={}
            for split in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                print(f"Hidden Size {hidden_size}, split {split}")
                argsdict['hidden_size_classifier']=hidden_size
                argsdict['split']=split
                argsdict['retrain']=False
                # createFoldersEDA(argsdict)
                acc_train, acc_val=main(argsdict)
                dicoTemp[split]=(acc_train, acc_val)
            dico[hidden_size]=dicoTemp
            with open(f"/media/frederic/DAGlue/Experiments/HiddenSizeLstmVsSplit/EDA.json", "w") as f:
                json.dump(dico, f)
    elif argsdict['test_randomSeed']:
        dico = {'help': 'key is random seed, then key inside is split and value is a tuple of train val accuracy (lists)'}
        argsdict['dataset_size'] = 5000
        argsdict['nb_epoch_lstm']=5
        argsdict['dropout_classifier']=0.3
        for random_seed in [7,3,9,13,100,500]:
            dicoTemp={}
            for split in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                print(f"Random Seed {random_seed}, split {split}")
                argsdict['random_seed']=random_seed
                argsdict['split']=split
                argsdict['retrain']=False
                # createFoldersEDA(argsdict)
                acc_train, acc_val=main(argsdict)
                dicoTemp[split]=(acc_train, acc_val)
            dico[random_seed]=dicoTemp
            with open(f"/media/frederic/DAGlue/Experiments/RandomSeed{argsdict['classifier']}VsSplit/EDA.json", "w") as f:
                json.dump(dico, f)
    elif argsdict['run_ds_split']:
        string=""
        for dsSize in [50,500,5000,30000]:
            print("=========")
            for split in [0, 0.2, 0.5, 1.0]:
                # print(f"Printing results with dataset of {dsSize} and split of {split}")
                argsdict['retrain']=False
                argsdict['dataset_size']=dsSize
                argsdict['split']=split
                acc_val=main(argsdict)
                string+= "{:.1f} & ".format(acc_val*100)
        print(string)
    else:
        acc_train, acc_dev=main(argsdict)
#Random seeds: 3,7,9,13,100, -500, 512, 12, 312, 888