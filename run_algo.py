
import subprocess
import argparse
import os
import shutil
import json

import run_glue_xgboost
import run_glue_DAN
# import run_glue_lstm

from process_data import *
from textaugment.AugmentWord2Vec import augment_w2v, augment_Trans
# from SentenceVAE.process_data import process_data_for_SVAE
# from SentenceVAE.processingText import createFoldersSVAE, add_data, checkTrained
from PrintResults import *
from Visualization import *
from create_graph import plot_graph
from PrintSamples import print_samples
import time
import math
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_external_process(process):
    output, error = process.communicate()
    if process.returncode != 0:
        raise SystemError
    return output, error


def output_results(argdict):
    resultsAll={'augmented':{}, 'baseline':{}}
    rerun=argdict['rerun']
    for seed in random_seeds:
        print(seed)
        argdict['random_seed'] = seed
        argdict = checkTrained(argdict)
        # folderNumGen = argdict['numFolderGenerated']
        # data = argdict['dataset']
        # split = argdict['split']  # Percent of data points added
        # dataset_size = argdict['dataset_size']
        tup, i = checkFolders(argdict)
        if type(tup) != int:
            resultsAll['augmented'][seed]=tup
        else:
            raise ValueError("Experiment not run")
    argdict['split']=0

    if not argdict['rerun_split_zero']:
        argdict['rerun']=False
    else:
        argdict['rerun']=True
    for seed in random_seeds:
        print(seed)
        argdict['random_seed'] = seed
        argdict = checkTrained(argdict)
        # folderNumGen = argdict['numFolderGenerated']
        # data = argdict['dataset']
        # split = argdict['split']  # Percent of data points added
        # dataset_size = argdict['dataset_size']
        tup, i = checkFolders(argdict)
        if type(tup) != int:
            # print(tup)
            resultsAll['baseline'][seed]=tup
        else:
            raise ValueError("Experiment not run")
    argdict['rerun']=rerun
    json.dump(resultsAll, open("temp.json", "w"))

_start_time = time.time()

# def tic():
#     global _start_time
#     _start_time = time.time()

def tac(t_sec):
    # t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60)
    print('Average Time of the augmentation algorithm: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))

def main(argdict):
    # retrainOG=argdict['retrain']
    averageTime=[]
    # if argdict['dataset_size']==0:
    #     random_seeds=[0,1,2,3,4,5]
    # print(random_seeds)
    # fsd
    # print(argdict['rerun'])
    # fds
    for seed in random_seeds:
        print(seed)
        if argdict['vary_starting_set']:
            argdict['starting_set_seed']=seed
        if argdict['second_dataset']:
            argdict['starting_set_seed']=1
        argdict['random_seed'] = seed
        argdict=createFolders(argdict)
        time0=time.time()
        tt=run(argdict)
        timeProcess = time.time() - time0
        averageTime.append(timeProcess)
    #     if seed==2:
    #         break
    # return 0
    Oldsplit=argdict['split']
    oldLstm=argdict['nb_epochs_lstm']
    oldRun=argdict['rerun']
    argdict['split']=0
    argdict['retrain']=False
    # if argdict['classifier']=='bert' and argdict['dataset']=="TREC6":
    #     argdict['nb_epochs_lstm']=8
    if not argdict['rerun_split_zero']:
        argdict['rerun'] = False
    else:
        argdict['rerun'] =True
    for seed in random_seeds:
        print(seed)
        if argdict['vary_starting_set']:
            argdict['starting_set_seed']=seed
        if argdict['second_dataset']:
            argdict['starting_set_seed']=1
        argdict['random_seed'] = seed
        argdict=createFolders(argdict)
        tt2=run(argdict)


    argdict['split']=Oldsplit
    argdict['nb_epochs_lstm'] = oldLstm
    argdict['rerun']=oldRun
    t_test(argdict)
    argdict['split'] = Oldsplit
    argdict['nb_epochs_lstm'] = oldLstm
    argdict['rerun']=oldRun
    # print("BNIPASNDKOIJOIJIJOJOIAIJODSA")
    # print(tt)
    averageTime=sum(averageTime)/len(averageTime)
    tac(averageTime)
    return tt



def run(argdict):
    #Check whether we should retrain
    set_seed(argdict['random_seed'])
    #CheckTrained add the argument "numFolderGenerated"
    argdict = checkTrained(argdict)
    folderNumGen=argdict['numFolderGenerated']
    # num_ex=num_exemples_generated(argdict)
    # if argdict['dataset_size']!=0:
    #     total_number_wanted=argdict['dataset_size']*argdict['split']
    # if total_number_wanted>num_ex:
    #     argdict['retrain']=True

    data = argdict['dataset']
    split = argdict['split']  # Percent of data points added
    dataset_size = argdict['dataset_size']
    tup, i = checkFolders(argdict)
    if type(tup) != int and not argdict['rerun']:
        return tup
    else:
        tup=i

    # print("BITCH")
    if argdict['algo']=='VAE':
        # Delete old generated data
        # print("HOLA")
        # print(argdict['retrain'])
        if argdict['retrain']:
            try:
                for files in os.listdir(f'{argdict["pathDataAdd"]}/GeneratedData/VAE/{data}/{dataset_size}/{folderNumGen}'):
                    os.remove(f'{argdict["pathDataAdd"]}/GeneratedData/VAE/{data}/{dataset_size}/{files}/{folderNumGen}')
            except:
                pass
            try:
                for folders in os.listdir(f'{argdict["pathDataAdd"]}/SentenceVAE/dumps'):
                    shutil.rmtree(f'{argdict["pathDataAdd"]}/SentenceVAE/dumps/{folders}')
            except:
                pass

        for cat in categories:
            if argdict['retrain']:
                for files in os.listdir(f'{argdict["pathDataAdd"]}/SentenceVAE/data'):
                    os.remove(f'{argdict["pathDataAdd"]}/SentenceVAE/data/{files}')
                for direc in os.listdir(f'SentenceVAE/bin'):
                    shutil.rmtree(f"SentenceVAE/bin/{direc}", ignore_errors=True)
            argdict['cat'] = cat
            num_points = process_data(argdict)
            # print(num_points)
            num_points = int(num_points)
            # Generate the maximum number of points, that is 5 times the dataset per class
            num_generated = round(num_points * 5)
            # num_generated=round(num_points * split / len(categories))
            argdict['num_to_add'] = round(num_points * split / len(categories))

            if argdict['retrain']:
                # #Training VA
                from SentenceVAE.VAE import VAE
                Algo = VAE(argdict)
                Algo.train()
                Algo.augment()
    elif argdict['algo']=='VAE_LinkedEnc':
        # Delete old generated data
        # print("HOLA")
        # print(argdict['retrain'])
        if argdict['retrain']:
            try:
                for files in os.listdir(f'{argdict["pathDataAdd"]}/GeneratedData/VAELinkedEnc/{data}/{dataset_size}/{folderNumGen}'):
                    os.remove(f'{argdict["pathDataAdd"]}/GeneratedData/VAELinkedEnc/{data}/{dataset_size}/{files}/{folderNumGen}')
            except:
                pass
            try:
                for folders in os.listdir(f'{argdict["pathDataAdd"]}/SentenceVAELinkedEnc/dumps'):
                    shutil.rmtree(f'{argdict["pathDataAdd"]}/SentenceVAELinkedEnc/dumps/{folders}')
            except:
                pass
        if argdict['retrain']:
            for files in os.listdir(f'{argdict["pathDataAdd"]}/SentenceVAELinkedEnc/data'):
                os.remove(f'{argdict["pathDataAdd"]}/SentenceVAELinkedEnc/data/{files}')
            for direc in os.listdir(f'SentenceVAELinkedEnc/bin'):
                shutil.rmtree(f"SentenceVAELinkedEnc/bin/{direc}", ignore_errors=True)
        num_points = process_data(argdict)
        # print(num_points)
        num_points = int(num_points)
        # Generate the maximum number of points, that is 5 times the dataset per class
        num_generated = round(num_points * 5)
        # num_generated=round(num_points * split / len(categories))
        argdict['num_to_add'] = round(num_points * split / len(categories))

        if argdict['retrain']:
            # #Training VA
            from SentenceVAELinkedEnc.VAE import VAE
            Algo = VAE(argdict)
            Algo.train()
            Algo.augment()

    elif argdict['algo']=='VAE_EncDec':
        # Delete old generated data
        if argdict['retrain']:
            try:
                for files in os.listdir(f'{argdict["pathDataAdd"]}/GeneratedData/VAE_EncDec/{data}/{dataset_size}/{folderNumGen}'):
                    os.remove(f'{argdict["pathDataAdd"]}/GeneratedData/VAE_EncDec/{data}/{dataset_size}/{folderNumGen}/{files}')
            except:
                pass
            try:
                for folders in os.listdir(f'{argdict["pathDataAdd"]}/VAE_EncDec/dumps'):
                    shutil.rmtree(f'{argdict["pathDataAdd"]}/VAE_EncDec/dumps/{folders}')
            except:
                pass


        if argdict['retrain']:
            if not argdict['unidir_algo']:
                bi = "-bi"
            else:
                bi = ""
            for files in os.listdir(f'{argdict["pathDataAdd"]}/VAE_EncDec/data'):
                os.remove(f'{argdict["pathDataAdd"]}/VAE_EncDec/data/{files}')
            for direc in os.listdir(f'VAE_EncDec/bin'):
                shutil.rmtree(f"VAE_EncDec/bin/{direc}", ignore_errors=True)
        # argdict['cat'] = cat
        num_points = process_data(argdict)
        # print(num_points)
        num_points = int(num_points)
        # Generate the maximum number of points, that is 5 times the dataset per class
        num_generated = round(num_points * 5)
        # num_generated=round(num_points * split / len(categories))
        argdict['num_to_add'] = round(num_points * split / len(categories))

        if argdict['retrain']:
            # #Training VAE
            from VAE_EncDec.VAE import VAE
            Algo=VAE(argdict)
            Algo.train()
            Algo.augment()

    elif argdict['algo']=='CVAE':
        # Delete old generated data
        # print("HOLA")
        # print(argdict['retrain'])
        if argdict['retrain']:
            try:
                for files in os.listdir(f'{argdict["pathDataAdd"]}/GeneratedData/CVAE/{data}/{dataset_size}/{folderNumGen}'):
                    os.remove(f'{argdict["pathDataAdd"]}/GeneratedData/CVAE/{data}/{dataset_size}/{files}/{folderNumGen}')
            except:
                pass
            try:
                for folders in os.listdir(f'{argdict["pathDataAdd"]}/CVAE/dumps'):
                    shutil.rmtree(f'{argdict["pathDataAdd"]}/CVAE/dumps/{folders}')
            except:
                pass

        if argdict['retrain']:
            for files in os.listdir(f'{argdict["pathDataAdd"]}/CVAE/data'):
                os.remove(f'{argdict["pathDataAdd"]}/CVAE/data/{files}')
            for direc in os.listdir(f'CVAE/bin'):
                shutil.rmtree(f"CVAE/bin/{direc}", ignore_errors=True)
        # argdict['cat'] = cat
        num_points = process_data(argdict)
        # print(num_points)
        num_points = int(num_points)
        # Generate the maximum number of points, that is 5 times the dataset per class
        num_generated = round(num_points * 5)
        # num_generated=round(num_points * split / len(categories))
        argdict['num_to_add'] = round(num_points * split / len(categories))

        if argdict['retrain']:
            # #Training VAE
            if not argdict['unidir_algo']:
                bi="-bi"
            else:
                bi=""
            # print(argdict['unidir_algo'])
            # print(bi)
            # bashCommand = f'python3 {argdict["pathDataAdd"]}/SentenceVAE/train.py --data_dir {argdict["pathDataAdd"]}/SentenceVAE/data -ep {argdict["nb_epoch_algo"]}' \
            #               f' -ls {argdict["latent_size"]} -hs {argdict["hidden_size_algo"]} -nl {argdict["nb_layers_algo"]} -ed {argdict["dropout_algo"]}' \
            #               f' -rnn {argdict["rnn_type_algo"]} -wd {argdict["word_dropout"]} -k {argdict["k"]} -x0 {argdict["x0"]} {bi} -bs {argdict["bs_algo"]}'
            # process = subprocess.Popen(bashCommand.split())
            # output, error = run_external_process(process)


            # #Training VAE
            from CVAE.CVAE import CVAE
            Algo = CVAE(argdict)
            Algo.train()
            Algo.augment()


    elif argdict['algo']=='CVAE_Classic':
        # Delete old generated data
        if argdict['retrain']:
            try:
                for files in os.listdir(f'{argdict["pathDataAdd"]}/GeneratedData/CVAE_Classic/{data}/{dataset_size}/{folderNumGen}'):
                    os.remove(f'{argdict["pathDataAdd"]}/GeneratedData/CVAE_Classic/{data}/{dataset_size}/{folderNumGen}/{files}')
            except:
                pass
            try:
                for folders in os.listdir(f'{argdict["pathDataAdd"]}/CVAE_Classic/dumps'):
                    shutil.rmtree(f'{argdict["pathDataAdd"]}/CVAE_Classic/dumps/{folders}')
            except:
                pass


        if argdict['retrain']:
            if not argdict['unidir_algo']:
                bi = "-bi"
            else:
                bi = ""
            for files in os.listdir(f'{argdict["pathDataAdd"]}/CVAE_Classic/data'):
                os.remove(f'{argdict["pathDataAdd"]}/CVAE_Classic/data/{files}')
            for direc in os.listdir(f'CVAE_Classic/bin'):
                shutil.rmtree(f"CVAE_Classic/bin/{direc}", ignore_errors=True)
        # argdict['cat'] = cat
        num_points = process_data(argdict)
        # print(num_points)
        num_points = int(num_points)
        # Generate the maximum number of points, that is 5 times the dataset per class
        num_generated = round(num_points * 5)
        # num_generated=round(num_points * split / len(categories))
        argdict['num_to_add'] = round(num_points * split / len(categories))

        if argdict['retrain']:
            # #Training VAE
            bashCommand = f'python3 {argdict["pathDataAdd"]}/CVAE_Classic/train.py --data_dir {argdict["pathDataAdd"]}/CVAE_Classic/data -ep {argdict["nb_epoch_algo"]}' \
                              f' -ls {argdict["latent_size"]} -hs {argdict["hidden_size_algo"]} -nl {argdict["nb_layers_algo"]} -ed {argdict["dropout_algo"]}' \
                              f' -rnn {argdict["rnn_type_algo"]} -wd {argdict["word_dropout"]} -k {argdict["k"]} -x0 {argdict["x0"]} -nc {len(argdict["categories"])} {bi} ' \
                              f'-bs {argdict["bs_algo"]}'
            process = subprocess.Popen(bashCommand.split())
            output, error = run_external_process(process)
            # Inference
            for dir in os.listdir(f'CVAE_Classic/bin'):
                # TODO CHANGE THIS IF MORE THAN 9
                #TODO SEE IF YOU CAN CHANGE IT FOR IRO
                dir = f'CVAE_Classic/bin/{dir}/E{argdict["nb_epoch_algo"]-1}.pytorch'
            # Prevent memory overflow
            max_gen = 500

            for cat in argdict['categories']:
                ind=argdict['categories'].index(cat)
                while num_generated > max_gen:
                    num_gen_cur = max_gen
                    num_generated = num_generated - max_gen

                    bashCommand = f'python3 {argdict["pathDataAdd"]}/CVAE_Classic/inference.py --data_dir {argdict["pathDataAdd"]}/CVAE_Classic/data -c {dir}' \
                              f' -n {max_gen} -ls {argdict["latent_size"]} -hs {argdict["hidden_size_algo"]} -ed {argdict["dropout_algo"]} -nl {argdict["nb_layers_algo"]}' \
                              f' -rnn {argdict["rnn_type_algo"]} -wd {argdict["word_dropout"]} {bi} -cat {ind} -nc {len(argdict["categories"])}'

                    with open(f"{argdict['pathDataAdd']}/GeneratedData/CVAE_Classic/{data}/{dataset_size}/{folderNumGen}/{cat}.txt", "a") as outfile:
                        process = subprocess.Popen(bashCommand.split(), stdout=outfile)
                        output, error = run_external_process(process)
                # print("BRU THIS IS RIDICULOUT")
                bashCommand = f'python3 {argdict["pathDataAdd"]}/CVAE_Classic/inference.py --data_dir {argdict["pathDataAdd"]}/CVAE_Classic/data -c {dir}' \
                              f' -n {num_generated} -ls {argdict["latent_size"]} -hs {argdict["hidden_size_algo"]} -ed {argdict["dropout_algo"]} -nl {argdict["nb_layers_algo"]}' \
                              f' -rnn {argdict["rnn_type_algo"]} -wd {argdict["word_dropout"]} {bi} -cat {ind} -nc {len(argdict["categories"])}'

                with open(f"{argdict['pathDataAdd']}/GeneratedData/CVAE_Classic/{data}/{dataset_size}/{folderNumGen}/{cat}.txt", "a") as outfile:
                    #TODO UNDO HERE SO IT GOES TO STDOUT
                    process = subprocess.Popen(bashCommand.split(), stdout=outfile)
                    output, error = run_external_process(process)
    elif argdict['algo']=='CBERT':
        # Delete old generated data
        print(argdict['retrain'])
        if argdict['retrain']:
            try:
                for files in os.listdir(
                        f'{argdict["pathDataAdd"]}/GeneratedData/CBERT/{data}/{dataset_size}/{folderNumGen}'):
                    os.remove(
                        f'{argdict["pathDataAdd"]}/GeneratedData/CBERT/{data}/{dataset_size}/{folderNumGen}/{files}')
            except:
                pass
            try:
                for folders in os.listdir(f'{argdict["pathDataAdd"]}/cbert_model'):
                    shutil.rmtree(f'{argdict["pathDataAdd"]}/cbert_model/{folders}')
            except:
                pass
        # argdict['cat'] = cat
        num_points = process_data(argdict)
        # print(num_points)
        num_points = int(num_points)
        # Generate the maximum number of points, that is 5 times the dataset per class
        num_generated = round(num_points * 5)
        #Generate more points than necessary
        argdict['num_to_add'] = round(num_points * split / len(categories))#round(num_points * split / len(categories))

        if argdict['dataset'] in ['SST-2', 'FakeNews', "Irony", "Subj"]:
            datasetCBERT='binaryData'
        elif argdict['dataset'] in ['TREC6']:
            datasetCBERT='SixClassData'
        elif argdict['dataset'] in ['IronyB']:
            datasetCBERT='FourClassData'
        else:
            raise ValueError("Dataset Not Found")

        if argdict['retrain']:
            # param=json.load(open(f"{argdict['pathDataAdd']}/CBERT/global.config", "r"))
            with open(f"{argdict['pathDataAdd']}/CBERT/global.config", "r") as f:
                param = json.load(f)
            # print(param)
            param["dataset"]=datasetCBERT
            param["epoch"]=argdict['nb_epoch_algo']
            json.dump(param, open(f"{argdict['pathDataAdd']}/CBERT/global.config", "w"))
            # print(param)
            # fds
            # #Training VAE
            if argdict['computer']!='labo25':
                venv='venvCBERT'
                pyt='python3'
            elif argdict['computer'] == 'labo25':
                venv='venvCBERT25'
                pyt='python3.7'

            bashCommand=f"{argdict['pathDataAdd']}/{venv}/bin/{pyt} CBERT/cbert_finetune.py --task_name {datasetCBERT} --data_dir {argdict['pathDataAdd']}/CBERT/datasets --seed " \
                        f"{argdict['random_seed']} --num_train_epochs {argdict['nb_epoch_algo']} --do_lower_case --train_batch_size {argdict['bs_algo']}"
            process = subprocess.Popen(bashCommand.split())
            output, error = run_external_process(process)


            bashCommand = f"{argdict['pathDataAdd']}/{venv}/bin/{pyt} {argdict['pathDataAdd']}/CBERT/cbert_augdata.py --data_dir {argdict['pathDataAdd']}/CBERT/datasets --output_dir {argdict['pathDataAdd']}/CBERT/aug_data --num_train_epochs 1 --seed " \
                        f"{argdict['random_seed']} --do_lower_case --num_aug {argdict['split']}"
            process = subprocess.Popen(bashCommand.split())
            output, error = run_external_process(process)

    elif argdict['algo']=='GAN':

        for cat in categories:
            if argdict['retrain']:
                try:
                    os.remove(f'{argdict["pathDataAdd"]}/TextGAN/dataset/test_iw_dict.txt')
                    os.remove(f'{argdict["pathDataAdd"]}/TextGAN/dataset/test_wi_dict.txt')
                except:
                    pass
            argdict['cat'] = cat
            num_points = process_data(argdict)
            num_points = int(num_points)
            # Generate the maximum number of points, that is 1 times the dataset per class
            argdict['num_to_add'] = math.ceil(num_points * split / len(categories))
            if argdict['retrain']:
                # #Training GAN - this saves the results in samples.txt
                bashCommand = f'python3 {argdict["pathDataAdd"]}/TextGAN/run/run_seqgan.py --job_id 3 --gpu_id 0 --path {argdict["pathDataAdd"]} --nb_epochs {argdict["nb_epoch_algo"]} --batch_size {argdict["bs_algo"]}'
                process = subprocess.Popen(bashCommand.split())
                output, error = run_external_process(process)
                # Move generated sentences to generatedData
                os.rename(f"{argdict['pathDataAdd']}/GeneratedData/GAN/samples.txt", f"{argdict['pathDataAdd']}/GeneratedData/GAN/{data}/{dataset_size}/{folderNumGen}/{cat}.txt")

    elif argdict['algo']=='GAN2':

        for cat in categories:
            if argdict['retrain']:
                try:
                    os.remove(f'{argdict["pathDataAdd"]}/TextGAN2/dataset/test_iw_dict.txt')
                    os.remove(f'{argdict["pathDataAdd"]}/TextGAN2/dataset/test_wi_dict.txt')
                except:
                    pass
            argdict['cat'] = cat
            num_points = process_data(argdict)
            num_points = int(num_points)
            # Generate the maximum number of points, that is 1 times the dataset per class
            argdict['num_to_add'] = math.ceil(num_points * split / len(categories))
            if argdict['retrain']:
                # #Training GAN - this saves the results in samples.txt
                bashCommand = f'python3 {argdict["pathDataAdd"]}/TextGAN2/run/run_seqgan.py --job_id 3 --gpu_id 0 --path {argdict["pathDataAdd"]} --nb_epochs {argdict["nb_epoch_algo"]} --batch_size {argdict["bs_algo"]}'
                process = subprocess.Popen(bashCommand.split())
                output, error = run_external_process(process)
                # Move generated sentences to generatedData
                os.rename(f"{argdict['pathDataAdd']}/GeneratedData/GAN2/samples.txt", f"{argdict['pathDataAdd']}/GeneratedData/GAN2/{data}/{dataset_size}/{folderNumGen}/{cat}.txt")
    elif argdict['algo']=='GAN3':

        for cat in categories:
            if argdict['retrain']:
                try:
                    os.remove(f'{argdict["pathDataAdd"]}/TextGAN3/dataset/test_iw_dict.txt')
                    os.remove(f'{argdict["pathDataAdd"]}/TextGAN3/dataset/test_wi_dict.txt')
                except:
                    pass
            argdict['cat'] = cat
            num_points = process_data(argdict)
            num_points = int(num_points)
            # Generate the maximum number of points, that is 1 times the dataset per class
            argdict['num_to_add'] = round(num_points * split / len(categories))
            if argdict['retrain']:
                # #Training GAN - this saves the results in samples.txt
                bashCommand = f'python3 {argdict["pathDataAdd"]}/TextGAN3/run/run_seqgan.py --job_id 3 --gpu_id 0 --path {argdict["pathDataAdd"]} --nb_epochs {argdict["nb_epoch_algo"]} --batch_size {argdict["bs_algo"]}'
                process = subprocess.Popen(bashCommand.split())
                output, error = run_external_process(process)
                # Move generated sentences to generatedData
                os.rename(f"{argdict['pathDataAdd']}/GeneratedData/GAN3/samples.txt", f"{argdict['pathDataAdd']}/GeneratedData/GAN3/{data}/{dataset_size}/{folderNumGen}/{cat}.txt")

    elif argdict['algo'] == 'CATGAN':
        #Need: Center file test.txt with all data + separate files with data per category
        # for pola in ['pos', 'neg']:
        for folders in os.listdir(f'{argdict["pathDataAdd"]}/CatGAN/save'):
            shutil.rmtree(f'{argdict["pathDataAdd"]}/CatGAN/save/{folders}')

        if argdict['retrain']:
            try:
                os.remove(f'{argdict["pathDataAdd"]}/CatGAN/dataset/test_iw_dict.txt')
                os.remove(f'{argdict["pathDataAdd"]}/CatGAN/dataset/test_wi_dict.txt')
            except:
                pass
        # argdict['polarity'] = pola
        num_points = process_data(argdict)
        num_points = int(num_points)
        # Generate the maximum number of points, that is 1 times the dataset per class
        argdict['num_to_add'] = round(num_points * split / len(categories))
        if argdict['retrain']:
            # #Training GAN - this saves the results in samples.txt
            bashCommand = f'python3 {argdict["pathDataAdd"]}/CatGAN/run/run_catgan.py --job_id 6 --gpu_id 0 --path {argdict["pathDataAdd"]} --nb_epochs {argdict["nb_epoch_algo"]} --nb_cat {len(argdict["categories"])}' \
                          f' --batch_size {argdict["bs_algo"]}'
            process = subprocess.Popen(bashCommand.split())
            output, error = run_external_process(process)



    elif argdict['algo']=='W2V':

        for cat in argdict['categories']:
            argdict['cat'] = cat
            num_points = process_data(argdict)
            print(f"Number of points in the dataset {num_points}")
            num_points = int(num_points)
            print(num_points)
            # Generate the maximum number of points, that is 5 times the dataset per class
            argdict['num_to_add'] = math.ceil(num_points * split / len(argdict['categories']))
            # augment data
            if argdict['retrain']:
                augment_w2v(argdict)

    elif argdict['algo']=="TRANS":
        for cat in argdict['categories']:
            argdict['cat'] = cat
            num_points = process_data(argdict)
            print(f"Number of points in the dataset {num_points}")
            num_points = int(num_points)
            # Generate the maximum number of points, that is 5 times the dataset per class
            argdict['num_to_add'] = round(num_points * split/len(argdict['categories']))
            # augment data
            if argdict['retrain']:
                augment_Trans(argdict)
    elif argdict['algo']=='EDA':
        split=argdict['split'] #Percent of data points added
        num_points=process_data(argdict)
        num_points = int(num_points)
        # Generate the maximum number of points, that is 5 times the dataset per class
        argdict['num_to_add'] = math.ceil(num_points * split/len(argdict['categories']))
        num_aug=math.ceil(split)
        # Creating new dataset
        if argdict["retrain"] and argdict['split']!=0:
            bashCommand = f'python3 {argdict["pathDataAdd"]}/EDA/code/augment.py --input={argdict["pathDataAdd"]}/EDA/data/input.txt --output={argdict["pathDataAdd"]}/GeneratedData/{argdict["algo"]}/{data}/{dataset_size}/{folderNumGen}/data.txt --num_aug={num_aug}' \
                          f' --alpha_sr={argdict["replace_EDA"]} --alpha_ri={argdict["insert_EDA"]} --alpha_rs={argdict["swap_EDA"]} --alpha_rd={argdict["delete_EDA"]}'
            process = subprocess.Popen(bashCommand.split())
            output, error = run_external_process(process)
        #Add data to dataframe

    elif argdict['algo']=="AEDA":
        split = argdict['split']  # Percent of data points added
        num_points = process_data(argdict)
        num_points = int(num_points)
        argdict['num_to_add'] = round(num_points * split / len(categories))
        if argdict["retrain"] and argdict['split'] != 0:
            from AEDA.AEDA import AEDA
            alg=AEDA(argdict)
            alg.augment(output_file=f'{argdict["pathDataAdd"]}/GeneratedData/{argdict["algo"]}/{data}/{dataset_size}/{folderNumGen}/data.txt')

    elif argdict['algo']=='GPT':
            # Delete old generated data
            if argdict['retrain']:
                try:
                    for files in os.listdir(f'{argdict["pathDataAdd"]}/GeneratedData/{argdict["algo"]}/{data}/{dataset_size}/{folderNumGen}'):
                        os.remove(f'{argdict["pathDataAdd"]}/GeneratedData/{argdict["algo"]}/{data}/{dataset_size}/{folderNumGen}/{files}')
                except:
                    pass
            num_points = process_data(argdict)
            num_points = int(num_points)
            num_generated = round(num_points)
            argdict['num_to_add'] = math.ceil(num_points * split / len(categories))

            if argdict['retrain']:
                bashCommand = f'python3 {argdict["pathDataAdd"]}/gpt/train.py --data_file {argdict["pathDataAdd"]}/gpt/data/data.txt --batch {argdict["bs_algo"]} --model_name {argdict["pathDataAdd"]}/temp/gpt' \
                              f' --epoch {argdict["nb_epoch_algo"]}'
                process = subprocess.Popen(bashCommand.split())
                output, error = run_external_process(process)

                for cat in argdict['categories']:
                    print(cat)
                    ind=argdict['categories'].index(cat)
                    bashCommand = f'python3 {argdict["pathDataAdd"]}/gpt/generate.py --model_name {argdict["pathDataAdd"]}/temp/gpt.pt --sentence {num_generated} --label {cat}'  # GeneratedData/{data}/{pola}.txt'

                    with open(f"{argdict['pathDataAdd']}/GeneratedData/GPT/{data}/{dataset_size}/{folderNumGen}/{cat}.txt", "a") as outfile:
                        process = subprocess.Popen(bashCommand.split(), stdout=outfile)
                        output, error = run_external_process(process)

    else:
        raise Exception
    if argdict['retrain'] and argdict['split']!=0:
        post_process_data(argdict)
    add_data(argsdict)

    #Classify
    if argdict['classifier']=='jiantLstm':
        bashCommand = f'python3 ../run_glue_jiantLSTM.py --dataset {argdict["dataset"]}' #GeneratedData/{data}/{pola}.txt'
        process = subprocess.Popen(bashCommand.split())
        output, error = run_external_process(process)
    # elif argdict['classifier']=='lstm':
    #     acc_train, acc_dev=run_glue_lstm.run_lstm(argdict)
    elif argdict['classifier']=='dan':
        acc_train, acc_dev, f1_train, f1_dev=run_glue_DAN.run_DAN(argdict)
    elif argdict['classifier']=='xgboost':
        acc_train, acc_dev, f1_train, f1_dev = run_glue_xgboost.run_xgboost(argdict)
    elif argdict['classifier']=='bert':
        import run_glue_bert
        shutil.rmtree(f"{argdict['pathDataAdd']}/content/exp/cache", ignore_errors=True)
        acc_train, acc_dev, f1_train, f1_dev = run_glue_bert.run_bert(argdict)
    elif argdict['classifier']=='bert2':
        import run_glue_bert2
        shutil.rmtree(f"{argdict['pathDataAdd']}/content/exp2/cache", ignore_errors=True)
        acc_train, acc_dev, f1_train, f1_dev = run_glue_bert2.run_bert(argdict)
    elif argdict['classifier']=='dummy':
        acc_train, acc_dev, f1_train, f1_dev= 0.5, 0.5, 0.5, 0.5
    else:
        raise ValueError("Classifier not found")
    print(f"Logging in {tup}")
    log_results(argdict, {'acc_train':acc_train, 'acc_dev':acc_dev, 'f1_train':f1_train, 'f1_dev':f1_dev}, tup)
    return acc_train, acc_dev, f1_train, f1_dev


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE for data augmentation')
    #General arguments on training
    parser.add_argument('--dataset', type=str, default='SST-2', help="dataset you want to run the process on. Includes SST-2, TREC6, FakeNews, QNLI, Irony, and IronyB")
    parser.add_argument('--classifier', type=str, default='dummy', help="classifier you want to use. Includes bert, lstm, dan, jiantLstm or xgboost")
    parser.add_argument('--computer', type=str, default='home', help="Whether you run at home or at iro. Automatically changes the base path")
    parser.add_argument('--split', type=float, default=1.0, help='percent of the dataset added')
    parser.add_argument('--retrain', action='store_true', help='whether to retrain the VAE or not')
    parser.add_argument('--rerun', action='store_true', help='whether to rerun knowing that is has already been ran')
    parser.add_argument('--rerun_split_zero', action='store_true', help='whether to rerun split zero knowing that is has already been ran')
    parser.add_argument('--dataset_size', type=int, default=10, help='number of example in the original dataset. If 0, use the entire dataset')
    parser.add_argument('--algo', type=str, default='EDA', help='data augmentation algorithm to use, includes, EDA, W2V, TRANS, GAN, CATGAN, VAE, VAE_ED, CVAE, CBERT, and GPT')
    parser.add_argument('--random_seed', type=int, default=7, help='Random seed ')
    parser.add_argument('--max_length', type=int, default=0, help='maximum length of the sentences')
    parser.add_argument('--starting_set_seed', type=int, default=0, help='maximum length of the sentences')
    parser.add_argument('--vary_starting_set', action='store_true')
    parser.add_argument('--second_dataset', action='store_true', help='Run the algorithm on the second dataset selected (starting set seed=1)')
    parser.add_argument('--new_equilibre', action='store_true', help='rerun the algorithm split 0 by balancing correctly the classes')

    #Algo specific arguments
    parser.add_argument('--nb_epoch_algo', type=int, default=10, help="Number of epoch for which to run the algo when applicable")
    parser.add_argument('--bs_algo', type=int, default=32, help="Batch Size Algo")
    parser.add_argument('--nb_layers_algo', type=int, default=1, help="Number of layers for the algo when applicable")
    parser.add_argument('--unidir_algo', action='store_true', default=False, help="When applicable, wether to have a unidir algo (VAE, CVAE)")
    parser.add_argument('--hidden_size_algo', type=int, default=512, help="Hidden Size for the algo when applicable (VAE, CVAE, GAN, CatGAN)")
    parser.add_argument('--replace_EDA', type=float, default=0.1, help="Percent of the sentence to be replaced")
    parser.add_argument('--insert_EDA', type=float, default=0.1, help="Percent of the sentence to be inserted")
    parser.add_argument('--swap_EDA', type=float, default=0, help="Percent of the sentence to be swapped")
    parser.add_argument('--delete_EDA', type=float, default=0.1, help="Percent of the sentence to be delete")
    parser.add_argument('--latent_size', type=int, default=100, help="Size of the latent space for the VAE, CVAE, or GANs")
    parser.add_argument('--bool_W2V', action='store_true', help="v parameter for W2V")
    parser.add_argument('--dropout_algo', type=float, default=0.75, help="dropout for the algo")
    parser.add_argument('--rnn_type_algo', type=str, default='gru', help="type of rnn for some algo (VAE, CVAE). Choice of gru, lstm, or rnn")
    parser.add_argument('--word_dropout', type=float, default=0, help="Word dropout for some algo (VAE, CVAE)")
    parser.add_argument('--k', default=0.0025, type=float, help='k parameter when needed (VAE, CVAE)')
    parser.add_argument('--x0', default=2500, type=int, help='x0 parameter when needed (VAE, CVAE)')



    #Classifier specific arguments
    parser.add_argument('--nb_epochs_lstm', type=int, default=10, help="Number of epoch to train the lstm with")
    parser.add_argument('--dropout_classifier', type=int, default=0.3, help="dropout parameter of the classifier")
    parser.add_argument('--hidden_size_classifier', type=int, default=128, help="dropout parameter of the classifier")


    #Cleaning
    parser.add_argument('--clean', action='store_true', help="Cleaning function of the folders")

    #Experiments
    parser.add_argument('--test_dss_vs_split', action='store_true', help='test the influence of the dataset size and ')
    parser.add_argument('--test_epochLstm_vs_split', action='store_true', help='test the influence of the number of epoch for the lstm classifier')
    parser.add_argument('--test_dropoutLstm_vs_split', action='store_true', help='test the influence of the dropout for the lstm classifier')
    parser.add_argument('--test_hsLstm_vs_split', action='store_true', help='test the influence of the dropout for the lstm classifier')
    parser.add_argument('--test_latent_size_vs_split', action='store_true', help='test the influence of the latent size')
    parser.add_argument('--test_randomSeed', action='store_true', help='test the influence of the random seed for LSTM')
    parser.add_argument('--run_ds_split', action='store_true', help='Run all DS and splits with specified arguments')
    parser.add_argument('--test_one_shot', action='store_true', help='test a one shot combination')
    parser.add_argument('--tsne_visualize', action='store_true', help="Run visualization on the original and generated data")
    parser.add_argument('--pca_visualize', action='store_true', help="Run visualization on the original and generated data with PCA")
    parser.add_argument('--lime', action='store_true', help="perform lime analysis")
    parser.add_argument('--doublons', action='store_true', help="perform doublons analysis")
    parser.add_argument('--test_hidden_size_algo_vs_split', action='store_true', help='test the influence of the hidden size on the algorithm')
    parser.add_argument('--print_results', action='store_true', help='simply run an experiment and prints the result')
    parser.add_argument('--grid_search_EDA', action='store_true', help='grid search parameter for EDA')
    parser.add_argument('--get_all_results', action='store_true', help='outputs the results of the experiments in a separate dataset')
    parser.add_argument('--quantify_std_seed', action='store_true', help='Quantify the variation brought by the starting set')
    parser.add_argument('--test_split_perfo', action='store_true', help='quantify the impact of the plit parameter. ')

    args = parser.parse_args()

    argsdict = args.__dict__
    if argsdict['computer']=='home':
        argsdict['pathDataOG'] = "/media/frederic/DAGlue/data"
        argsdict['pathDataAdd'] = "/media/frederic/DAGlue"
    elif argsdict['computer']=='labo' or argsdict['computer']=='labo25':
        argsdict['pathDataOG'] = "/data/rali5/Tmp/piedboef/data/jiantData/OG/OG"
        argsdict['pathDataAdd'] = "/u/piedboef/Documents/DAGlue"
    # Create directories for the runs
    # Categories
    if argsdict['dataset'] == "SST-2":
        categories = ["neg", "pos"]
    elif argsdict['dataset'] == "TREC6":
        categories = ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]
    elif argsdict['dataset'] == "FakeNews":
        categories = ["Fake", "Real"]
    elif argsdict['dataset'] == "QNLI":
        categories = ["entailment", "not_entailment"]
    elif argsdict['dataset'] == "Irony":
        categories= ["NotIro", "Iro"]
    elif argsdict['dataset'] == "IronyB":
        categories = ["Clash", "Situational", "Other", "NotIro"]
    elif argsdict['dataset']=="Subj":
        categories = ["Objective", "Subjective"]
    else:
        raise ValueError("Dataset not found")
    argsdict['categories'] = categories

    if argsdict['classifier'] == 'bert' and argsdict['dataset'] == "TREC6" and argsdict['dataset_size'] in [100,500]:
        argsdict['nb_epochs_lstm'] = 20
    elif argsdict['classifier'] == 'bert' and argsdict['dataset'] == "TREC6" and argsdict['dataset_size'] in [1000]:
        argsdict['nb_epochs_lstm'] = 15
    elif argsdict['classifier'] == 'bert' and argsdict['dataset'] == "IronyB" and argsdict['dataset_size'] in [100]:
        argsdict['nb_epochs_lstm'] = 8
    elif argsdict['classifier'] == 'bert' and argsdict['dataset'] == "IronyB" and argsdict['dataset_size'] in [500]:
        argsdict['nb_epochs_lstm'] = 20
    elif argsdict['classifier'] == 'bert' and argsdict['dataset'] == "IronyB" and argsdict['dataset_size'] in [1000]:
        argsdict['nb_epochs_lstm'] = 10
        # argsdict['nb_epochs_lstm'] = 11
    elif argsdict['classifier'] == 'bert' and argsdict['dataset'] == "IronyB" and argsdict['dataset_size'] in [0]:
        argsdict['nb_epochs_lstm'] = 11



    # json.dump(argsdict, open(f"/media/frederic/DAGlue/SentenceVAE/GeneratedData/{argsdict['dataset']}/{argsdict['dataset_size']}/parameters.json", "w"))


    print("=================================================================================================")
    print(argsdict)
    print("=================================================================================================")

    if argsdict['clean']:
        cleanup(argsdict)
    elif argsdict['test_dss_vs_split']:
        dico={'help':'key is dataset size, then key inside is split and value is a tuple of train val accuracy'}
        for ds in [50,500,5000,50000]:
            dicoTemp={}
            for split in [0, 0.2,  0.4, 0.6, 0.8, 1]:
                print(f"Dataset size {ds}, split {split}")
                argsdict['dataset_size']=ds
                argsdict['split']=split
                # argsdict['retrain']=False
                # createFoldersEDA(argsdict)
                acc_train, acc_val=main(argsdict)
                dicoTemp[split]=acc_val
            dico[ds]=dicoTemp
            # with open(f"/media/frederic/DAGlue/Experiments/DSSVSSplit/EDA.json", "w") as f:
            #     json.dump(dico, f)
            plot_graph(dico, "dss", argsdict)
            print_samples(argsdict)
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
        for dsSize in [50,500,1000, 5000]:
            for split in [0, 0.2, 0.5, 1.0]:
                print(f"Running with dataset of {dsSize} and split of {split}")
                argsdict['retrain']=False
                argsdict['dataset_size']=dsSize
                argsdict['split']=split
                acc_train, acc_val=main(argsdict)
    elif argsdict['print_results']:
        argsdict = createFolders(argsdict)
        tt=run(argsdict)
        print_samples(argsdict)
    elif argsdict['tsne_visualize'] and argsdict['classifier']=='xgboost':
        argsdict = createFolders(argsdict)
        tt=run(argsdict)
        tsne_visualize(argsdict)
    elif argsdict['tsne_visualize'] and argsdict['classifier'] == 'dan':
        argsdict = createFolders(argsdict)
        tt = run(argsdict)
        tsne_visualize_dan_linear(argsdict)
    elif argsdict['tsne_visualize'] and argsdict['classifier']=='bert':
        argsdict = createFolders(argsdict)
        tt=run(argsdict)
        tsne_visualize_bert_linear(argsdict)
    elif argsdict['tsne_visualize'] and argsdict['classifier'] == 'dummy':
        raise ValueError("No visualization for the dummy classifier")
    elif argsdict['pca_visualize'] and argsdict['classifier']=='xgboost':
        argsdict = createFolders(argsdict)
        tt=run(argsdict)
        pca_visualize(argsdict)
    elif argsdict['pca_visualize'] and argsdict['classifier']=='bert':
        argsdict = createFolders(argsdict)
        tt=run(argsdict)
        pca_visualize_bert(argsdict)
    elif argsdict['lime'] and argsdict['classifier']=='xgboost':
        argsdict = createFolders(argsdict)
        tt = run(argsdict)
        from Analysis.LIME.lime_analysis import *
        lime_xgboost(argsdict)
    elif argsdict['lime'] and argsdict['classifier']=='bert':
        argsdict = createFolders(argsdict)
        tt = run(argsdict)
        from Analysis.LIME.lime_analysis import *
        lime_bert(argsdict)
    elif argsdict['lime']:
        raise ValueError("No lime protocol with this classifier")
    elif argsdict['doublons']:
        argsdict = createFolders(argsdict)
        tt = run(argsdict)
        from Analysis.Doublons import doublons
        result, length, length_overall=doublons(argsdict)
        with open('tefdasfdasfdsadsfa', 'w') as f:
            f.write(str(result))
        with open('tefdasfdasfdsadsfalength', 'w') as f:
            f.write(str(length))
        with open('tefdasfdasfdsadsfalength_overall', 'w') as f:
            f.write(str(length_overall))
    elif argsdict['test_one_shot']:
        # argdict['random_seed'] = seed
        # if seed != 3:
        #     If its not the first time overwrite the retrain
            # argdict['retrain'] = False
        argsdict['retrain']=True
        argsdict['rerun']=True
        argsdict = createFolders(argsdict)
        tt = run(argsdict)
    elif argsdict['get_all_results']:
        argsdict=createFolders(argsdict)
        output_results(argsdict)
        # return [4,5,6]
    elif argsdict['quantify_std_seed']:
        starting_sizes=[10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        argsdict['vary_starting_set']=True
        results={}
        for sd in starting_sizes:
            argsdict['split']=1
            argsdict['dataset_size']=sd
            acc_train, acc_dev, f1_train, f1_dev = main(argsdict)
            # print("Result")
            # acc, _, f1, _ = getAllResult(argsdict)
            array_augmented = getArray(argsdict)
            # print(f"Dev accuracy {acc} F1 {f1}")
            # print("Baseline")
            argsdict['split'] = 0
            # acc, _, f1, _ = getAllResult(argsdict)
            # print(f"Dev accuracy {acc} F1 {f1}")
            array_baseline = getArray(argsdict)
            diffs = []
            # print(array_augmented, array_baseline)
            for aug, bas in zip(array_augmented, array_baseline):
                    diffs.append(aug['acc_dev'] - bas['acc_dev'])
            diffs = np.array(diffs)
            std=np.std(diffs)
            results[sd]=std
        with open(f"Analysis/STD_vs_starting_size.json", "w") as f:
            json.dump(results, f)
    elif argsdict['test_split_perfo']:
        splits=[0.5, 1, 2, 3, 4, 5]
        for split in splits:
            argsdict['split']=split
            acc_train, acc_dev, f1_train, f1_dev = main(argsdict)
            # print(argsdict['split'])
            print_samples(argsdict)
            # print(argsdict['split'])
            print("Result")
            acc, _, f1, _ = getAllResult(argsdict)
            array_augmented = getArray(argsdict)
            print(f"Dev accuracy {acc} F1 {f1}")
            print("Baseline")
            argsdict['split'] = 0

            acc, _, f1, _ = getAllResult(argsdict)
    else:
        # argsdict['numFolderGenerated']=0
        # add_data(argsdict, "GPT")
        acc_train, acc_dev, f1_train, f1_dev=main(argsdict)
        # print(argsdict['split'])
        print_samples(argsdict)
        # print(argsdict['split'])
        print("Result")
        acc, _, f1, _=getAllResult(argsdict)
        array_augmented=getArray(argsdict)
        print(f"Dev accuracy {acc} F1 {f1}")
        print("Baseline")
        argsdict['split']=0
        # if argsdict['classifier'] == 'bert' and argsdict['dataset'] == "TREC6":
        #     argsdict['nb_epochs_lstm'] = 8
        acc, _, f1, _ = getAllResult(argsdict)
        print(f"Dev accuracy {acc} F1 {f1}")
        array_baseline=getArray(argsdict)
        if argsdict['vary_starting_set']:
            diffs=[]
            # print(array_augmented, array_baseline)
            for aug, bas in zip(array_augmented, array_baseline):
                if argsdict['classifier'] not in ['bert', 'xgboost']:
                    diffs.append(max(aug['acc_dev'])-max(bas['acc_dev']))
                else:
                    diffs.append(aug['acc_dev']-bas['acc_dev'])
            diffs=np.array(diffs)
            print(f"Average augmentation: {np.mean(diffs)}, deviation standard: {np.std(diffs)}, min: {np.min(diffs)}, max: {np.max(diffs)}")
#Random seeds: 3,7,9,13,100, -500, 512, 12, 312, 888