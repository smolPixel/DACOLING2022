"Create pos or neg files for training of the SVAE"

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score,matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import argparse, os, json, shutil
from pathlib import Path
import re
import math

ImportantAlgo = ["nb_epoch_algo", "random_seed", "algo", "dataset_size", "unidir_algo", "hidden_size_algo",
          "replace_EDA", "insert_EDA", "swap_EDA", "delete_EDA", "latent_size", "nb_layers_algo",
          "bool_W2V", "dropout_algo", "rnn_type_algo", "word_dropout", "k", "x0", "bs_algo", "starting_set_seed", "split"]

ImportantSplitZero=["nb_epochs_lstm", "dropout_classifier", "hidden_size_classifier", "split", "random_seed", "dataset_size", "max_length", "starting_set_seed"]
ImportantClassifier=ImportantAlgo+ImportantSplitZero

random_seeds=[0,1,2,3,4,5,6,7,8,9,
              10,11,12,13,14]#,15,16,17,18,19,
              # 20,21,22,23,24,25,26,27,28,29]

def read_fucked_files(file, skip_header=False):
    # Take in a file name and return an array of array containing the columns in order
    fp = open(file, 'r')
    arr = None
    for j, line in enumerate(fp.readlines()):
        if j == 0 and skip_header:
            continue
        line = line.split('\t')
        if arr is None:
            arr = []
            for _ in line:
                arr.append([])
        for i, ll in enumerate(line):
            arr[i].append(ll)
    return arr


def fuse_sentences(L1, L2):
    arr = []
    for l1, l2 in zip(L1, L2):
        arr.append(l1 + " [SEP] " + l2)
    return arr

def sample_class(df, i, prop, argdict):
    """Sample the class i from the dataframe, with oversampling if needed. """
    size_class=len(df[df['label'] == i])
    ds=argdict['dataset_size'] if argdict['dataset_size']!=0 else len(df)
    num_sample_tot=math.ceil(ds * prop)
    #Sample a first time
    num_sample = min(num_sample_tot, size_class)
    sample = df[df['label'] == i].sample(n=num_sample)
    num_sample_tot-=num_sample
    while num_sample_tot!=0:
        num_sample=min(num_sample_tot, size_class)
        sampleTemp=df[df['label'] == i].sample(n=num_sample)
        sample = pd.concat([sample, sampleTemp])
        num_sample_tot-=num_sample
    return sample

def get_dataFrame(argdict):
    """Get the dataframe for the particular split. If it does not exist: create it"""
    if argdict['dataset']=='SST2':
        task="SST-2"
    else:
        task=argdict['dataset']

    create_train=False

    if argdict['vary_starting_set'] or argdict['second_dataset']:
        dfName=f'train_{argdict["starting_set_seed"]}'
    else:
        dfName='train'

    if argdict['max_length']==0:
        path=f"{argdict['pathDataAdd']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}"
        dfVal = pd.read_csv(f'{argdict["pathDataOG"]}/{task}/dev.tsv', sep='\t')
    else:
        path = f"{argdict['pathDataAdd']}/SelectedDataMaxLength/{argdict['max_length']}/{argdict['dataset']}/{argdict['dataset_size']}"
        dfVal = pd.read_csv(f'{argdict["pathDataOG"]}/{task}/dev.tsv', sep='\t')
        dfVal = dfVal[dfVal.sentence.apply(lambda x: len(str(x).split()) < argdict['max_length'])]

    try:
        dfTrain=pd.read_csv(f"{path}/{dfName}.tsv", sep='\t')
    except:
        create_train=True


    if create_train:
        dfTrain=pd.read_csv(f'{argdict["pathDataOG"]}/{task}/train.tsv', sep='\t').dropna(axis=1)


    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    if create_train:
        #Sampling balanced data
        # prop=len(dfTrain[dfTrain['label']==0])/len(dfTrain)
        if argdict['max_length']!=0:
            #Keep only in dfTrain the examples that have a lenght (defined by split()) less than max_length
            dfTrain=dfTrain[dfTrain.sentence.apply(lambda x: len(str(x).split())<argdict['max_length'])]
            print(f"len dfVal: {len(dfVal)}")
            dfVal = dfVal[dfVal.sentence.apply(lambda x: len(str(x).split()) < argdict['max_length'])]
            print(f"len dfVal: {len(dfVal)}")
        #TODO HERE
        if argdict['dataset_size']==0:
            max_size = dfTrain['label'].value_counts().max()
            prop=max_size/len(dfTrain)
        else:
            prop=1/len(argdict['categories'])
        # NewdfTrain=dfTrain[dfTrain['label']==0].sample(n=math.ceil(argdict['dataset_size']*prop))
        NewdfTrain=sample_class(dfTrain, 0, prop, argdict)
        for i in range(1, len(argdict['categories'])):
            prop = len(dfTrain[dfTrain['label'] == i]) / len(dfTrain)
            # TODO HERE
            prop = 1 / len(argdict['categories'])
            NewdfTrain=pd.concat([NewdfTrain ,sample_class(dfTrain, i, prop, argdict)])
        # g=dfTrain.grouby('label')
        # dfTrain=g.apply(lambda x: x.sample(int(argdict['dataset_size']/2)).reset_index(drop=True))
        # dfTrain=dfTrain.sample(n=argdict['dataset_size'])
        dfTrain=NewdfTrain
        Path(path).mkdir(parents=True, exist_ok=True)
        dfTrain.to_csv(f"{path}/{dfName}.tsv", sep='\t')
    # fdasklj
    print(f"Length of the dataframe {len(dfTrain)}")
    return dfTrain, dfVal


def process_data(argdict):
    # count_vect = CountVectorizer()
    # tfidf_transformer = TfidfTransformer()

    #Tasks in SST-2, CoLA, MNLI, MRPC, QNLI, QQP, RTE, SNLI, STS-B, WNLI

    if argdict['dataset']=='SST2':
        task="SST-2"
    else:
        task=argdict['dataset']

    dfTrain, dfVal=get_dataFrame(argdict)

    XTrain=dfTrain['sentence']
    YTrain=dfTrain['label']
    #Separate by category
    Xcats=[XTrain[YTrain==i] for i in range(len(argdict['categories']))]
    # XPos=XTrain[YTrain==1]
    # XNeg=XTrain[YTrain==0]
    XVal=dfVal['sentence']
    YVal=dfVal['label']
    XcatsVal = [XVal[YVal == i] for i in range(len(argdict['categories']))]
    # elif task in ["MRPC"]:
    #     XTrain=fuse_sentences(dfTrain[3], dfTrain[4])
    #     YTrain=[int(i) for i in dfTrain[0]]
    #     XVal=fuse_sentences(dfVal[3], dfVal[4])
    #     YVal=[int(i) for i in dfVal[0]]
    # elif task in ["QNLI"]:
    #     XTrain=fuse_sentences(dfTrain[1], dfTrain[2])
    #     YTrain=[i.strip() for i in dfTrain[3]]
    #     XVal=fuse_sentences(dfVal[1], dfVal[2])
    #     YVal=[i.strip() for i in dfVal[3]]
    # elif task in ["STS-B"]:
    #     XTrain=fuse_sentences(dfTrain[7], dfTrain[8])
    #     YTrain=[float(i) for i in dfTrain[9]]
    #     XVal=fuse_sentences(dfVal[7], dfVal[8])
    #     YVal=[float(i) for i in dfVal[9]]
    # elif task in ["QQP"]:
    #     XTrain=fuse_sentences(dfTrain["question1"], dfTrain["question2"])
    #     YTrain=dfTrain["is_duplicate"]
    #     XVal = fuse_sentences(dfVal["question1"], dfVal["question2"])
    #     YVal = dfVal["is_duplicate"]
    # elif task in ["RTE", "WNLI"]:
    #     XTrain=fuse_sentences(dfTrain["sentence1"], dfTrain["sentence2"])
    #     YTrain=list(dfTrain["label"])
    #     XVal = fuse_sentences(dfVal["sentence1"], dfVal["sentence2"])
    #     YVal = list(dfVal["label"])
    # elif task in ["MNLI"]:
    #     XTrain=fuse_sentences(dfTrain[8], dfTrain[9])
    #     YTrain=dfTrain[11]
    #     XVal=fuse_sentences(dfVal[8], dfVal[9])
    #     YVal=dfVal[15]
    #     XValMimatched = fuse_sentences(dfValMis[8], dfValMis[9])
    #     YValMismatched = dfValMis[15]


    # dfTrain.to_csv(f"/media/frederic/DAGlue/SentenceVAE/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/train.tsv", sep='\t')
    trainFile=""
    valFile=""
    #Those who take files that contains one class
    if argdict['algo'] in ['VAE', 'GAN', 'GAN2', 'GAN3', 'W2V', 'TRANS']:
        i=argdict['categories'].index(argdict['cat'])
        for line in Xcats[i]:
            trainFile += line + "\n"
        for line in XVal:
            valFile+=line+"\n"

        if argdict['algo']=="VAE":
            file=open(f"{argdict['pathDataAdd']}/SentenceVAE/data/ptb.train.txt", "w")
            file.write(trainFile)
            file.close()
            file=open(f"{argdict['pathDataAdd']}/SentenceVAE/data/ptb.valid.txt", "w")
            file.write(valFile)
            file.close()
            return len(dfTrain)
        elif argdict['algo']=="GAN":
            file=open(f"{argdict['pathDataAdd']}/TextGAN/dataset/test.txt", "w")
            file.write(trainFile)
            file.close()
            file=open(f"{argdict['pathDataAdd']}/TextGAN/dataset/testdata/test_test.txt", "w")
            file.write(valFile)
            file.close()
            return len(dfTrain)
        elif argdict['algo']=="GAN2":
            file=open(f"{argdict['pathDataAdd']}/TextGAN2/dataset/test.txt", "w")
            file.write(trainFile)
            file.close()
            file=open(f"{argdict['pathDataAdd']}/TextGAN2/dataset/testdata/test_test.txt", "w")
            file.write(valFile)
            file.close()
            return len(dfTrain)
        elif argdict['algo']=="GAN3":
            file=open(f"{argdict['pathDataAdd']}/TextGAN3/dataset/test.txt", "w")
            file.write(trainFile)
            file.close()
            file=open(f"{argdict['pathDataAdd']}/TextGAN3/dataset/testdata/test_test.txt", "w")
            file.write(valFile)
            file.close()
            return len(dfTrain)
        elif argdict['algo']=="W2V":
            file = open(f"{argdict['pathDataAdd']}/textaugment/Data/temp.txt", "w")
            file.write(trainFile)
            file.close()
    elif argdict['algo'] in ['CBERT']:
        if argdict['dataset'] in ['SST-2', 'FakeNews', 'Irony', 'Subj']:
            datasetCBERT='binaryData'
        elif argdict['dataset'] in ['TREC6']:
            datasetCBERT='SixClassData'
        elif argdict['dataset'] in ['IronyB']:
            datasetCBERT='FourClassData'
        else:
            raise ValueError("Dataset not found")
        copy=int(argdict['split'])
        rest=argdict['split']%1
        #TODO REST
        dfTemp = dfTrain[['sentence', 'label']]
        dfFinal=dfTrain[['sentence', 'label']]
        # for i in range(copy-1):
        #     dfFinal=dfFinal.append(dfTemp)
        # for i in range(len(argdict['categories'])):
        #     num_sample=int(rest*len(dfTemp)/len(argdict['categories']))
        #     dfClass=dfTemp[dfTemp['label'] == i].sample(n=num_sample)
        #     dfFinal=dfFinal.append(dfClass)
        dfTemp.to_csv(f"{argdict['pathDataAdd']}/CBERT/datasets/{datasetCBERT}/train.tsv", sep='\t', index=False)
        dfFinal.to_csv(f"{argdict['pathDataAdd']}/CBERT/aug_data/{datasetCBERT}/train_origin.tsv", sep='\t', index=False)
        dfFinal.to_csv(f"{argdict['pathDataAdd']}/CBERT/aug_data/{datasetCBERT}/train.tsv", sep='\t', index=False)
        dfFinal = dfVal[['sentence', 'label']]
        dfFinal.to_csv(f"{argdict['pathDataAdd']}/CBERT/datasets/{datasetCBERT}/dev.tsv", sep='\t', index=False)
        # fdas
    elif argdict['algo'] in ['CATGAN', 'GPT', 'CVAE', 'CVAE_Classic', 'EDA', 'VAE_EncDec', 'VAE_LinkedEnc' ,'AEDA']:
        #CatGan needs both the full file as well as the files per categorie
        trainCat=""
        valCat=""
        for cat in argdict['categories']:
            i = argdict['categories'].index(cat)
            if argdict['algo'] == "GPT":
                for line in Xcats[i]:
                    trainFile+=cat + "\t" + line +"\n"
                file = open(f"{argdict['pathDataAdd']}/gpt/data/data.txt", "w")
                file.write(trainFile)
                file.close()
            elif argdict['algo'] == "VAE_EncDec":
                i = argdict['categories'].index(cat)
                for line in Xcats[i]:
                    trainFile += line + "\n"
                    trainCat += str(i) + "\n"
                for line in XVal:
                    valFile += line + "\n"
                    valCat += str(i) + "\n"

                file = open(f"{argdict['pathDataAdd']}/VAE_EncDec/data/ptb.train.txt", "w")
                file.write(trainFile)
                file.close()
                file = open(f"{argdict['pathDataAdd']}/VAE_EncDec/data/ptb.train.label.txt", "w")
                file.write(trainCat)
                file.close()
                file = open(f"{argdict['pathDataAdd']}/VAE_EncDec/data/ptb.dev.txt", "w")
                file.write(valFile)
                file.close()
                file = open(f"{argdict['pathDataAdd']}/VAE_EncDec/data/ptb.dev.label.txt", "w")
                file.write(valCat)
                file.close()
            elif argdict['algo'] == "VAE_LinkedEnc":
                i = argdict['categories'].index(cat)
                for line in Xcats[i]:
                    trainFile += line + "\n"
                    trainCat += str(i) + "\n"
                for line in XVal:
                    valFile += line + "\n"
                    valCat += str(i) + "\n"

                file = open(f"{argdict['pathDataAdd']}/SentenceVAELinkedEnc/data/ptb.train.txt", "w")
                file.write(trainFile)
                file.close()
                file = open(f"{argdict['pathDataAdd']}/SentenceVAELinkedEnc/data/ptb.train.label.txt", "w")
                file.write(trainCat)
                file.close()
                file = open(f"{argdict['pathDataAdd']}/SentenceVAELinkedEnc/data/ptb.valid.txt", "w")
                file.write(valFile)
                file.close()
                file = open(f"{argdict['pathDataAdd']}/SentenceVAELinkedEnc/data/ptb.valid.label.txt", "w")
                file.write(valCat)
                file.close()
            elif argdict['algo']=="CATGAN":
                for line in Xcats[i]:
                    trainFile += line + "\n"
                for line in XVal:
                    valFile += line + "\n"
                file = open(f"{argdict['pathDataAdd']}/CatGAN/dataset/test.txt", "w")
                file.write(trainFile)
                file.close()
                file = open(f"{argdict['pathDataAdd']}/CatGAN/dataset/testdata/test_test.txt", "w")
                file.write(valFile)
                file.close()
                trainCat = ""
                valCat = ""
                for line in Xcats[i]:
                    trainCat+=line +"\n"
                file = open(f"{argdict['pathDataAdd']}/CatGAN/dataset/test_cat{i}.txt", "w")
                file.write(trainCat)
                file.close()
                for line in XcatsVal[i]:
                    valCat+=line +"\n"
                file = open(f"{argdict['pathDataAdd']}/CatGAN/dataset/testdata/test_cat{i}_test.txt", "w")
                file.write(valCat)
                file.close()
            elif argdict['algo']=="CVAE":
                i = argdict['categories'].index(cat)
                for line in Xcats[i]:
                    trainFile += line + "\n"
                    trainCat+= str(i) + "\n"
                for line in XVal:
                    valFile += line + "\n"
                    valCat += str(i) + "\n"

                file = open(f"{argdict['pathDataAdd']}/CVAE/data/ptb.train.txt", "w")
                file.write(trainFile)
                file.close()
                file = open(f"{argdict['pathDataAdd']}/CVAE/data/ptb.train.label.txt", "w")
                file.write(trainCat)
                file.close()
                file = open(f"{argdict['pathDataAdd']}/CVAE/data/ptb.valid.txt", "w")
                file.write(valFile)
                file.close()
                file = open(f"{argdict['pathDataAdd']}/CVAE/data/ptb.valid.label.txt", "w")
                file.write(valCat)
                file.close()

            elif argdict['algo']=="CVAE_Classic":
                i = argdict['categories'].index(cat)
                for line in Xcats[i]:
                    trainFile += line + "\n"
                    trainCat+= str(i) + "\n"
                for line in XVal:
                    valFile += line + "\n"
                    valCat += str(i) + "\n"

                file = open(f"{argdict['pathDataAdd']}/CVAE_Classic/data/ptb.train.txt", "w")
                file.write(trainFile)
                file.close()
                file = open(f"{argdict['pathDataAdd']}/CVAE_Classic/data/ptb.train.label.txt", "w")
                file.write(trainCat)
                file.close()
                file = open(f"{argdict['pathDataAdd']}/CVAE_Classic/data/ptb.valid.txt", "w")
                file.write(valFile)
                file.close()
                file = open(f"{argdict['pathDataAdd']}/CVAE_Classic/data/ptb.valid.label.txt", "w")
                file.write(valCat)
                file.close()
            elif argdict['algo']=="EDA":
                i = argdict['categories'].index(cat)
                for line in Xcats[i]:
                    trainFile += str(i)+ "\t" + line + "\n"
                # for lineNeg, linePos in zip(XNeg, XPos):
                #     trainFile += "0\t" + lineNeg + "\n"
                #     trainFile += "1\t" + linePos + "\n"

                file = open(f"{argdict['pathDataAdd']}/EDA/data/input.txt", "w")
                file.write(trainFile)
                file.close()
            elif argdict['algo']=="AEDA":
                i = argdict['categories'].index(cat)
                for line in Xcats[i]:
                    trainFile += str(i)+ "\t" + line + "\n"
                # for lineNeg, linePos in zip(XNeg, XPos):
                #     trainFile += "0\t" + lineNeg + "\n"
                #     trainFile += "1\t" + linePos + "\n"

                file = open(f"{argdict['pathDataAdd']}/AEDA/data/input.txt", "w")
                file.write(trainFile)
                file.close()
    else:
        raise ValueError("Algorithm not found")
    return len(dfTrain)


def post_process_data(argdict):
    """Post process the data so that it is usable by the classifiers, split into categories so that the same processus can be used after"""
    if argdict['algo']=="CBERT":
        if argdict['dataset'] in ['SST-2', 'FakeNews', 'Irony', 'Subj']:
            ds="binaryData"
        elif argdict['dataset'] in ['IronyB']:
            ds='FourClassData'
        elif argdict['dataset'] in ['TREC6']:
            ds="SixClassData"
        else:
            raise ValueError("Dataset not found")
        dataAdd = pd.read_csv(f"{argdict['pathDataAdd']}/CBERT/aug_data/{ds}/train_augment.tsv", sep='\t').dropna()
        dataOG = pd.read_csv(f"{argdict['pathDataAdd']}/CBERT/aug_data/{ds}/train.tsv", sep='\t')
        nb_points=len(dataOG)
        for i in range(len(argdict['categories'])):
            nb_per_class=int(nb_points*argdict['split'] / len(argdict['categories']))
            dataSamples = dataAdd[dataAdd['label'] == i].sample(n=1 if nb_per_class<1 else nb_per_class)
            doc=""
            for sent in dataSamples['sentence']:
                doc+=sent+"\n"
            with open(f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{argdict['categories'][i]}.txt", "w") as f:
                f.write(doc)
    elif argdict["algo"]=="EDA":
        file = open(f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/data.txt", "r")
        lines=["" for i in range(len(argdict['categories']))]
        for i, line in enumerate(file.readlines()):
            # if i==argdict['num_to_add']:
            #     break
            # if i%2==1:
            #     continue
            label, sentence=line.split('\t')
            lines[int(label)]+=sentence
        for i in range(len(argdict['categories'])):
            with open(f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{argdict['categories'][i]}.txt", "w") as f:
                f.write(lines[i])
    elif argdict["algo"]=="CATGAN":
        #You just have to regroup the samples_x together and put them in the right thingy
        for i in range(len(argdict['categories'])):
            fileOut=""
            for j in range(2):
                file=open(f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/samples_{i}_{j}.txt", "r")
                for ll in file.readlines():
                    fileOut+=ll
            with open(f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{argdict['categories'][i]}.txt", "w") as f:
                f.write(fileOut)
    elif argdict["algo"]=="W2V":
        for cat in argdict['categories']:
            os.rename(f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{cat}.txt",
                      f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{cat}.txt")
    elif argdict["algo"]=="GPT":
        for cat in argdict['categories']:
            file=open(f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{cat}.txt", "r").readlines()[1:]
            file=' '.join(file)
            #split by end of sentence
            file=file.split('<|endoftext|>')
            file=[ff.replace('<\|endoftext\|>', ' ').replace('\n', ' ') for ff in file]
            file=[re.sub(f'{cat}:', '', ff) for ff in file]
            file='\n'.join(file)
            newfile = open(f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{cat}.txt","w")
            newfile.write(file)


def add_data(argdict):
    """Add the generated data to the dataframe for classification"""
    #TODO HERE
    # path=argdict['pathDataAdd']
    if argdict['vary_starting_set'] or argdict['second_dataset']:
        dfName=f'train_{argdict["starting_set_seed"]}'
    else:
        dfName='train'

    if argdict['max_length']==0:
        path=f"{argdict['pathDataAdd']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}"

    else:
        path = f"{argdict['pathDataAdd']}/SelectedDataMaxLength/{argdict['max_length']}/{argdict['dataset']}/{argdict['dataset_size']}"

    task=argdict['dataset']
    algo=argdict['algo']
    data = pd.read_csv(f"{path}/{dfName}.tsv", sep='\t')

    if argdict['dataset_size']==0:
        values=data['label'].value_counts()
        print(values)
        #We want to balance max min.
        min=values.min()
        #For each category we want to get twice the min
        total_data=min*2
        num_to_add=[total_data-values[i] for i in range(len(argdict['categories']))]
    else:
        num_to_add=[argdict['num_to_add'] for cat in argdict['categories']]

    dataOg=data
    counter=len(data)
    data=data.to_dict()
    dss=len(data['sentence'])
    # Add data to dico
    if argdict['split']!=0:
        if algo in ["CVAE_Classic"]:
            for index_cat, cat in enumerate(argdict['categories']):
                file=open(f"{argdict['pathDataAdd']}/GeneratedData/{algo}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{cat}.txt")
                print(f"Num of points to add for the category {cat}: {num_to_add[index_cat]}")
                for i, line in enumerate(file.readlines()):
                    if i==num_to_add[index_cat]:
                        break
                    #process line: remove eos,
                    data['sentence'][counter]=line[:-6]
                    data['label'][counter]=argdict['categories'].index(cat)
                    counter+=1
                print(f"Added {i} examples for the class")
        #Regular added Dat
        elif algo in ["GAN", "W2V", "CBERT", "CATGAN", "EDA", "AEDA", "VAE", "CVAE", "GPT", "GAN2", "GAN3", "VAE_EncDec", "VAE_LinkedEnc"]:
            for index_cat, cat in enumerate(argdict['categories']):
                print(f"Num of points to add for the category {cat}: {num_to_add[index_cat]}")
                file=open(f"{argdict['pathDataAdd']}/GeneratedData/{algo}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{cat}.txt")
                num=num_to_add[index_cat]
                for i, line in enumerate(file.readlines()):
                    if line.strip()=="":
                        num+=1
                        continue
                    if i==num:
                        break
                    #process line: remove eos,
                    data['sentence'][counter]=line.strip()
                    data['label'][counter]=argdict['categories'].index(cat)
                    counter+=1
                print(f"Added {i} examples for the {cat} category")
        else:
            raise Exception

    print(f"Length of the augmented dataset: {len(data['sentence'])}")
    print(f"Length of the selected data: {dss}")
    if argdict['dataset_size']==0:
        if argdict['split']!=0:
            desired_size=dss+sum(num_to_add)
        else:
            desired_size=dss
    else:
        desired_size=dss+dss*argdict['split']
    print(f"Desired length: {desired_size}")
    if len(data['sentence'])<desired_size:
        raise ValueError("Not enough data")
    data={'sentence':data['sentence'], 'label':data['label']}

    #Adding the data for the classifier
    if argdict['classifier']=='xgboost':
        df=pd.DataFrame.from_dict(data)

        folderName=argdict['dataset']
        df.to_csv(f"{argdict['pathDataOG']}/xgboost/{folderName}/train.tsv", sep='\t', index=False)
    elif argdict['classifier']=='lstm':
        pathout=f"{argdict['pathDataOG']}/lstm"
        df = pd.DataFrame.from_dict(data)
        df.to_csv(f"{pathout}/{task}/train.csv", index=False)
    elif argdict['classifier']=='dan':
        folderName = argdict['dataset']
        pathout=f"{argdict['pathDataOG']}/dan"
        df = pd.DataFrame.from_dict(data)
        df.to_csv(f"{pathout}/{task}/train.csv", index=False)
    elif argdict['classifier']=='jianLstm':
        pathout=f"{argdict['pathDataAdd']}/jiant-v1-legacy/data/"
        df = pd.DataFrame.from_dict(data)
        df.to_csv(f"{pathout}/{task}/train.tsv", sep='\t', index=False)
    elif argdict['classifier'] in ['bert', 'bert2']:
        #For bert, you need to also change the dev df
        _, dfVal=get_dataFrame(argdict)

        dataVal = {'sentence': dfVal['sentence'], 'label': dfVal['label']}

        if argdict['dataset']=='SST-2':
            dataset='sst'
            pathout = f"{argdict['pathDataAdd']}/data/{argdict['classifier']}/data/{dataset}"
            try:
                os.remove(f"{pathout}/train.jsonl")
            except:
                pass
            try:
                os.remove(f"{pathout}/val_max_length.jsonl")
            except:
                pass
            with open(f"{pathout}/train.jsonl", "a") as f:
                for i in data['label']:
                    dicoTemp = json.dumps({"idx": i, "label": str(data['label'][i]), "text": data['sentence'][i]})
                    f.write(dicoTemp)
                    f.write("\n")
            with open(f"{pathout}/val_max_length.jsonl", "a") as f:
                for i, row in dfVal.iterrows():
                    dicoTemp = json.dumps({"idx": i, "label": str(row['label']), "text": row['sentence']})
                    f.write(dicoTemp)
                    f.write("\n")
        elif argdict['dataset']=='QNLI':
            dataset = 'qnli'
            pathout = f"{argdict['pathDataAdd']}/data/{argdict['classifier']}/data/{dataset}"
            try:
                os.remove(f"{pathout}/train.jsonl")
            except:
                pass
            with open(f"{pathout}/train.jsonl", "a") as f:
                for i in data['label']:
                    sent=data['sentence'][i]
                    regexp=re.compile(r'[SEP]')
                    if not regexp.search(sent):
                        continue
                    premise, hypothesis=sent.split('[SEP]')
                    dicoTemp = json.dumps({"idx": i, "label": argdict['categories'][data['label'][i]], "premise": premise, "hypothesis": hypothesis})
                    f.write(dicoTemp)
                    f.write("\n")
        else:
            if argdict['dataset']=='TREC6':
                dataset='trec'
            else:
                dataset=argdict['dataset']
            pathout = f"{argdict['pathDataAdd']}/data/{argdict['classifier']}/data/{dataset}"
            try:
                os.remove(f"{pathout}/train.tsv")
            except:
                pass
            df = pd.DataFrame.from_dict(data)
            df.to_csv(f"{pathout}/train.tsv", sep='\t', index=False)
            dfVal.to_csv(f"{pathout}/dev_max_length.tsv", sep='\t', index=False)
        # #Creating config file
        # if argdict['dataset']=='SST-2':
        #     ext='.jsonl'
        # else:
        #     ext='.tsv'
        # dico='{"task": "'+dataset+'",paths": {"train": "'+path+dataset+'/train'+ext+'",' \
        #      '"test": "'+path+dataset+'/'+"test" if dataset=="sst" else "dev"+ext+'",' \
        #       '"val": "'+path+dataset+'/'+"val" if dataset=="sst" else "dev"+ext+'"},"name": "trec"}'
        # dico={"task":dataset, "paths": { "train": f"{argdict['pathDataAdd']}/data/bert/data/{dataset}/train{ext}",
        #                                  "test": f"{argdict['pathDataAdd']}/data/bert/data/{dataset}/{'test' if dataset=='sst' else 'dev'}{ext}",
        #                                  "val": f"{argdict['pathDataAdd']}/data/bert/data/{dataset}/{'val' if dataset=='sst' else 'dev'}{ext}"},
        #       "name": {dataset if dataset!="TREC6" else 'trec'}}
        # json.dump(dico, open(f"{argdict['pathDataAdd']}/data/bert/configs/{dataset}_config.json", "w"))

def cleanFolders(argdict):
    """Automatically remove empty folders for the experiment"""
    path = f"{argdict['pathDataAdd']}/Experiments/Record/{argdict['dataset']}/{argdict['algo']}/{argdict['classifier']}"
    for folders in os.listdir(path):
        try:
            param = json.load(open(f"{path}/{folders}/param.json", 'r'))
        except:
            os.rmdir(f"{path}/{folders}")

def cleanFoldersGen(argdict):
    path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}"
    try:
        for folders in os.listdir(path):
            try:
                param = json.load(open(f"{path}/{folders}/param.json", 'r'))
            except:
                shutil.rmtree(f"{path}/{folders}", ignore_errors=True)
    except:
        #The folder has not been created yet
        pass

def checkFolders(argdict, v=True):
    """Check in Experiments/Record if the experiment has not been done already. If not, create new experiments number"""
    found=False
    # i=0
    cleanFolders(argdict)
    if argdict['split']==0:
        for algo in ['CATGAN', 'CBERT', 'CVAE', "CVAE_Classic" 'EDA', 'GAN', 'GAN2', 'GAN3', 'GPT', 'VAE', 'W2V']:
            i=0
            while not found:
                path = f"{argdict['pathDataAdd']}/Experiments/Record/{argdict['dataset']}/{algo}/{argdict['classifier']}/{i}"
                try:
                    # Open the folder. If it works then check if its the same experiment, else go to next one
                    param = json.load(open(f"{path}/param.json", 'r'))
                    results = json.load(open(f"{path}/out.json", 'r'))
                    if compareFiles(argdict, param, "exp"):
                        if v:
                            print(f"Experiment was already run in folder {algo}/{i}")
                            print(f"Results are {results}")
                        found = True
                        return (results['acc_train'], results['acc_dev'], results['f1_train'], results['f1_dev']), i
                    i += 1
                except:
                    #Break out of th
                    break
    i=0
    #Not found in any of the folders = Find num folder and create it
    while not found:
        path=f"{argdict['pathDataAdd']}/Experiments/Record/{argdict['dataset']}/{argdict['algo']}/{argdict['classifier']}/{i}"
        try:
            #Open the folder. If it works then check if its the same experiment, else go to next one
            param=json.load(open(f"{path}/param.json", 'r'))
            results=json.load(open(f"{path}/out.json", 'r'))

            if compareFiles(argdict, param, "exp"):
                if v:
                    print(f"Experiment was already run in folder {i}")
                    print(f"Results are {results}")
                found=True
                # print((results['acc_train'], results['acc_dev'], results['f1_train'], results['f1_dev']), i)
                return (results['acc_train'], results['acc_dev'], results['f1_train'], results['f1_dev']), i
            i+=1
        except:
            #We could not open this folder, aka empty experiment. Create folder
            print("Creating new folder")
            os.mkdir(f"{path}")
            return i, i

def checkFoldersGenerated(argdict):
    """Check in GeneratedData if the experiment has not been done already. If not, create new Generation Folder"""
    found=False
    i=0
    cleanFoldersGen(argdict)
    while not found:

        path=f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{i}"
        # if i==33:
        #     asdgasd
        try:
            #Open the folder. If it works then check if its the same experiment, else go to next one
            param=json.load(open(f"{path}/param.json", 'r'))
            # if i==32:
            if compareFiles(argdict, param, "gen"):
                print(f"Dataset was already generated in folder {i}")
                found=True
                return i
            i+=1
        except:
            #We could not open this folder, aka empty experiment. Create folder
            print("Creating new folder for Generation")
            Path(path).mkdir(parents=True, exist_ok=True)
            return i

def num_exemples_generated(argdict):
    """For an experiment that was already generated, check how many data is available"""
    path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}"
    num_data=0
    for cat in argdict['categories']:
        file=open(f"{path}/{cat}.txt", "r").readlines()
        num_data+=len(file)
    return num_data

def compareFiles(argdict, file, folder):
    """Compare the argdict and the string of parameter EXCEPT for the computer parameter
    folder can be gen or exp, for generated or experiments"""

    if folder=="gen":
        # notInt=["computer", "retrain", "rerun", "run_ds_split", "test_latent_size_vs_split", "pathDataOG", "pathDataAdd", "categories", "cat", "nb_epoch_lstm",
        #         "dropout_classifier", "hidden_size_classifier", "numFolderGenerated", "classifier", "split"]
        #Instead what is important
        In=ImportantAlgo
    else:
        # notInt=["computer", "retrain", "rerun", "run_ds_split", "test_latent_size_vs_split", "pathDataOG", "pathDataAdd", "categories", "cat",
        #         "test_hidden_size_algo_vs_split", "numFolderGenerated"]
        #Everything that is important for generating the dataset makes a difference in classification
        In=ImportantClassifier
        if argdict['split']==0:
            #Remove the algo from the important characteristics since the split is 0
            In=ImportantSplitZero
    for key, value in argdict.items():
        if key in In:
            try:
                value2=file[key]
                if value2!=value:
                    return False
            except:
                return False
    return True

def log_results(argdict, results, i):
    path = f"{argdict['pathDataAdd']}/Experiments/Record/{argdict['dataset']}/{argdict['algo']}/{argdict['classifier']}/{i}"
    #Log results
    with open(f"{path}/param.json", "w") as f:
        json.dump(argdict, f)
    with open(f"{path}/out.json", "w") as f:
        json.dump(results, f)
    path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}"
    with open(f"{path}/param.json", "w") as f:
        json.dump(argdict, f)

def get_results_bert(argdict):
    #First delete all useless files
    if argdict['dataset']=='SST-2':
        task='sst'
    elif argdict['dataset']=='TREC6':
        task='trec'
    elif argdict['dataset']=="QNLI":
        task='qnli'
    else:
        task=argdict['dataset']
    path=f"{argdict['pathDataAdd']}/content/exp/runs/simple_{task}_bert-base-uncased"
    # os.remove(f"{path}/args.json")
    # os.remove(f"{path}//done_file")
    # os.remove(f"{path}//last_model.metadata.json")
    # os.remove(f"{path}//last_model.p")
    # os.remove(f"{path}//simple_run_config.json")
    # os.remove(f"{path}//val_metrics.json")
    results=json.load(open(f"{path}/val_metrics.json", "r"))
    return results[task]['loss'], results[task]['metrics']['minor']['acc'], 0, results[task]['metrics']['minor']['f1']

def get_results_bert2(argdict):
    #First delete all useless files
    if argdict['dataset']=='SST-2':
        task='sst'
    elif argdict['dataset']=='TREC6':
        task='trec'
    elif argdict['dataset']=="QNLI":
        task='qnli'
    else:
        task=argdict['dataset']
    path=f"{argdict['pathDataAdd']}/content/exp2/runs/simple_{task}_bert-base-uncased"
    # os.remove(f"{path}/args.json")
    # os.remove(f"{path}/args.json")
    # os.remove(f"{path}//done_file")
    # os.remove(f"{path}//last_model.metadata.json")
    # os.remove(f"{path}//last_model.p")
    # os.remove(f"{path}//simple_run_config.json")
    # os.remove(f"{path}//val_metrics.json")
    results=json.load(open(f"{path}/val_metrics.json", "r"))
    return results[task]['loss'], results[task]['metrics']['minor']['acc'], 0, results[task]['metrics']['minor']['f1']

def createFolders(argdict):
    """Create all folders necessary to the runs so you dont have to create them yourself"""
    """The specific folders that need to be created are in GeneratedData, SelectedData and ../data"""

    num=checkFoldersGenerated(argdict)
    argdict['numFolderGenerated']=num
    Path(f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{num}").mkdir(parents=True, exist_ok=True)
    Path(f"{argdict['pathDataAdd']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}").mkdir(parents=True, exist_ok=True)
    Path(f"{argdict['pathDataAdd']}/data/{argdict['classifier']}/{argdict['dataset']}").mkdir(parents=True, exist_ok=True)
    Path(f"{argdict['pathDataAdd']}/Experiments/Record/{argdict['dataset']}/{argdict['algo']}/{argdict['classifier']}").mkdir(parents=True, exist_ok=True)
    return argdict


def checkTrained(argdict):
    """Checks if it was already trained once at least, else change retrain to true
    We need to check if there is something in GeneratedData"""
    #Find folder number and check all categories
    # folder=checkFoldersGenerated(argdict)
    folder=argdict['numFolderGenerated']
    try:
        if argdict['algo'] not in ['EDA']:
            for cat in argdict['categories']:
                print(f'{argdict["pathDataAdd"]}/GeneratedData/{argdict["algo"]}/{argdict["dataset"]}/{argdict["dataset_size"]}/{folder}/{cat}.txt')
                file = open(f'{argdict["pathDataAdd"]}/GeneratedData/{argdict["algo"]}/{argdict["dataset"]}/{argdict["dataset_size"]}/{folder}/{cat}.txt')
        elif argdict['algo'] in ['EDA']:
            file = open(f'{argdict["pathDataAdd"]}/GeneratedData/{argdict["algo"]}/{argdict["dataset"]}/{argdict["dataset_size"]}/{folder}/data.txt')
    except:
        if argdict['split']!=0:
            argdict['retrain']=True
    return argdict

def cleanup(argdict):
    """Cleanup old generated and logs folder that do not have all the attributes we are looking for"""
    algos=["EDA", "W2V", "TRANS", "GAN", "GAN2", "GAN3", "CATGAN", "VAE", "VAE_EncDec", "CVAE", "CVAE_Classic", "CBERT", "GPT"]
    classifiers=["xgboost", "bert", "lstm", "dan"]
    datasets=["SST-2", "TREC6"]
    #Generation
    for algo in algos:
        for dataset in datasets:
            d=f"{argdict['pathDataAdd']}/GeneratedData/{algo}/{dataset}"
            try:
                listFolders=[os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]
                #For the various dataset Sizes
                for folder in listFolders:
                    #i is the reading pointer, j is the writing pointer
                    i=0
                    j=0
                    #Loop for all folders:
                    FolderExist=True
                    while FolderExist:
                        try:
                            param=json.load(open(f"{folder}/{i}/param.json", "r"))
                            keys=list(param.keys())
                            #check if all items are there
                            allItemsPresent=True
                            for items in ImportantAlgo:
                                if items in keys:
                                    pass
                                else:
                                    allItemsPresent=False
                                    #HERE, DELETE FOLDER
                                    shutil.rmtree(f"{folder}/{i}", ignore_errors=True)
                            if allItemsPresent:
                                #If this is a good folder and j<i, that means we should rewrite it
                                if j<i:
                                    shutil.move(f"{folder}/{i}/param.json", f"{folder}/{j}/param.json")
                                j+=1
                            i+=1
                        except:
                            FolderExist=False
            except:
                pass
    # Experiments
    for algo in algos:
        for dataset in datasets:
            for classifier in classifiers:
                folder = f"{argdict['pathDataAdd']}/Experiments/Record/{dataset}/{algo}/{classifier}"
                try:
                    # i is the reading pointer, j is the writing pointer
                    i = 0
                    j = 0
                    # Loop for all folders:
                    FolderExist = True
                    while FolderExist:
                        try:
                            param = json.load(open(f"{folder}/{i}/param.json", "r"))
                            keys = list(param.keys())
                            # check if all items are there
                            allItemsPresent = True
                            for items in ImportantClassifier:
                                if items in keys:
                                    pass
                                else:
                                    allItemsPresent = False
                                    # HERE, DELETE FOLDER
                                    shutil.rmtree(f"{folder}/{i}", ignore_errors=True)
                            if allItemsPresent:
                                # If this is a good folder and j<i, that means we should rewrite it
                                if j < i:
                                    shutil.move(f"{folder}/{i}/param.json", f"{folder}/{j}/param.json")
                                j += 1
                            i += 1
                        except:
                            FolderExist = False
                except:
                    pass



# parser = argparse.ArgumentParser(description='Processing data for further experiments')
# parser.add_argument('--dataset', type=str, default='SST2')
# parser.add_argument('--polarity', type=str, default='pos', help='class considered, pos/neg')
# parser.add_argument('--path', type=str, default='/media/frederic/DAGlue/jiant-v1-legacy/data', help='base path of the project (useful for switching between computers)')
# parser.add_argument('--dataset_size', type=int, default=0, help='number of example in the original dataset. If 0, use the entire dataset')
# args = parser.parse_args()
#
#
# argdict = args.__dict__
# main(argdict)
