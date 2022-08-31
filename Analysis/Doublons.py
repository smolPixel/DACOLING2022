"""Find the number of identical sentences in the generated sentences vs selected"""

import pandas as pd
import numpy as np
def doublons(argdict):
    SelectedData = pd.read_csv(f"{argdict['pathDataAdd']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/train.tsv", sep='\t')
    # Train on basline, get prediction and true result
    dfVal = pd.read_csv(f'{argdict["pathDataOG"]}/{argdict["dataset"]}/dev.tsv', sep='\t')

    listeSent=list(SelectedData['sentence'])
    # print(listeSent)
    index = len(SelectedData)
    num_ex=0
    num_sim=0
    length=[]
    length_overall=[]
    for i, cat in enumerate(argdict['categories']):
        path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}/{cat}.txt"
        # num_to_add=round(n / len(argdict['categories']))
        ds=argdict['dataset_size'] if int(argdict['dataset_size'])!=0 else 15000
        n=int(ds/len(argdict['categories']))
        file = open(path, 'r').readlines()[:n]
        print("birtchcasdsa")
        # if 'but not without cheesy fun factor .' in listeSent:
        #     print("Bitch")
            # fdasdfa
        for line in file:
            line=line.strip()
            # print(line)
            if line in listeSent:
                num_sim+=1
                # print(line)
                length.append(len(line.split(' ')))
                # fds
            length_overall.append(len(line.split(' ')))
            num_ex+=1

    # print(num_ex)
    # print(num_sim)
    # print(np.mean(length))
    # fds
    # print(listeSent)
    print(f"Percent of identical sentences: {float(num_sim)*100/num_ex}")
    return float(num_sim)*100/num_ex, np.mean(length), np.mean(length_overall)

