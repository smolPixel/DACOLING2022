import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt

def run_external_process(process):
    output, error = process.communicate()
    if process.returncode != 0:
        raise SystemError
    return output, error

commands={'EDA': 'python3 run_algo.py --dataset_size 500 --algo EDA --bs_algo 25',
          'VAE': 'python3 run_algo.py --algo VAE --dataset_size 500 --x0 500 --latent_size 5 --hidden_size_algo 1024 --nb_epoch_algo 20 --dropout_algo 0.5 --word_dropout 0.6 --unidir_algo',
          'CATGAN':'python3 run_algo.py --dataset_size 500 --algo CATGAN --nb_epoch_algo 5 --bs_algo 25',
          'CBERT':'python3 run_algo.py --algo CBERT --dataset_size 500 --bs_algo 32'}



def extractResult(results, classifier):
    if classifier in ['xgboost', 'bert']:
        return results[1]*100
    elif classifier=='dan':
        return max(results[1])*100
#
#
baseline={'xgboost':0, 'dan':0, 'bert':0}
resultsFinal={'EDA':{'xgboost':[], 'dan':[], 'bert':[], 'color':'green'},
              'VAE':{'xgboost':[], 'dan':[], 'bert':[], 'color':'blue'},
              'CATGAN':{'xgboost':[], 'dan':[], 'bert':[], 'color':'red'},
              'CBERT':{'xgboost':[], 'dan':[], 'bert':[], 'color':'black'}}

for algo, command in commands.items():
    for split in [1,2,3,4,5, 6, 7, 8]:
        for classifier in ['xgboost', 'dan', 'bert']:#, 'dan', 'bert']:
            # baseline = np.zeros((30, 5))
            # augmented = np.zeros((30, 5))
            print(algo, split, classifier, command)
            commandToRun = command + f" --split {split} --classifier {classifier} --get_all_results"
            process = subprocess.Popen(commandToRun.split(), stdout=open('tefdasfdasfdsadsfa', 'w'))
            output, error = run_external_process(process)
            results = json.load(open("temp.json", "r"))
            aug = results['augmented']
            bas = results['baseline']
            aug_average=[]
            bas_average=[]
            # print(aug)
            # print(extractResult(aug[str(i)], classifier))
            for i in range(30):
                # print(aug[str(i)])
                aug_average.append(extractResult(aug[str(i)], classifier))
                bas_average.append(extractResult(bas[str(i)], classifier))
            resultsFinal[algo][classifier].append(np.mean(aug_average))
            baseline[classifier]=np.mean(bas_average)
            print(baseline)

xgboost = plt.figure()

#Graphes
for algo, results in resultsFinal.items():
    print(algo, results)
    plt.plot([baseline['xgboost']]+results['xgboost'], c=results['color'], label=algo)

plt.legend()
plt.xlabel("Percent of augmentation")
plt.ylabel("Accuracy")

plt.savefig("Graphes/Num_Aug_Vs_Perfo_xgboost.png")

xgboost = plt.figure()

#Graphes
for algo, results in resultsFinal.items():
    print(algo, results)
    plt.plot([baseline['dan']]+results['dan'], c=results['color'], label=algo)

plt.legend()
plt.xlabel("Percent of augmentation")
plt.ylabel("Accuracy")

plt.savefig("Graphes/Num_Aug_Vs_Perfo_dan.png")

xgboost = plt.figure()

#Graphes
for algo, results in resultsFinal.items():
    print(algo, results)
    plt.plot([baseline['bert']]+results['bert'], c=results['color'], label=algo)

plt.legend()
plt.xlabel("Percent of augmentation")
plt.ylabel("Accuracy")

plt.savefig("Graphes/Num_Aug_Vs_Perfo_bert.png")



print(baseline)
print(resultsFinal)