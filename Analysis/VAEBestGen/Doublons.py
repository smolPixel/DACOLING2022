from Paired_Ttest import commands, run_external_process
import subprocess
import numpy as np


command="python3 ../../run_algo.py --algo VAE_LinkedEnc --split 1 --dataset_size 1000 --x0 15 --latent_size 15 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025"
# for dataset, command in enumerate(splitCommands[split]):
command = command + f" --classifier dummy --doublons --computer labo"
# print(command)
process = subprocess.Popen(command.split(), stdout=open("trsesdgffgs", "w"))
output, error = run_external_process(process)
with open('tefdasfdasfdsadsfa', 'r') as f:
    content=f.read()
    identical.append(float(content))
print(f"Algo {algo} split {split} percent of identical sentences {np.mean(identical)}")
