from Paired_Ttest import commands, run_external_process
import subprocess
import numpy as np


command="python3 run_algo.py --algo VAE_LinkedEnc --split 1 --x0 15 --latent_size 15 --hidden_size_algo 2048 --nb_epoch_algo 30 --dropout_algo 0.5 --word_dropout 0.6 --k 0.00025"
# command="python3 run_algo.py --algo CBERT --split 1"
identical=[]
length=[]
length_overall=[]
datasets=['SST-2', 'Irony', 'FakeNews', 'Subj']
dataset_sizes=[500,1000,1500,2000]
for dataset in datasets:
	for i, ds in enumerate(dataset_sizes):
		command = command + f" --classifier dummy --doublons --computer labo --dataset_size {ds} --dataset {dataset}"
		# print(command)
		process = subprocess.Popen(command.split(), stdout=open("trsesdgffgs", "w"))
		output, error = run_external_process(process)
		with open('tefdasfdasfdsadsfa', 'r') as f:
			content=f.read()
			print(content)
			identical.append(float(content))
		with open('tefdasfdasfdsadsfalength', 'r') as f:
			content=f.read()
			print(content)
			if content !="nan":
				length.append(float(content))
		with open('tefdasfdasfdsadsfalength_overall', 'r') as f:
			content=f.read()
			print(content)
			if content !="nan":
				length_overall.append(float(content))
print(identical)
print(f"percent of identical sentences {np.mean(identical)} with average length of {np.mean(length)}, while the true average is {np.mean(length_overall)}")
