import matplotlib.pyplot as plt
import numpy as np

split=[1,2,3,4,5]#,6,7,8,9]
#VAE do .5 word dropout lol it better (well redo all with .5
results_VAE=[88.6, 88.7, 88.6, 88.8, 88.9]
results_VAE_LinkedEnc=[88.4, 88.6, 88.7, 88.7, 88.6]
results_VAE_EncDec=[87.2, 86.9, 86.8, 86.8, 86.1]
results_CVAE=[88.2, 87.8, 87.5,  87.2, 87.2]
results_CBERT=[88.0, 88.3, 88.6, 88.6, 88.4]

Latent_space=[10,15,20,25,30]
results_SST_Latent=[88.1, 88.6, 88.5, 88.3, 88.4]
results_SST_Latent_CVAE=[87.8, 88.2, 87.9, 87.9, 87.8]

Hidden_size=[512, 1024, 2048, 4096]
results_SST_hidden=[88.6, 88.3, 88.6, 88.5]
results_SST_hidden_CVAE=[88.0, 87.7, 87.8, 87.7]

word_dropouts=[0, 0.3, 0.6, 0.9]
results_SST_word_dropout=[88.2, 88.6, 88.6, 88.2]
results_SST_word_dropout_CVAE=[88.5, 88.4, 87.8, 87.8]

dropout_algo=[0, 0.3, 0.6, 0.9]
results_SST_dropout=[88.6, 88.3, 88.4, 88.5]
results_SST_dropout_CVAE=[88.5, 88.5, 88.1, 87.7]

num_epochs=[10,20,30,40,50]
results_SST_num_epochs=[88.4, 88.7, 88.6, 88.3, 88.3]
results_SST_num_epochs_CVAE=[87.9, 88.0, 88.1, 88.3, 88.4]

k=[0.00025, 0.0025, 0.01, 0.1]
results_SST_k=[88.7, 88.7, 88.7, 88.7]

#RERUN ALL WITH 20

#Arrays are 500,1000, 1500, 2000
resultsBaseline={"SST-2":[83.3, 87.7, 88.2, 90.0],
				 "FakeNews":[89.9, 91.1, 92.5, 93.0],
				 "Irony":[62.5, 65.0, 67.8, 69.5],
				 "Subj":[89.2, 91.6, 93.2, 94.7]}

resultsBaseline_std={"SST-2":[1.3, 0.7, 0.6, 0.4],
				 "FakeNews":[0.3, 0.3, 0.2, 0.2],
				 "Irony":[1.2, 0.9, 0.6, 0.7],
				 "Subj":[0.8, 0.8, 0.5, 0.3]}

resultsCBERT={"SST-2":[85.7, 88.0, 89.3, 90.0],
				 "FakeNews":[90.0, 91.6, 92.6, 93.0],
				 "Irony":[64.2, 66.6, 68.3, 70.3],
				 "Subj":[89.1, 91.9, 93.0, 94.2]}

resultsCBERT_std={"SST-2":[0.9, 0.6, 0.4, 0.4],
				 "FakeNews":[0.3, 0.2, 0.3, 0.3],
				 "Irony":[0.8, 1.2, 0.9, 0.9],
				 "Subj":[0.5, 0.5, 0.5, 0.5]}

resultsVAE={"SST-2":[86.6, 88.8, 89.4, 90.0],
				 "FakeNews":[90.9, 92.2, 92.7,  93.2],
				 "Irony":[65.3, 66.8, 68.1,  69.9],
				 "Subj":[89.5, 92.1, 93.6, 94.7]}

#SST500 : RNN Deco untied: RNN Deco + output untied:  86.3 + embedd entré untied: 86.8

resultsVAE_std={"SST-2":[0.8, 0.6, 0.4, 0.5],
				 "FakeNews":[0.2, 0.3, 0.2, 0.3],
				 "Irony":[0.7, 1.0, 0.8, 0.7],
				 "Subj":[0.6, 0.5, 0.4, 0.5]}

#FakeNews 500: Normal: 90.8, latent size 20: 90.9
#Irony 500. Reducing the latent space to 5 : 64.5. Augmenting to 15: 65.1, augmenting to 20: 65.3, augmenting to 25: 64.6, augmenting to 30: 64.8
# 1500: normal 67.7, latent size 20 : 68.1, 25:67.9
# 2000: normal: 69.7, latent size 20: 69.2, latent size 25: 69.6
	#hidden_size norma: 69.7, 4096: 69.4
#Subj 500: Normal 89.3, latent size 20: 89.5

#86.3%?

resultsVAE_LinkedEnc={"SST-2":[86.9, 88.3, 89.3, 90.0],
				 "FakeNews":[90.9, 92.0, 92.7, 93.1],
				 "Irony":[65.0, 67.0, 68.1, 70.4],
				 "Subj":[89.2, 92.0, 93.4, 94.7]}

resultsVAE_LinkedEnc_std={"SST-2":[0.7, 0.5, 0.6, 0.7],
				 "FakeNews":[0.3, 0.2, 0.3, 0.3],
				 "Irony":[1.0, 0.6, 0.9, 1.3],
				 "Subj":[0.5, 0.6, 0.3, 0.6]}

#VAEPar deviation from reg
#SSt-2: 500, latent size 10, 1K:
#Fakew News: 500: ls 10
#Subj: 500: ls
resultsVAEPar={"SST-2":[85.2, 87.2, 88.2, 89.1],
				 "FakeNews":[90.9, 92.0, 92.8,  93.1],
				 "Irony":[64.4, 66.9, 68.4, 69.9],
				 "Subj":[88.4, 90.4, 92.1, 93.3]}

resultsVAEPar_std={"SST-2":[1.1, 1.0, 0.6, 0.5],
				 "FakeNews":[0.2, 0.3, 0.3, 0.1],
				 "Irony":[0.9, 0.8, 0.8, 1.0],
				 "Subj":[0.9, 0.6, 0.6, 0.6]}
#CVAE Deviation


#Irony: 1500: ls 10, 500: ls 10
resultsCVAE={"SST-2":[86.4, 88.2, 88.7, 89.3],
				 "FakeNews":[91.0, 92.1, 92.7,  93.3],
				 "Irony":[65.1, 67.3, 68.1, 69.4],
				 "Subj":[89.3, 91.4, 93.0, 94.2]}

resultsCVAE_std={"SST-2":[0.8, 0.5, 0.5, 0.6],
				 "FakeNews":[0.3, 0.3, 0.3, 0.3],
				 "Irony":[1.2, 1.1, 1.1, 0.8 ],
				 "Subj":[0.7, 0.7, 0.5, 0.7]}
dataset_sizes=[500, 1000, 1500, 2000]
datasets=["SST-2", "FakeNews", "Irony", "Subj"]
algos=["Baseline", "CBERT", "VAE", "VAE-Linked", "VAE-Paraphrase", "CVAE"]
dicos=[resultsBaseline, resultsCBERT, resultsVAE, resultsVAE_LinkedEnc, resultsVAEPar, resultsCVAE]
dicosstds=[resultsBaseline_std, resultsCBERT_std, resultsVAE_std,  resultsVAE_LinkedEnc_std, resultsVAEPar_std, resultsCVAE_std]
#Graphes
for dataset in datasets:
	graphe = plt.figure()
	for i, algo in enumerate(algos):
		print(algo)
		err=np.array(dicosstds[i][dataset])
		err=err
		plt.errorbar(dataset_sizes, dicos[i][dataset], yerr=err, capsize=3, label=algo)
	plt.legend()
	plt.xlabel("Initial size of the dataset")
	plt.ylabel("Accuracy")
	plt.xticks(ticks=dataset_sizes, labels=dataset_sizes)
	plt.savefig(f"{dataset}.png")


def get_average_result(dic, ds):
	loc=dataset_sizes.index(ds)
	arr=[]
	for dataset in datasets:
		arr.append(dic[dataset][loc])
	return np.mean(arr)

print("Table of average results")
print("------------")

print("\\begin{tabular}{c|c| c| c | c | c |}")
print(f" & {' & '.join([str(ds) for ds in dataset_sizes])} \\\\")
for i, algo in enumerate(algos):
	results=[]
	stds=[]
	for ds in dataset_sizes:
		results.append(get_average_result(dicos[i], ds))
		stds.append(get_average_result(dicosstds[i], ds))
	#Now ou results contains all results for a line
	print(f"{algo} & {' & '.join([f'{round(results[i], 1)} ({round(stds[i], 1)})' for i in range(len(dataset_sizes))])} & {round(np.mean(results), 1)}\\\\")
	# print(f"{algo} & {round(results[0],1)} & {round(results[1],1)} & {round(results[2],1)}\\\\")

"""Graphes of hyperparameters"""
graphe = plt.figure()
plt.plot(Latent_space, results_SST_Latent)
plt.legend()
plt.xlabel("Number of dimensions of the latent space")
plt.ylabel("Accuracy")
plt.xticks(ticks=Latent_space, labels=Latent_space)
plt.savefig(f"SST2LatentSize.png")

graphe = plt.figure()
plt.plot(Hidden_size, results_SST_hidden)
plt.legend()
plt.xlabel("Hidden size of both the encoder and decoder")
plt.ylabel("Accuracy")
plt.xticks(ticks=Hidden_size, labels=Hidden_size)
plt.savefig(f"SST2Hidden.png")

graphe = plt.figure()
plt.plot(word_dropouts, results_SST_word_dropout)
plt.legend()
plt.xlabel("Percent of word dropped in the decoder")
plt.ylabel("Accuracy")
plt.xticks(ticks=word_dropouts, labels=word_dropouts)
plt.savefig(f"SST2WordDropout.png")

graphe = plt.figure()
plt.plot(dropout_algo, results_SST_dropout)
plt.legend()
plt.xlabel("Dropout parameter")
plt.ylabel("Accuracy")
plt.xticks(ticks=dropout_algo, labels=dropout_algo)
plt.savefig(f"SST2Dropout.png")





"""Graphes of acc vs spilts"""
graphe = plt.figure()
for results, name in zip([results_VAE, results_CVAE, results_VAE_EncDec, results_VAE_LinkedEnc, results_CBERT], ["VAE", "CVAE", "VAE-Par", "VAE-Linked", "CBERT"]):
	x=[1000]
	x.extend([ss*1000+1000 for ss in split])
	result=[87.5]
	result.extend(results)
	plt.plot(x, result, label=name)
plt.xlabel("Final dataset size")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(f"SST21KvsSplit.png")

# for algo, results in resultsFinal.items():
#     print(algo, results)
#     plt.plot([baseline['xgboost']]+results['xgboost'], c=results['color'], label=algo)