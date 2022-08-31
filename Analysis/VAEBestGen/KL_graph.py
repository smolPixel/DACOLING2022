import math
import numpy as np
import matplotlib.pyplot as plt
bs=32
k=0.0025
x0=15
epochs=30

ds=500






def kl_anneal_function(anneal_function, step, k, x0):
	if anneal_function == 'logistic':
		return float(1 / (1 + np.exp(-k * (step - x0))))
	elif anneal_function == 'linear':
		return min(1, step / x0)
graphe = plt.figure()
	# for i, algo in enumerate(algos):
	# 	plt.plot(dataset_sizes, dicos[i][dataset], label=algo)


for ds in [500, 1000, 1500, 2000]:
	step=0
	batches_per_epochs=math.ceil(ds/bs)
	print(batches_per_epochs)
	x=np.arange(30)
	y=[]
	for i in range(epochs):
		for j in range(batches_per_epochs):
			KL_Weight=kl_anneal_function("logistic", step, k, x0*batches_per_epochs)
			step+=1
		y.append(KL_Weight)
		print(f"Epoch {i}, KL_Weight {KL_Weight}")
	plt.plot(x, y, label=ds)

plt.legend(title="Dataset Size")
plt.xlabel("Epoch")
plt.ylabel("KL Weight")
# plt.xticks(ticks=np.arange(30))#, labels=dataset_sizes)
plt.savefig(f"KL_Weight.png")
