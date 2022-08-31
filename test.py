import numpy as np
import itertools
print(3.465724215775732**5)

#We have 4 per dimensions
print(10**(0.5))

#Let say 1 dimension, then
#For now, imagine the grid spans all the way to -1, 1
large=2/3
dim=[-1+large*i for i in range(5)]
print(dim)
all_possibilities=[]
for comb in itertools.product([0,1,2,3], repeat=5):
	all_possibilities.append(comb)

print(all_possibilities)
print(len(all_possibilities))