import numpy as np
file=open("../TODOIPM", "r").readlines()

lengths=[]

text=["for decades we ' ve marveled at disney ' s rendering of water , snow , flames and shadows in a hand-drawn animated world .",
"much ' s the scariest guy you ' ll see all summer .",
"amazingly dopey .",
"a company .",
"although sensitive to a fault , often overwritten , with a surfeit of weighty revelations , flowery dialogue , and nostalgia for the past and roads not taken .",
"this would have been better than the incomprehensible anne , it ' s based upon the movie ."]


for line in text:
	lengths.append(len(line.split(" ")))

print(np.mean(lengths))