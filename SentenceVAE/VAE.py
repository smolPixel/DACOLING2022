"""Wrapper for the SSVAE"""
import os
import json
import time
import torch
import argparse
import shutil
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from sklearn.metrics import accuracy_score
import math

from SentenceVAE.ptb import PTB
from SentenceVAE.utils import to_var, idx2word, expierment_name
from SentenceVAE.model import SentenceVAE


class VAE():

	def __init__(self, argdict):
		self.argdict=argdict
		self.splits=['train', 'valid']
		# self.datasets = datasets
		# self.datasetsLabelled = datasetLabelled
		self.model, self.params, self.datasets=self.init_model_dataset()
		# optimizers
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  # self.argdict.learning_rate)
		self.loss_function_discriminator = torch.nn.CrossEntropyLoss()
		self.step=0
		self.epoch = 0

	def init_model_dataset(self):
		# splits = ['train', 'valid']  # + (['test'] if self.argdict.test else [])

		datasets = OrderedDict()
		for split in self.splits:
			datasets[split] = PTB(
				data_dir=f'{self.argdict["pathDataAdd"]}/SentenceVAE/data',
				split=split,
				create_data=False,  # self.argdict.create_data,
				max_sequence_length=60,  # self.argdict.max_sequence_length,
				min_occ=0  # self.argdict.min_occ
			)

		# print("BLIBLBILBi")
		# print(datasetsLabelled['train'])

		params = dict(
			vocab_size=datasets['train'].vocab_size,
			sos_idx=datasets['train'].sos_idx,
			eos_idx=datasets['train'].eos_idx,
			pad_idx=datasets['train'].pad_idx,
			unk_idx=datasets['train'].unk_idx,
			max_sequence_length=60,  # self.argdict.max_sequence_length,
			embedding_size=300,  # self.argdict.embedding_size,
			rnn_type='gru',  # self.argdict.rnn_type,
			hidden_size=self.argdict['hidden_size_algo'],
			word_dropout=self.argdict['word_dropout'],  # self.argdict.word_dropout,
			embedding_dropout=self.argdict['dropout_algo'],  # self.argdict.embedding_dropout,
			latent_size=self.argdict['latent_size'],
			num_layers= self.argdict['nb_layers_algo'],
			bidirectional= False
		)
		model = SentenceVAE(**params)
		if torch.cuda.is_available():
			model = model.cuda()

		self.step=0
		self.epoch=0

		return model, params, datasets

	def kl_anneal_function(self, anneal_function, step, k, x0):
		if anneal_function == 'logistic':
			return float(1 / (1 + np.exp(-k * (step - x0))))
		elif anneal_function == 'linear':
			return min(1, step / x0)

	def loss_fn(self, logp, target, mean, logv, anneal_function, step, k):
		NLL = torch.nn.NLLLoss(ignore_index=self.datasets['train'].pad_idx, reduction='sum')
		# cut-off unnecessary padding from target, and flatten
		target = target.contiguous().view(-1)
		logp = logp.view(-1, logp.size(2))

		# Negative Log Likelihood
		NLL_loss = NLL(logp, target)

		# KL Divergence
		KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
		# print(self.argdict['x0'], k, self.dataset_length)
		KL_weight = self.kl_anneal_function(anneal_function, step, k, self.dataset_length*self.argdict['x0'])

		return NLL_loss, KL_loss, KL_weight
	def run_epoch(self):
		for split in self.splits:

			data_loader = DataLoader(
				dataset=self.datasets[split],
				batch_size=32,  # self.argdict.batch_size,
				shuffle=split == 'train',
				num_workers=cpu_count(),
				pin_memory=torch.cuda.is_available()
			)

			# tracker = defaultdict(tensor)

			# Enable/Disable Dropout
			if split == 'train':
				self.model.train()
				self.dataset_length=len(data_loader)
			else:
				self.model.eval()

			for iteration, batch in enumerate(data_loader):

				batch_size = batch['input'].size(0)

				for k, v in batch.items():
					if torch.is_tensor(v):
						batch[k] = to_var(v)

				# Forward pass
				logp, mean, logv, z = self.model(batch['input'])

				# loss calculation
				# NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'],
				#                                        batch['length'], mean, logv, self.argdict.anneal_function, step,
				#                                        self.argdict.k, self.argdict.x0)
				NLL_loss, KL_loss, KL_weight = self.loss_fn(logp, batch['target'], mean, logv, 'logistic', self.step,
															0.0025)

				loss = (NLL_loss + KL_weight * KL_loss) / batch_size

				# backward + optimization
				if split == 'train':
					self.optimizer.zero_grad()
					loss.backward()
					self.optimizer.step()
					self.step += 1

				if iteration % 50 == 0 or iteration + 1 == len(data_loader):
					print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
						  % (
							  split.upper(), iteration, len(data_loader) - 1, loss.item(), NLL_loss.item() / batch_size,
							  KL_loss.item() / batch_size, KL_weight))

				# if split == 'valid':
				#     if 'target_sents' not in tracker:
				#         tracker['target_sents'] = list()
				#     tracker['target_sents'] += idx2word(batch['target'].data, i2w=self.datasets['train'].get_i2w(),
				#                                         pad_idx=self.datasets['train'].pad_idx)
				#     tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)

			# print("%s Epoch %02d/%i, Mean ELBO %9.4f" % (split.upper(), epoch, self.argdict['nb_epoch_algo'], tracker['ELBO'].mean()))
			print("%s Epoch %02d/%i, Mean ELBO %9.4f" % (split.upper(), self.epoch, self.argdict['nb_epoch_algo'], 0))

	def train(self):
		ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())


		print(self.model)

		# if self.argdict.tensorboard_logging:
		#     writer = SummaryWriter(os.path.join(self.argdict.logdir, expierment_name(self.argdict, ts)))
		#     writer.add_text("model", str(model))
		#     writer.add_text("self.argdict", str(self.argdict))
		#     writer.add_text("ts", ts)

		save_model_path = os.path.join(self.argdict['pathDataAdd'], 'bin')
		# shutil.
		os.makedirs(save_model_path, exist_ok=True)

		# with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
		#     json.dump(self.params, f, indent=4)



		# tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
		# step = 0
		# for epoch in range(self.argdict.epochs):
		for epoch in range(self.argdict['nb_epoch_algo']):
			self.epoch=epoch
			self.run_epoch()


				# if self.argdict.tensorboard_logging:
				#     writer.add_scalar("%s-Epoch/ELBO" % split.upper(), torch.mean(tracker['ELBO']), epoch)

				# save a dump of all sentences and the encoded latent space
				# if split == 'valid':
				#     dump = {'target_sents': tracker['target_sents'], 'z': tracker['z'].tolist()}
				#     if not os.path.exists(os.path.join('dumps', ts)):
				#         os.makedirs('dumps/' + ts)
				#     with open(os.path.join('dumps/' + ts + '/valid_E%i.json' % epoch), 'w') as dump_file:
				#         json.dump(dump, dump_file)

				# save checkpoint
				# if split == 'train':
				#     checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % epoch)
				#     torch.save(self.model.state_dict(), checkpoint_path)
				#     print("Model saved at %s" % checkpoint_path)

	def encode(self):
		dico={}
		for split in self.splits:
			data_loader = DataLoader(
				dataset=self.datasets[split],
				batch_size=64,#self.argdict.batch_size,
				shuffle=False,
				num_workers=cpu_count(),
				pin_memory=torch.cuda.is_available()
			)
			# Enable/Disable Dropout

			self.model.eval()
			# print(f"The dataset length is {len(data_loader.dataset)}")
			dataset = torch.zeros(len(data_loader.dataset), self.params['latent_size'])
			labels = torch.zeros(len(data_loader.dataset))
			counter = 0
			for iteration, batch in enumerate(data_loader):
				# print("Oh la la banana")
				batch_size = batch['input'].size(0)
				# print(batch['input'].shape)
				for k, v in batch.items():
					if torch.is_tensor(v):
						batch[k] = to_var(v)
				#
				# print(batch['input'])
				# print(batch['input'].shape)
				z = self.model.encode(batch['input'])
				# print(batch_size)
				# print(z.shape)
				dataset[counter:counter + batch_size] = z
				labels[counter:counter + batch_size] = batch['label']
				counter += batch_size
			# print(dataset)
			dico[f"encoded_{split}"]=dataset
			# torch.save(labels, f"bin/labels_{split}.pt")
			# torch.save(dataset, f"bin/encoded_{split}.pt")
		return dico["encoded_train"]

	def generate(self, datapoints):
		#Generates from fixed datapoints
		self.model.eval()

		samples, z = self.model.inference(z=datapoints)
		# print(samples)
		# print('----------SAMPLES----------')
		return idx2word(samples, i2w=self.datasets['train'].get_i2w(), pad_idx=self.datasets['train'].get_w2i()['<pad>'], eos_idx=self.datasets['train'].get_w2i()['<eos>'])

	def augment(self):
		# fds
		data_loader = DataLoader(
			dataset=self.datasets['train'],
			batch_size=64,#self.argdict.batch_size,
			shuffle=True,
			num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)

		sent_pres=self.datasets['train'].list_sentences()
		# Enable/Disable Dropout

		self.model.eval()
		# print(f"The dataset length is {len(data_loader.dataset)}")
		dataset = torch.zeros(len(data_loader.dataset), self.params['latent_size'])
		labels = torch.zeros(len(data_loader.dataset))
		counter = 0
		# files=[""]*len(self.argdict['categories'])
		file=""

		num_data=self.argdict['dataset_size']
		num_to_generate=math.ceil(num_data*self.argdict['split'])
		bs=32
		while num_to_generate>0:
			# print("Oh la la banana")
			batch_size = bs
			samples, z = self.model.inference(n=batch_size)
			generated_this_batch=0
			for sentence in idx2word(samples, i2w=self.datasets['train'].get_i2w(), pad_idx=self.datasets['train'].get_w2i()['<pad>'], eos_idx=self.datasets['train'].get_w2i()['<eos>']):
				# print(sentence)
				# print(sentence.strip())
				# print('---')
				# sentence=sentence.replace(' .', '.').replace(" ' ", "'").strip()
				# if sentence not in sent_pres:
				file+=sentence+"\n"
				# 	generated_this_batch+=1
				generated_this_batch+=1
			# fds

			num_to_generate-=generated_this_batch

			# print(batch_size)
			# print(z.shape)
		with open(f"{self.argdict['pathDataAdd']}/GeneratedData/VAE/{self.argdict['dataset']}/{self.argdict['dataset_size']}/{self.argdict['numFolderGenerated']}/{self.argdict['cat']}.txt", "w") as f:
			f.write(file)
		# for i in range(math.ceil(self.argdict['split'])):
		# 	for iteration, batch in enumerate(data_loader):
		# 		# print("Oh la la banana")
		# 		batch_size = batch['input'].size(0)
		# 		# print(batch['input'].shape)
		#
		# 		for k, v in batch.items():
		# 			if torch.is_tensor(v):
		# 				batch[k] = to_var(v)
		# 		#
		# 		# print(batch['input'])
		# 		# print(batch['input'].shape)
		# 		# z = self.model.encode(batch['input'])
		# 		samples, z = self.model.inference(n=batch_size)
		# 		for sentence in idx2word(samples, i2w=self.datasets['train'].get_i2w(), pad_idx=self.datasets['train'].get_w2i()['<pad>'], eos_idx=self.datasets['train'].get_w2i()['<eos>']):
		# 			file+=sentence+"\n"
		#
		# 		# print(batch_size)
		# 		# print(z.shape)
		# 		with open(f"{self.argdict['pathDataAdd']}/GeneratedData/VAE/{self.argdict['dataset']}/{self.argdict['dataset_size']}/{self.argdict['numFolderGenerated']}/{self.argdict['cat']}.txt", "w") as f:
		# 			f.write(file)


		# print(dataset)