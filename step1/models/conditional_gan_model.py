import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks
from .losses import init_loss
import torch.nn.functional as F
from .introvae import *

import pdb


try:
	xrange          # Python2
except NameError:
	xrange = range  # Python 3

class ConditionalGAN(BaseModel):
	def name(self):
		return 'ConditionalGANModel'

	def initialize(self, opt):
		self.opt = opt
		BaseModel.initialize(self, opt)
		self.isTrain = opt.isTrain
		# define tensors
		self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
								   opt.fineSize, opt.fineSize)
		self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
								   opt.fineSize, opt.fineSize)

		# load/define networks
		# Temp Fix for nn.parallel as nn.parallel crashes oc calculating gradient penalty
		use_parallel = 0

		if opt.which_model_netG == 'introAE':
			str_to_list = lambda x: [int(xi) for xi in x.split(',')]
			self.netG = IntroAE(norm=opt.norm, gpuId = opt.gpu_ids,cdim=3, hdim=opt.hdim, channels=str_to_list(opt.channels), image_size=opt.output_height)
			self.old_lr = opt.lr
			if len(opt.gpu_ids) > 0:
				self.netG.cuda(opt.gpu_ids[0])
			if not self.isTrain or opt.continue_train:
				self.load_network(self.netG.encoder, 'G_Encoder1', opt.which_epoch)
				self.load_network(self.netG.decoder, 'G_Decoder', opt.which_epoch)
			self.optimizer_G_E1 = optim.Adam(self.netG.encoder.parameters(), lr=opt.lr)
			self.optimizer_G_D = optim.Adam(self.netG.decoder.parameters(), lr=opt.lr)
			self.contentLoss = init_loss(opt, self.Tensor)
			print('---------- Networks initialized -------------')
			networks.print_network(self.netG.encoder)
			networks.print_network(self.netG.decoder)
			print('-----------------------------------------------')	
		
		else:
			raise ValueError('This repo only support the autoencoder modified from introAE, i.e., opt.which_model_netG == introAE. \
			But you can use this option to add new model')

	def set_input(self, input):
		AtoB = self.opt.which_direction == 'AtoB'
		input_A = input['A' if AtoB else 'B']	# projection image
		input_B = input['B' if AtoB else 'A']	# gt
		self.input_A.resize_(input_A.size()).copy_(input_A)
		self.input_B.resize_(input_B.size()).copy_(input_B)
		self.image_paths = input['A_paths' if AtoB else 'B_paths']

	def validation(self):
		if self.opt.which_model_netG == 'introAE':
			with torch.no_grad():
				self.real_A = torch.autograd.Variable(self.input_A)
				self.real_B = torch.autograd.Variable(self.input_B)
				self.netG.eval()
				self.latent, self.fake_B = self.netG(self.real_B)
				self.fake_B = self.fake_B.detach()
				self.latent = self.latent.detach()

				self.loss_G_Content = self.contentLoss.get_loss(self.fake_B, self.real_B)
				self.loss_G_L1 = self.contentLoss.get_l1loss(self.fake_B, self.real_B)
				self.loss_G_L2 = self.contentLoss.get_mseloss(self.fake_B, self.real_B)
		else:
			raise ValueError('This repo only support the autoencoder modified from introAE, i.e., opt.which_model_netG == introAE. \
			But you can use this option to add new model')

	def forward(self):
		self.netG.train()
		gpu_ids = self.opt.gpu_ids
		self.real_A = Variable(self.input_A)
		self.real_B = Variable(self.input_B)
		if self.opt.which_model_netG == 'introAE':
			self.latent, self.fake_B = self.netG(self.real_A)
		else:
			raise ValueError('This repo only support the autoencoder modified from introAE, i.e., opt.which_model_netG == introAE. \
			But you can use this option to add new model')

	# get image paths
	def get_image_paths(self):
		return self.image_paths

	def optimize_parameters(self):
		if self.opt.which_model_netG == 'introAE':
			self.forward()
			self.loss_G_Content = self.contentLoss.get_loss(self.fake_B, self.real_B)
			self.loss_G_L1 = self.contentLoss.get_l1loss(self.fake_B, self.real_B)
			self.loss_G_L2 = self.contentLoss.get_mseloss(self.fake_B, self.real_B)
			self.loss_G = self.loss_G_L1*0.001 + self.loss_G_Content*5	# only use L1 and perceptual loss
			self.optimizer_G_E1.zero_grad()
			self.optimizer_G_D.zero_grad()
			self.loss_G.backward()
			self.optimizer_G_E1.step()
			self.optimizer_G_D.step()
		else:
			raise ValueError('This repo only support the autoencoder modified from introAE, i.e., opt.which_model_netG == introAE. \
			But you can use this option to add new model')

	def get_current_errors_ae(self):
		return OrderedDict([('AE_Loss', self.loss_AE.data[0])
							])

	def get_current_errors(self):
		return OrderedDict([('G_percetual No', self.loss_G_Content.item()),
							('G_L1 *1', self.loss_G_L1.item()),
							('G_L2 *1000', self.loss_G_L2.item())
							])

	def get_current_errors_val(self):
		return OrderedDict([
			('G_percetual_val', self.loss_G_Content.item()),
			('G_L1_val', self.loss_G_L1.item()),
			('G_L2_val', self.loss_G_L2.item())
								])

	def get_current_visuals(self):
		real_A = util.tensor2im(self.real_A.data)
		fake_B = util.tensor2im(self.fake_B.data)
		real_B = util.tensor2im(self.real_B.data)
		return OrderedDict([('Blurred_Train', real_A), ('Restored_Train', fake_B), ('Sharp_Train', real_B)])


	def save(self, label):
		if self.opt.which_model_netG == 'introAE':
			self.save_network(self.netG.decoder, 'G_Decoder', label, self.gpu_ids)
			self.save_network(self.netG.encoder, 'G_Encoder1', label, self.gpu_ids)
		else:
			raise ValueError('This repo only support the autoencoder modified from introAE, i.e., opt.which_model_netG == introAE. \
			But you can use this option to add new model')

	def update_learning_rate(self):
		lrd = self.opt.lr / self.opt.niter_decay
		lr = self.old_lr - lrd
		for param_group in self.optimizer_G_E1.param_groups:
			param_group['lr'] = lr
		for param_group in self.optimizer_G_D.param_groups:
			param_group['lr'] = lr
		print('update learning rate: %f -> %f' % (self.old_lr, lr))
		self.old_lr = lr
