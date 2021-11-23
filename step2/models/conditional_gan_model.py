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
from .networks import weights_init as weights_init
import math
import pdb
from torch.nn.parallel import data_parallel
from torch.nn.parallel import DistributedDataParallel
from geomloss import SamplesLoss

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
		self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
								   opt.fineSize, opt.fineSize)
		self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
								   opt.fineSize, opt.fineSize)

		use_parallel = 0

		if opt.which_model_netG == 'introAE':
			str_to_list = lambda x: [int(xi) for xi in x.split(',')]
			self.netG = IntroAE(norm=opt.norm, gpuId = opt.gpu_ids,cdim=3, hdim=opt.hdim, channels=str_to_list(opt.channels), image_size=opt.output_height)
			self.encoder2 =IntroAEEncoder(norm=opt.norm,cdim=3, hdim=opt.hdim, channels=str_to_list(opt.channels), image_size=opt.output_height)
			self.old_lr = opt.lr
			self.netG.cuda()
			self.encoder2.cuda()
			self.encoder2.apply(weights_init)
			which_data = opt.which_data
			which_ep = opt.which_ep

			# load E1 and D1
			self.load_ae(self.netG.encoder, which_ep,'E1', which_data,opt.norm)
			self.load_ae(self.netG.decoder, which_ep,'D', which_data,opt.norm)
			in_content = input('Press Enter to CONFIRM trained AE weight loaded')

			# continue train?
			if not self.isTrain or opt.continue_train:
				self.load_network(self.netG.encoder, 'G_Encoder1', opt.which_epoch)
				self.load_network(self.netG.decoder, 'G_Decoder', opt.which_epoch)
				self.load_network(self.encoder2, 'G_Encoder2', opt.which_epoch)

			if opt.lossType == 'L2':
				self.optimizer_G_E2 = torch.optim.AdamW(self.encoder2.parameters(),lr=opt.lr,weight_decay=0.01)
			else:
				self.optimizer_G_E2 = torch.optim.Adam(self.encoder2.parameters(),lr=opt.lr)
			self.contentLoss = init_loss(opt, self.Tensor)
			print('---------- Networks initialized -------------')
			networks.print_network(self.netG.encoder)
			networks.print_network(self.encoder2)
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
				self.encoder2.eval()
				self.latent_t, self.fake_B = self.netG(self.real_B)
				self.latent_t = self.latent_t.detach()
				self.fake_B = self.fake_B.detach()
				self.latent_i = self.encoder2(self.real_A)
				self.fake_Bi = self.netG.decoder(self.latent_i)
				loss_ot = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
				self.latent_otLoss = self.contentLoss.get_otloss(self.latent_i,self.latent_t)
				self.loss_G_Content = self.contentLoss.get_mseloss(self.fake_Bi, self.real_B)

	def forward(self):
		self.encoder2.train()
		self.real_A = Variable(self.input_A)
		self.real_B = Variable(self.input_B)
		if self.opt.which_model_netG == 'introAE':
			self.netG.eval()
			self.latent_t, self.fake_B = self.netG(self.real_B)
			self.latent_t = self.latent_t.detach()
			self.fake_B = self.fake_B.detach()
			self.latent_i = self.encoder2(self.real_A)
			self.fake_Bi = self.netG.decoder(self.latent_i)
		else: 
			raise ValueError('This repo only support the autoencoder modified from introAE, i.e., opt.which_model_netG == introAE. \
			But you can use this option to add new model')

	# get image paths
	def get_image_paths(self):
		return self.image_paths

	def backward_G_E2(self):
		if self.opt.which_model_netG == 'introAE':
			self.latent_otLoss = self.contentLoss.get_otloss(self.latent_i,self.latent_t)
			self.loss_G_Content = self.contentLoss.get_mseloss(self.fake_Bi, self.real_B)*0.0001
			if self.opt.which_data == 'MNIST':
				self.loss_E1D = self.latent_otLoss*0.0001
			if self.opt.which_data == 'chaomo':
				self.loss_E1D = self.latent_otLoss*0.001
			else:
				self.loss_E1D = self.latent_otLoss*0.001
			self.loss_E1D.backward()
		else: 
			raise ValueError('This repo only support the autoencoder modified from introAE, i.e., opt.which_model_netG == introAE. \
			But you can use this option to add new model')

	def optimize_parameters(self):
		if self.opt.which_model_netG == 'introAE':
			self.netG.encoder.eval()
			self.netG.decoder.eval()
			self.forward()
			self.optimizer_G_E2.zero_grad()
			self.backward_G_E2()
			self.optimizer_G_E2.step()
		else: 
			raise ValueError('This repo only support the autoencoder modified from introAE, i.e., opt.which_model_netG == introAE. \
			But you can use this option to add new model')

	def get_current_errors_ae(self):
		return OrderedDict([('AE_Loss', self.loss_AE.data[0])
							])

	def get_current_errors(self):
		return OrderedDict([('G_otlatent No', self.latent_otLoss.item()),
							('G_Content No', self.loss_G_Content.item())
							])

	def get_current_errors_val(self):
		return OrderedDict([('G_otlatent_val', self.latent_otLoss.item()),
							('G_Content_val', self.loss_G_Content.item())
						])

	def get_current_visuals(self):
		real_A = util.tensor2im(self.real_A.data)
		fake_B = util.tensor2im(self.fake_B.data)
		fake_Bi = util.tensor2im(self.fake_Bi.data)
		real_B = util.tensor2im(self.real_B.data)
		return OrderedDict([('Blurred_Train', real_A), ('Restored_Train', fake_B), ('Restored_Train_fromi', fake_Bi),('Sharp_Train', real_B)])


	def save(self, label):
		if self.opt.which_model_netG == 'introAE':
			self.save_network(self.encoder2, 'G_Encoder2', label, self.gpu_ids)
			self.save_network(self.netG.encoder, 'G_Encoder1', label, self.gpu_ids)
			self.save_network(self.netG.decoder, 'G_Decoder', label, self.gpu_ids)
		else: 
			raise ValueError('This repo only support the autoencoder modified from introAE, i.e., opt.which_model_netG == introAE. \
			But you can use this option to add new model')


	def update_learning_rate(self):
		lrd = self.opt.lr / self.opt.niter_decay
		lr = self.old_lr - lrd
		for param_group in self.optimizer_G_E2.param_groups:
			param_group['lr'] = lr
		print('update learning rate: %f -> %f' % (self.old_lr, lr))
		self.old_lr = lr
