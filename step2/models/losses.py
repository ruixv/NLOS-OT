import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.autograd as autograd
import numpy as np
import torchvision.models as models
import util.util as util
from torch.autograd import Variable
import pdb
import matplotlib.pylab as pl
import ot
###############################################################################
# Functions
###############################################################################

class PerceptualLoss():
	def contentFunc(self):
		conv_3_3_layer = 14
		cnn = models.vgg19()
		pre = torch.load('/share2/data/AONLOS/Dataset/TrainedWeight/'+'vgg19-dcbb9e9d.pth')
		cnn.load_state_dict(pre)
		cnn = cnn.features
		cnn = cnn.cuda()
		model = nn.Sequential()
		model = model.cuda()
		print('********************************')
		for i,layer in enumerate(list(cnn)):
			model.add_module(str(i),layer)
			if i == conv_3_3_layer:
				break
		return model

	def initialize(self, loss):
		self.criterion = loss
		self.contentFunc = self.contentFunc()
			
	def get_loss(self, fakeIm, realIm):
		f_fake = self.contentFunc.forward(fakeIm) 
		f_real = self.contentFunc.forward(realIm)
		f_real_no_grad = f_real.detach()
		loss = self.criterion(f_fake, f_real_no_grad)
		return loss

	def get_mseloss(self, fakeIm, realIm):
		loss_fn = nn.MSELoss(reduction='sum')
		loss = loss_fn(fakeIm, realIm.detach())
		return loss
	
	def get_l1loss(self,latent_i,latent_t):
		L1_loss = nn.L1Loss(reduction='sum')
		output = L1_loss(latent_i, latent_t.detach())
		return output

	def get_otloss(self, latent_i,latent_t):
		loss_ot = 0
		pi_loss = nn.L1Loss(reduction='sum')
		batchsize = latent_i.shape[0]
		if batchsize != latent_t.shape[0]:
			raise ValueError('The length of the two latent codes must be the same.')
		for ii in range(batchsize):
			for jj in range(batchsize):
				if ii == jj:
					c_loss = 1
				else:
					c_loss = 0
				loss_ot = loss_ot + c_loss * pi_loss(latent_i[ii],latent_t[jj].detach())
		return loss_ot


def init_loss(opt, tensor):
	ae_loss = None
	content_loss = None
	
	content_loss = PerceptualLoss()
	content_loss.initialize(nn.MSELoss())
	
	return content_loss