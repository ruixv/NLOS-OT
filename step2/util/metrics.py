import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
from . import pytorch_ssim
import pdb

def gaussian(window_size, sigma):
	gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
	return gauss/gauss.sum()

def create_window(window_size, channel):
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
	return window

def SSIM(img1, img2):
	(_, channel, _, _) = img1.size()
	window_size = 11
	window = create_window(window_size, channel)
	mu1 = F.conv2d(img1, window, padding = window_size/2, groups = channel)
	mu2 = F.conv2d(img2, window, padding = window_size/2, groups = channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1*mu2

	sigma1_sq = F.conv2d(img1*img1, window, padding = window_size/2, groups = channel) - mu1_sq
	sigma2_sq = F.conv2d(img2*img2, window, padding = window_size/2, groups = channel) - mu2_sq
	sigma12 = F.conv2d(img1*img2, window, padding = window_size/2, groups = channel) - mu1_mu2

	C1 = 0.01**2
	C2 = 0.03**2

	ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
	return ssim_map.mean()

def ssim(img1,img2):
	# pdb.set_trace()
	img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0)/255.0
	img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0)/255.0   
	img1 = Variable( img1,  requires_grad=False)    # torch.Size([256, 256, 3])
	img2 = Variable( img2, requires_grad = False)
	# ssim_value = pytorch_ssim.ssim(img1, img2).item()
	ssim_value = float(pytorch_ssim.ssim(img1, img2))
	print(ssim_value)
	return ssim_value


def PSNR(img1, img2):
	# pdb.set_trace()
	img1 = np.float64(img1)
	img2 = np.float64(img2)
	mse = np.mean( (img1 - img2) ** 2 )
	if mse == 0:
		return 100
	PIXEL_MAX = 255.0
	psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
	print(psnr)
	return psnr

def getpsnr(img1, img2):
	# pdb.set_trace()
	D_sum = 0
	img1 = np.float64(img1)
	img2 = np.float64(img2)
	for i in range(3):
		D = np.array(img1[:,:,i] - img2[:,:,i])
		D[:,:] = D[:,:]**2
		D_sum = D_sum + D.sum()
	RMSE = D_sum/img1.size
	psnr = 10*math.log10(float(255**2)/RMSE)
	print('psnr_Liu',psnr)
	return psnr

# def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
# 	rgb_range = 255

#     diff = (sr - hr).data.div(rgb_range)
#     if benchmark:
#         shave = scale
#         if diff.size(1) > 1:
#             convert = diff.new(1, 3, 1, 1)
#             convert[0, 0, 0, 0] = 65.738
#             convert[0, 1, 0, 0] = 129.057
#             convert[0, 2, 0, 0] = 25.064
#             diff.mul_(convert).div_(256)
#             diff = diff.sum(dim=1, keepdim=True)
#     else:
#         shave = scale + 6

#     valid = diff[:, :, shave:-shave, shave:-shave]
#     mse = valid.pow(2).mean()

#     return -10 * math.log10(mse)