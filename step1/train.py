import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM
import pdb
import torch
from collections import OrderedDict
import os

def train(opt, data_loader, model, visualizer):
	dataset = data_loader.load_data()
	dataset_size = len(data_loader)
	print('#training images = %d' % dataset_size)
	total_steps = 0
	start_iters = 0
	model.ganStep = 0
	model.aeStep = 1
	# start train
	for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
		opt.phase = 'train'
		data_loader = CreateDataLoader(opt)
		dataset = data_loader.load_data()
		dataset_size_train = len(data_loader)
		print('#train images = %d' % dataset_size_train)
		model.ganStep = 1
		epoch_start_time = time.time()
		epoch_iter = 0
		for i, data in enumerate(dataset):
			iter_start_time = time.time()
			total_steps += opt.batchSize
			epoch_iter += opt.batchSize
			model.set_input(data)
			model.optimize_parameters()

			if total_steps % opt.display_freq == 0:
				results = model.get_current_visuals()
				psnrMetric = PSNR(results['Restored_Train'],results['Sharp_Train'])
				print('PSNR on Train = %f' %
					  (psnrMetric))
				visualizer.display_current_results(results,epoch)

			if total_steps % opt.print_freq == 0:
				errors = model.get_current_errors()
				t = (time.time() - iter_start_time) / opt.batchSize
				visualizer.print_current_errors(epoch, epoch_iter, errors, t)
				if opt.display_id > 0:
					visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size_train, opt, errors)

			if total_steps % opt.save_latest_freq == 0:
				print('saving the latest model (epoch %d, total_steps %d)' %
					  (epoch, total_steps))
				model.save('latest')

		# start validation
		if epoch % 1 == 0:
			opt.phase = 'val'
			print('start validation')
			data_loader = CreateDataLoader(opt)
			dataset = data_loader.load_data()
			dataset_size = len(data_loader)
			print('#val images = %d' % dataset_size)
			loss_G_Content_val = 0
			loss_G_L1_val = 0
			loss_G_L2_val = 0
			for i, data in enumerate(dataset):
				model.set_input(data)
				model.validation()
				errors = model.get_current_errors_val()
				loss_G_Content_val = loss_G_Content_val+ errors['G_percetual_val']
				loss_G_L1_val = loss_G_L1_val+ errors['G_L1_val']
				loss_G_L2_val = loss_G_L2_val+ errors['G_L2_val']

			errors = OrderedDict([('G_percetual_val', loss_G_Content_val/1000),
						('G_L1_val', loss_G_L1_val/100),
						('G_L2_val', loss_G_L2_val/1000)
					])
			if opt.display_id > 0:
				visualizer.plot_current_errors_val(epoch, float(epoch_iter)/dataset_size_train, opt, errors)
			print('G_percetual_val %d ,G_L1_val %d, G_L2_val %d' % (loss_G_Content_val/1000,loss_G_L1_val/100,loss_G_L2_val/1000))
			txtName = "val_loss.txt"
			filedir = os.path.join('./checkpoints/',opt.name,txtName)
			f=open(filedir, "a+")
			recordTime = 'Epoch=' + str(epoch) +'\n'
			new_context = 'G_percetual_val = '+  str(loss_G_Content_val/1000) + ';G_L1_val=' + str(loss_G_L1_val/100) + ';G_L2_val=' + str(loss_G_L2_val/1000) + '\n'
			f.write(recordTime)
			f.write(new_context)
			torch.cuda.empty_cache()

		if epoch % opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' %
				  (epoch, total_steps))
			model.save('latest')
			model.save(epoch)

		print('End of epoch %d / %d \t Time Taken: %d sec' %
			  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

		if epoch > opt.niter:
			model.update_learning_rate() 

opt = TrainOptions().parse() 
data_loader = CreateDataLoader(opt)
model = create_model(opt)
visualizer = Visualizer(opt)
train(opt, data_loader, model, visualizer)




