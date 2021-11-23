import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM
from collections import OrderedDict
import torch
import pdb
import os

def train(opt, train_data_loader,val_data_loader , model, visualizer):
	train_dataset = train_data_loader.load_data()
	val_dataset = val_data_loader.load_data()
	train_dataset_size = len(train_data_loader)
	val_dataset_size = len(val_data_loader)
	print('#training images = %d' % train_dataset_size)
	total_steps = 0
	start_iters = 0
	model.ganStep = 0
	model.aeStep = 1

	# train
	for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
		opt.phase = 'train'
		print('#train images = %d' % train_dataset_size)
		model.ganStep = 1
		epoch_start_time = time.time()
		epoch_iter = 0
		for i, data in enumerate(train_dataset):
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
					visualizer.plot_current_errors(epoch, float(epoch_iter)/train_dataset_size, opt, errors)

			if total_steps % opt.save_latest_freq == 0:
				print('saving the latest model (epoch %d, total_steps %d)' %
					  (epoch, total_steps))
				model.save('latest')

		# val
		if epoch % 1 == 0:
			opt.phase = 'val'
			print('start validation')
			print('#val images = %d' % val_dataset_size)
			G_otlatent_val = 0
			G_Content_val = 0
			G_latent_L1loss_val = 0
			for i, data in enumerate(val_dataset):
				model.set_input(data)
				model.validation()
				errors = model.get_current_errors_val()
				G_otlatent_val = G_otlatent_val+ errors['G_otlatent_val']
				G_Content_val = G_otlatent_val+ errors['G_Content_val']
				G_latent_L1loss_val = G_otlatent_val+ errors['G_latent_L1loss_val']

			errors = OrderedDict([('G_otlatent_val', G_otlatent_val/1000),
						('G_Content_val', G_Content_val/100),
						('G_latent_L1loss_val', G_latent_L1loss_val/1000)
					])
			if opt.display_id > 0:
				visualizer.plot_current_errors_val(epoch, float(epoch_iter)/train_dataset_size, opt, errors)
			print('G_otlatent_val %d ,G_Content_val %d, G_latent_L1loss_val %d' % (G_otlatent_val/1000,G_Content_val/100,G_latent_L1loss_val/1000))
			txtName = "val_loss.txt"
			filedir = os.path.join('./checkpoints/',opt.name,txtName)
			f=open(filedir, "a+")
			recordTime = 'Epoch=' + str(epoch) +'\n'
			new_context = 'G_otlatent_val = '+  str(G_otlatent_val/1000) + ';G_Content_val=' + str(G_Content_val/100) + ';G_latent_L1loss_val=' + str(G_latent_L1loss_val/1000) + '\n'
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
opt.phase = 'train'
train_data_loader = CreateDataLoader(opt)
opt.phase = 'val'
val_data_loader = CreateDataLoader(opt)
model = create_model(opt)
visualizer = Visualizer(opt)
train(opt, train_data_loader, val_data_loader,model, visualizer)




