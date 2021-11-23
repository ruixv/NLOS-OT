import os
import torch
import pdb


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename) 
        state_dict = torch.load(save_path)
        network.load_state_dict(state_dict)

    # load E1 and D1 trained in step 1
    def load_ae(self, network, which_ep, whichblock, which_data, which_norm):
        if self.opt.which_model_netG == 'introAE':
            if whichblock == 'E1':
                if which_data == 'MNIST':
                    if which_norm == 'batch':
                        save_filename = 'MNIST_batch_Encoder_tanh.pth'
                        weight_path = '/share2/data/AONLOS/Dataset/TrainedWeight'
                        save_path = os.path.join(weight_path, save_filename)
                        state_dict = torch.load(save_path)
                        # for k in list(state_dict.keys()):
                        #     if (k.find('running_mean')>0) or (k.find('running_var')>0):
                        #         del state_dict[k]
                        network.load_state_dict(state_dict)
                        print(save_filename + 'has been loaded')
                    elif which_norm == 'instance':
                        save_filename = 'MNIST_Instance_Encoder_tanh.pth'
                        weight_path = '/share2/data/AONLOS/Dataset/TrainedWeight'
                        save_path = os.path.join(weight_path, save_filename)
                        state_dict = torch.load(save_path)
                        network.load_state_dict(state_dict)
                        print(save_filename + 'has been loaded')
                elif which_data == 'chaomo':
                    if which_norm == 'batch':
                        save_filename = 'chaomo_batch_Encoder_tanh.pth'
                        weight_path = '/share2/data/AONLOS/Dataset/TrainedWeight'
                        save_path = os.path.join(weight_path, save_filename)
                        state_dict = torch.load(save_path)
                        network.load_state_dict(state_dict)
                        print(save_filename + 'has been loaded')
                    elif which_norm == 'instance':
                        save_filename = 'chaomo_Instance_Encoder_tanh.pth'
                        weight_path = '/share2/data/AONLOS/Dataset/TrainedWeight'
                        save_path = os.path.join(weight_path, save_filename)
                        state_dict = torch.load(save_path)
                        network.load_state_dict(state_dict)
                        print(save_filename + 'has been loaded')
                elif which_data == 'anime':
                    if which_norm == 'batch':
                        save_filename = 'anime_batch_Encoder_tanh_ep10.pth'
                        weight_path = '/share2/data/AONLOS/Dataset/TrainedWeight'
                        save_path = os.path.join(weight_path, save_filename)
                        state_dict = torch.load(save_path)
                        network.load_state_dict(state_dict)
                        print(save_filename + 'has been loaded')
                    elif which_norm == 'instance':
                        save_filename = 'anime_Instance_Encoder_tanh.pth'
                        weight_path = '/share2/data/AONLOS/Dataset/TrainedWeight'
                        save_path = os.path.join(weight_path, save_filename)
                        state_dict = torch.load(save_path)
                        network.load_state_dict(state_dict)
                        print(save_filename + 'has been loaded')
                elif which_data == 'stl10':
                    if which_norm == 'batch':
                        save_filename = 'stl10_batch_Encoder_tanh_ep' + which_ep + '.pth'
                        weight_path = '/share2/data/AONLOS/Dataset/TrainedWeight'
                        save_path = os.path.join(weight_path, save_filename)
                        state_dict = torch.load(save_path)
                        network.load_state_dict(state_dict)
                        print(save_filename + 'has been loaded')  

            elif whichblock == 'D':
                if which_data == 'MNIST':
                    if which_norm == 'batch':
                        save_filename = 'MNIST_batch_Decoder_tanh.pth'
                        weight_path = '/share2/data/AONLOS/Dataset/TrainedWeight'
                        save_path = os.path.join(weight_path, save_filename)
                        state_dict = torch.load(save_path)
                        # for k in list(state_dict.keys()):
                        #     if ((k.find('running_mean')>0) or (k.find('running_var')>0)) and (k.find('13')<=0):
                        #         del state_dict[k]
                        #         print('\n'.join(map(str,sorted(state_dict.keys()))))
                        network.load_state_dict(state_dict)
                        print(save_filename + 'has been loaded')
                    if which_norm == 'instance':
                        save_filename = 'MNIST_Instance_Decoder_tanh.pth'
                        weight_path = '/share2/data/AONLOS/Dataset/TrainedWeight'
                        save_path = os.path.join(weight_path, save_filename)
                        state_dict = torch.load(save_path)
                        network.load_state_dict(state_dict)
                        print(save_filename + 'has been loaded')
                elif which_data == 'chaomo':
                    if which_norm == 'batch':
                        save_filename = 'chaomo_batch_Decoder_tanh.pth'
                        weight_path = '/share2/data/AONLOS/Dataset/TrainedWeight'
                        save_path = os.path.join(weight_path, save_filename)
                        state_dict = torch.load(save_path)
                        network.load_state_dict(state_dict)
                        print(save_filename + 'has been loaded')
                    if which_norm == 'instance':
                        save_filename = 'chaomo_Instance_Decoder_tanh.pth'
                        weight_path = '/share2/data/AONLOS/Dataset/TrainedWeight'
                        save_path = os.path.join(weight_path, save_filename)
                        state_dict = torch.load(save_path)
                        network.load_state_dict(state_dict)
                        print(save_filename + 'has been loaded')
                elif which_data == 'anime':
                    if which_norm == 'batch':
                        save_filename = 'anime_batch_Decoder_tanh_ep10.pth'
                        weight_path = '/share2/data/AONLOS/Dataset/TrainedWeight'
                        save_path = os.path.join(weight_path, save_filename)
                        state_dict = torch.load(save_path)
                        network.load_state_dict(state_dict)
                        print(save_filename + 'has been loaded')
                    if which_norm == 'instance':
                        save_filename = 'anime_Instance_Decoder_tanh.pth'
                        weight_path = '/share2/data/AONLOS/Dataset/TrainedWeight'
                        save_path = os.path.join(weight_path, save_filename)
                        state_dict = torch.load(save_path)
                        network.load_state_dict(state_dict)
                        print(save_filename + 'has been loaded')
                elif which_data == 'stl10':
                    if which_norm == 'batch':
                        save_filename = 'stl10_batch_Decoder_tanh_ep' + which_ep + '.pth'
                        weight_path = '/share2/data/AONLOS/Dataset/TrainedWeight'
                        save_path = os.path.join(weight_path, save_filename)
                        state_dict = torch.load(save_path)
                        network.load_state_dict(state_dict)
                        print(save_filename + 'has been loaded')  
        else:
            raise ValueError('This repo only support the autoencoder modified from introAE, i.e., opt.which_model_netG == introAE. \
			But you can use this option to add new model')


    def update_learning_rate():
        pass
