from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
from .introvae import *
import pdb

class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        self.isTrain = opt.isTrain
        BaseModel.initialize(self, opt)
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        use_parallel = 0
        
        if opt.which_model_netG == 'introAE':
            str_to_list = lambda x: [int(xi) for xi in x.split(',')]
            self.netG = IntroAE(norm=opt.norm, gpuId = opt.gpu_ids,cdim=3, hdim=opt.hdim, channels=str_to_list(opt.channels), image_size=opt.output_height)
            self.encoder2 =IntroAEEncoder(norm=opt.norm,cdim=3, hdim=opt.hdim, channels=str_to_list(opt.channels), image_size=opt.output_height)
            self.netG.cuda()
            self.encoder2.cuda()
            print(self.netG)
            self.load_network(self.netG.encoder, 'G_Encoder1', opt.which_epoch)
            self.load_network(self.netG.decoder, 'G_Decoder', opt.which_epoch)
            self.load_network(self.encoder2, 'G_Encoder2', opt.which_epoch)
        else:
            raise ValueError('This repo only support the autoencoder modified from introAE, i.e., opt.which_model_netG == introAE. \
			But you can use this option to add new model')


    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        temp = self.input_A.clone()
        temp.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_A = temp
        self.image_paths = input['A_paths']

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_B = Variable(self.input_B, volatile=True)
        if self.opt.which_model_netG == 'introAE':
            self.netG.eval()
            self.encoder2.eval()
            self.latent_t, self.fake_B = self.netG(self.real_B)
            self.latent_i = self.encoder2(self.real_A)
            self.fake_Bi = self.netG.decoder(self.latent_i)
        else: 
            raise ValueError('This repo only support the autoencoder modified from introAE, i.e., opt.which_model_netG == introAE. \
			But you can use this option to add new model')
            
    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_Bi = util.tensor2im(self.fake_Bi.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B),('fake_Bi', fake_Bi),('real_B', real_B)])
