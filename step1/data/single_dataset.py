import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import re
# import pdb

class SingleDataset(BaseDataset):
    def initialize(self, opt):
        # pdb.set_trace()
        if opt.phase=='train':
            self.opt = opt
            self.root = opt.datarootTarget
            self.A_paths = []
            self.B_paths = []

            self.dir_A = re.split(',',opt.datarootData)
            for i in range(len(self.dir_A)):
                dir_A = self.dir_A[i]
                Apath = make_dataset(dir_A)
                Apath = sorted(Apath)
                self.A_paths = self.A_paths + Apath

            self.dir_B = re.split(',',opt.datarootTarget)
            for i in range(len(self.dir_B)):
                dir_B = self.dir_B[i]
                Bpath = make_dataset(dir_B)
                Bpath = sorted(Bpath)
                self.B_paths = self.B_paths + Bpath
            transform_list = [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

            self.transform = transforms.Compose(transform_list)

        elif opt.phase == 'val':
            self.opt = opt
            self.root = opt.datarootTarget

            self.A_paths = []
            self.B_paths = []

            self.dir_A = re.split(',',opt.datarootValData)
            for i in range(len(self.dir_A)):
                dir_A = self.dir_A[i]
                Apath = make_dataset(dir_A)
                Apath = sorted(Apath)
                self.A_paths = self.A_paths + Apath

            self.dir_B = re.split(',',opt.datarootValTarget)
            for i in range(len(self.dir_B)):
                dir_B = self.dir_B[i]
                Bpath = make_dataset(dir_B)
                Bpath = sorted(Bpath)
                self.B_paths = self.B_paths + Bpath
            transform_list = [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

            self.transform = transforms.Compose(transform_list)

        elif opt.phase == 'test':
            self.opt = opt
            self.root = opt.datarootTarget
            self.A_paths = []
            self.B_paths = []

            self.dir_A = re.split(',',opt.datarootData)
            for i in range(len(self.dir_A)):
                dir_A = self.dir_A[i]
                Apath = make_dataset(dir_A)
                Apath = sorted(Apath)
                self.A_paths = self.A_paths + Apath

            self.dir_B = re.split(',',opt.datarootTarget)
            for i in range(len(self.dir_B)):
                dir_B = self.dir_B[i]
                Bpath = make_dataset(dir_B)
                Bpath = sorted(Bpath)
                self.B_paths = self.B_paths + Bpath
            transform_list = [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

            self.transform = transforms.Compose(transform_list)

        else:
            raise ValueError('The phase must be train, val or test.')

    def __getitem__(self, index):
        # load groundtruth
        B_path = self.B_paths[index]
        B_img = Image.open(B_path).convert('RGB')
        B_img = B_img.resize((256, 256), Image.BICUBIC)
        B_img = self.transform(B_img)

        # load input
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A_img = A_img.resize((256, 256), Image.BICUBIC)
        A_img = self.transform(A_img)

        return {'A': A_img, 'A_paths': A_path,'B': B_img, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
