import argparse
import os
from PIL.Image import Image
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from lib_cyclegan.models import *
from lib_cyclegan.datasets import *
from lib_cyclegan.utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

# os.environ["CUDA_VISIBLE_DEVICES"]='0, 1, 2, 3, 4, 5, 6'
# torch.cuda.set_device(6)

parser = argparse.ArgumentParser()
# parser.add_argument("--epoch", type=int, default=10, help="epoch to start training from")
# parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
# parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
# parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
# parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=512, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
# parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
# parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
# parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
# parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument("--source_path", type=str, default="./datasets/Sim2Real/train/A", help="path of the source data")
parser.add_argument("--saved_path", type=str, default="./datasets/GAN_dataset", help="path of the generated data")
parser.add_argument("--model", type=str, default="./log_cyclegan/saved_models/Sim2Rail_s2r/G_AB_0", help="path of the generator")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs(opt.saved_path, exist_ok=True)

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)

if cuda:
    G_AB = G_AB.cuda()

G_AB.load_state_dict(torch.load(opt.model))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    # transforms.RandomCrop((opt.img_height, opt.img_width)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
]
# Training data loader
dataloader = DataLoader(
    # ImageDataset_names(opt.source_path, transforms_=transforms_, unaligned=True),
    ImageDataset_image_name(opt.source_path, transforms_=transforms_, unaligned=True),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
)

print('Generating')
print(len(dataloader))
# ----------
#  Generating
# ----------

prev_time = time.time()
with torch.no_grad():
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        image_name = batch["name"]

        # ------------------
        #  Generating
        # ------------------
        fake_B = G_AB(real_A)
        # print(os.path.join(opt.saved_path, image_name[0]))
        save_image(fake_B, os.path.join(opt.saved_path, image_name[0]), normalize=True)
