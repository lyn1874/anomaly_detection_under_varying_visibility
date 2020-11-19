import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
import model.utils as input_utils
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from sklearn.metrics import roc_auc_score
from utils import *
import random

import argparse

print("--------------PyTorch VERSION:", torch.__version__)
print("..............device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset/', help='directory of data')
parser.add_argument('--dataset_augment_type', type=str, default="training", help='the augmented version or not augmented version')
parser.add_argument('--dataset_augment_test_type', type=str, default='original_testing', help='the augmented version')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
parser.add_argument('--version', type=int, default=0, help='experiment version')
parser.add_argument('--decay_step', type=int, default=30, help='learning rate decay steps')

args = parser.parse_args()

torch.manual_seed(2020)

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

for arg in vars(args):
    print(arg, getattr(args, arg))

train_folder, test_folder = input_utils.give_data_folder(args.dataset_type, 
                                                         args.dataset_path, 
                                                         args.dataset_augment_type, 
                                                         args.dataset_augment_test_type)

# Loading dataset
frame_transform = input_utils.give_frame_trans(args.dataset_type, [args.h, args.w])

train_dataset = input_utils.DataLoader(train_folder, frame_transform, 
                           resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

test_dataset = input_utils.DataLoader(test_folder, frame_transform, 
                          resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

train_size = len(train_dataset)
test_size = len(test_dataset)

print("There are %d training images and %d testing images" % (train_size, test_size))

train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)


# Model setting
model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)
params_encoder =  list(model.encoder.parameters()) 
params_decoder = list(model.decoder.parameters())
params = params_encoder + params_decoder
optimizer = torch.optim.Adam(params, lr = args.lr)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.decay_step], gamma=0.2)  # version 2
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)
model.cuda()


# Report the training process
log_dir = os.path.join(args.exp_dir, args.dataset_type, args.dataset_augment_type + '_decaystep_%d_version_%d' % (args.decay_step,
                                                                                                                  args.version))

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'),'w')
sys.stdout= f

loss_func_mse = nn.MSELoss(reduction='none')

# Training

m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda() # Initialize the memory items

for epoch in range(args.epochs):
    labels_list = []
    model.train()
    
    start = time.time()
    for j,(imgs) in enumerate(train_batch):
        
        imgs = Variable(imgs).cuda()
        outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(imgs[:,0:12], m_items, True)
        optimizer.zero_grad()
        loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:,12:]))
        loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
        loss.backward(retain_graph=True)
        optimizer.step()
    
    if epoch == 0 or epoch == 1:
        img_npy = imgs.detach().cpu().numpy()
        img_npy = np.transpose(img_npy[0], (1, 2, 0)) # [imh, imw, ch]
        img_npy = img_npy * 0.5 + 0.5
        for _imiter in range(np.shape(img_npy)[-1] // 3):
            cv2.imwrite(log_dir + "/im_%d_%d.jpg" % (epoch, _imiter), (img_npy[:, :, _imiter * 3 : (_imiter + 1) * 3] * 255.0).astype('uint8')[:,:,::-1])
        
    scheduler.step()
    
    print('----------------------------------------')
    print('Epoch:', epoch+1)
    print('Loss: Reconstruction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(loss_pixel.item(), compactness_loss.item(), separateness_loss.item()))
    print('Memory_items:')
    print(m_items)
    print('----------------------------------------')
    if epoch % 10 == 0 or epoch == args.epochs - 1:
        torch.save(model, log_dir + "/model-{:04d}.pth".format(epoch))
        torch.save(m_items, log_dir + "/keys-{:04d}.pt".format(epoch))
            
print('Training is finished')

sys.stdout = orig_stdout
f.close()



