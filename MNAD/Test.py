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
from tqdm import tqdm
import cv2
import pickle
import math
import aug_data as aug_data
from collections import OrderedDict
import copy
import time
import model.utils as input_utils

from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
import glob

import argparse

print("------------------------------------------------------")
print("Torch Version", torch.__version__)
print("------------------------------------------------------")

parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset/', help='directory of data')
parser.add_argument('--dataset_augment_type', type=str, default="training", help='the augmented version or not augmented version')
parser.add_argument('--dataset_augment_test_type', type=str, default='original_testing', help='the augmented version')
parser.add_argument('--exp_dir', type=str, default='/project/bo/exp_data/MNAD_exp/', help='directory of log')
parser.add_argument('--version', type=int, default=0)
parser.add_argument('--decay_step', type=int, default=30)
parser.add_argument('--ckpt_step', type=int, default=30)
parser.add_argument('--model_dir', type=str, default="None", help='whether use pretrained model or not')

args = parser.parse_args()

torch.manual_seed(2020)

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance



if args.dataset_augment_test_type != "frames/testing/" and "venue" in args.dataset_type:
    rain_type = str(args.dataset_augment_test_type.strip().split('_')[0])
    brightness = int(args.dataset_augment_test_type.strip().split('_')[-1])/10
    data_dir = args.dataset_path + "Avenue/frames/%s_testing/bright_%.2f/" % (rain_type, brightness)
    if not os.path.exists(data_dir):
        aug_data.save_avenue_rain_or_bright(args.dataset_path, rain_type, True, "testing", bright_space=brightness)
else:
    data_dir = args.dataset_path + '/%s/%s/' % ("Avenue", args.dataset_augment_test_type)

test_folder = data_dir

test_transform = input_utils.give_frame_trans("Avenue", 
                                         [args.h, args.w])

test_dataset = input_utils.DataLoader(test_folder, test_transform,
                                      resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)
test_size = len(test_dataset)
test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

loss_func_mse = nn.MSELoss(reduction='none')

log_dir = os.path.join(args.exp_dir, args.dataset_type, args.dataset_augment_type + '_decaystep_%d_version_%d' % (args.decay_step,
                                                                                                                  args.version))


    

if args.model_dir == "None":
    model_dir = log_dir + '/model-00%d.pth' % args.ckpt_step
    m_items_dir = log_dir + '/keys-00%d.pt' % args.ckpt_step
else:
    model_dir = args.model_dir + "/Ped2_prediction_model.pth"
    m_items_dir = args.model_dir + "/Ped2_prediction_keys.pt"

print("model ckpt is saved at.....................", model_dir)
print("model key is saved at......................", m_items_dir)


orig_stdout = sys.stdout
if args.dataset_augment_test_type == "frames/testing/":
    first = "original_1.00"
else:
    first = args.dataset_augment_test_type
f = open(os.path.join(log_dir, 'output_%s_%d.txt' % (first, args.ckpt_step)),'w')
sys.stdout= f




model = torch.load(model_dir)
model.cuda()
m_items = torch.load(m_items_dir)

labels = np.load('/project/bo/SToA_ad/MNAD/data/frame_labels_avenue.npy', allow_pickle=True)

videos = OrderedDict()
videos, videos_list = input_utils.setup(test_folder, videos)
labels_list = []
label_length = 0
psnr_list = {}
feature_distance_list = {}

print('Evaluation of', args.dataset_type)

# Setting for video anomaly detection
for iterr, video_name in enumerate(videos_list):
    labels_list = np.append(labels_list, labels[0][4+label_length:videos[video_name]['length']+label_length])
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []
    feature_distance_list[video_name] = []
    print(iterr, video_name, label_length)
    
label_length = 0
video_num = 0
label_length += videos[videos_list[video_num]]['length']
m_items_test = m_items.clone()

save_file = ['original_1.0' if args.dataset_augment_test_type == "frames/testing/" else args.dataset_augment_test_type][0]


if os.path.isfile(log_dir + "/feature_stat_%s_%d.npy" % (save_file, args.ckpt_step)):
    psnr_list = pickle.load(open(log_dir + "/psnr_stat_%s_%d" % (save_file, args.ckpt_step), 'rb'))
    feature_distance_list = pickle.load(open(log_dir + "/feature_stat_%s_%d" % (save_file, args.ckpt_step), 'rb'))
    print(np.shape(psnr_list), np.shape(feature_distance_list))
    anomaly_score_total_list = []
    for video in videos_list:
        video_name = video
        anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]), 
                                         anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)
    anomaly_score_total_list = np.asarray(anomaly_score_total_list)
    accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))
    print('The result of ', args.dataset_type)
    print('AUC: %.2f' % (accuracy*100))
    exit()

model.eval()

print("The number of images........", len(test_batch))
print("The number of labels........", len(labels_list))

progress_bar = tqdm(test_batch)
for k,(imgs) in enumerate(progress_bar):
    progress_bar.update()
    if k == label_length-4*(video_num+1):
        video_num += 1
        label_length += videos[videos_list[video_num]]['length']
    imgs = Variable(imgs).cuda()
    outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)
    mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
    mse_feas = compactness_loss.item()
    # Calculating the threshold for updating at the test time
    point_sc = point_score(outputs, imgs[:,3*4:])
    if  point_sc < args.th:
        query = F.normalize(feas, dim=1)
        query = query.permute(0,2,3,1) # b X h X w X d
        m_items_test = model.memory.update(query, m_items_test, False)

    psnr_list[videos_list[video_num]].append(psnr(mse_imgs))
    feature_distance_list[videos_list[video_num]].append(mse_feas)

# Measuring the abnormality score and the AUC
pickle.dump(psnr_list, open(log_dir + "/psnr_stat_%s_%d" % (save_file, args.ckpt_step), 'wb'))
pickle.dump(feature_distance_list, open(log_dir + "/feature_stat_%s_%d" % (save_file, args.ckpt_step), 'wb'))


anomaly_score_total_list = []
for video in videos_list:
    video_name = video
    anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]), 
                                     anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)

anomaly_score_total_list = np.asarray(anomaly_score_total_list)

accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))

print('The result of ', args.dataset_type)
print('AUC: %.2f' % (accuracy*100))

sys.stdout = orig_stdout
f.close()


