import random
import torch
import numpy as np
import cv2
import glob
import os
import scipy.io as scio
import torchvision.transforms as transforms
from collections import OrderedDict
from torch.utils.data import Dataset


def np_load_frame(filename, resize_h, resize_w):
    img = cv2.imread(filename)
    image_resized = cv2.resize(img, (resize_w, resize_h)).astype('float32')
    image_resized = (image_resized / 127.5) - 1.0  # to -1 ~ 1
#    image_resized = np.transpose(image_resized, [2, 0, 1])  # to (C, W, H)
    return image_resized


class train_dataset(Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step  # time_step = 4
        self._num_pred = num_pred  # num_pred=1
        if type(self.dir) is str:
            self.videos, self.video_name = setup(self.dir, self.videos)
        elif type(self.dir) is list:
            self.videos, self.video_name = setup_multiple(self.dir, self.videos)
        self.samples = self.get_all_samples(self.video_name)
    
    def get_all_samples(self, video_string):
        frames = []
        for video in video_string:
            for i in range(len(self.videos[video]['frame']) - self._time_step):
                frames.append(self.videos[video]['frame'][i])
        return frames

    def __getitem__(self, index):
        frame_name = int(self.samples[index].split('/')[-1].split('_')[-1].split('.jpg')[0])        
        if "venue" in self.dir or "venue" in self.dir[0]:
            video_name = self.samples[index].strip().split('frames/')[1].strip().split('_frame')[0]
        elif "UCSD" in self.dir:
            video_name = self.samples[index].split('/')[-1].split('_')[0]
            frame_name -= 1
        batch = []
        for i in range(self._time_step+self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name+i], self._resize_height, self._resize_width)
            if self.transform is not None:
                batch.append(self.transform(image))
        flow_str = video_name.strip().split('_')[-1]+"_%d-%d" % (frame_name+3, frame_name+4)
        return np.concatenate(batch, axis=0), flow_str      
        
    def __len__(self):
        return len(self.samples)

    
class train_dataset_original(Dataset):
    """
    No data augmentation.
    Normalized from [0, 255] to [-1, 1], the channels are BGR due to cv2 and liteFlownet.
    """
    def __init__(self, imshape, folder):
        self.img_h, self.img_w = imshape
        self.clip_length = 5        
        videos = {}
        videos, video_string = setup(folder, videos)
        self.videos = []
        self.all_seqs = []
        for single_string in video_string:
            all_imgs = videos[single_string]['frame']
            random_seq = list(range(len(all_imgs)-4))
            random.shuffle(random_seq)
            self.all_seqs.append(random_seq)
            self.videos.append(all_imgs)
    def __len__(self):  # This decide the indice range of the PyTorch Dataloader.
        return len(self.videos)

    def __getitem__(self, indice):  # Indice decide which video folder to be loaded.
        one_folder = self.videos[indice]
        video_clip = []
        start = self.all_seqs[indice][-1]  # Always use the last index in self.all_seqs.
        for i in range(start, start + self.clip_length):
            _im = np_load_frame(one_folder[i], self.img_h, self.img_w)
            _im = np.transpose(_im, (2, 0, 1))
            video_clip.append(_im)
        video_clip = np.array(video_clip).reshape((-1, self.img_h, self.img_w))
        video_clip = torch.from_numpy(video_clip)
        flow_str = f'{indice}_{start + 3}-{start + 4}'
        return indice, video_clip, flow_str
    

class test_dataset:
    def __init__(self, imgs, imshape):
        self.img_h, self.img_w = imshape
        self.clip_length = 5
        self.imgs = imgs
    def __len__(self):
        return len(self.imgs) - (self.clip_length - 1)  # The first [input_num] frames are unpredictable.

    def __getitem__(self, indice):
        video_clips = []
        for frame_id in range(indice, indice + self.clip_length):
            _im = np_load_frame(self.imgs[frame_id], self.img_h, self.img_w)
            _im = np.transpose(_im, (2, 0, 1))
            video_clips.append(_im)
#             video_clips.append(np_load_frame(self.imgs[frame_id], self.img_h, self.img_w))
        video_clips = np.array(video_clips).reshape((-1, self.img_h, self.img_w))
        return video_clips


def setup_multiple(path, videos):
    path_mom = path[0].strip().split('frames/')[0] + 'frames/'
    video_string_group = []
    video_filenames_group = []
    for single_path in path:
        _subpath = [single_path + v for v in os.listdir(single_path)]
        _video_string = np.unique([v.strip().split('frames/')[1].strip().split('_frame')[0] for v in _subpath])
        _video_string = sorted(_video_string, key=lambda s:int(s.strip().split('_')[-1]))
        video_string_group.append(_video_string)
        video_filenames_group.append(_subpath)
    video_string_group = [v for j in video_string_group for v in j]
    video_filenames_group = [v for j in video_filenames_group for v in j]
    for video in video_string_group:
        videos[video] = {}
        videos[video]['path'] = path_mom + video
        _subframe = [v for v in video_filenames_group if video + '_' in v]
        _subframe = sorted(_subframe, key=lambda s:int(s.strip().split('_')[-1].strip().split('.jpg')[0]))
        videos[video]['frame'] = np.array(_subframe)
        videos[video]['length'] = len(_subframe)
    return videos, video_string_group
    
        
def setup(path, videos):
    if "venue" in path:
        all_video_frames = np.array([path + v for v in os.listdir(path)])
        video_string = np.unique([v.strip().split('frames/')[1].strip().split('_frame')[0] for v in all_video_frames])
        video_string = sorted(video_string, key=lambda s:int(s.strip().split('_')[-1]))
    elif "UCSDped" in path:
        video_string = np.unique([v.strip().split('_')[0] for v in os.listdir(path)])
        video_string = sorted(video_string)
        all_video_frames = np.array([v for v in os.listdir(path) if '.jpg' in v])
    print(video_string)
    if "veneu" in path:
        path = path.strip().split('frames/')[0] + 'frames/'
    for video in video_string:
        videos[video] = {}
        videos[video]['path'] = path + video
        _subframe = [v for v in all_video_frames if video + '_' in v]
        if "venue" in path:
            _subframe = sorted(_subframe, key=lambda s:int(s.strip().split('_')[-1].strip().split('.jpg')[0]))
        else:
            _subframe = sorted(_subframe)            
        videos[video]['frame'] = np.array(_subframe)
        videos[video]['length'] = len(_subframe)
    return videos, video_string

class Label_loader:
    def __init__(self, cfg, video_folders):
        assert cfg.dataset in ('ped2', 'avenue', 'shanghaitech'), f'Did not find the related gt for \'{cfg.dataset}\'.'
        self.cfg = cfg
        self.name = cfg.dataset
        self.frame_path = cfg.test_data
        self.mat_path = f'{cfg.data_root + self.name}/{self.name}.mat'
        self.video_folders = video_folders

    def __call__(self):
        if self.name == 'shanghaitech':
            gt = self.load_shanghaitech()
        else:
            gt = self.load_ucsd_avenue()
        return gt

    def load_ucsd_avenue(self):
        abnormal_events = scio.loadmat(self.mat_path, squeeze_me=True)['gt']

        all_gt = []
        for i in range(abnormal_events.shape[0]):
            length = len(os.listdir(self.video_folders[i]))
            sub_video_gt = np.zeros((length,), dtype=np.int8)

            one_abnormal = abnormal_events[i]
            if one_abnormal.ndim == 1:
                one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))

            for j in range(one_abnormal.shape[1]):  # number of frames
                start = one_abnormal[0, j] - 1
                end = one_abnormal[1, j]

                sub_video_gt[start: end] = 1

            all_gt.append(sub_video_gt)

        return all_gt

    def load_shanghaitech(self):
        np_list = glob.glob(f'{self.cfg.data_root + self.name}/frame_masks/')
        np_list.sort()

        gt = []
        for npy in np_list:
            gt.append(np.load(npy))

        return gt


def give_data_folder(dataset_type, dataset_path, dataset_augment_type, dataset_augment_test_type):
    if dataset_type == "Avenue":
        train_folder = dataset_path + dataset_type + '/frames/' + dataset_augment_type + '/'
        test_folder = dataset_path + dataset_type + '/frames/' + dataset_augment_test_type + '/'

    elif dataset_type == "Avenue_bright":
        train_folder_parent = dataset_path + "Avenue/frames/"
        train_folder = [train_folder_parent + v for v in ["training/", "original_training/bright_0.30/", 
                                                          "original_training/bright_0.40/",                                                                                 "original_training/bright_0.50/",
                                                          "original_training/bright_0.60/",
                                                          "original_training/bright_0.70/",
                                                          "original_training/bright_0.80/",
                                                          "original_training/bright_0.90/",
                                                          "original_training/bright_1.10/"]]
        test_folder = dataset_path + "Avenue/frames/testing/"

    elif dataset_type == "Avenue_rain":
        train_folder_parent = dataset_path + "Avenue/frames/"
        train_folder = [train_folder_parent + v for v in ["training/", "heavy_training/bright_0.70/", "torrential_training/bright_0.70/"]]
        test_folder = dataset_path + "Avenue/frames/testing/"    

    elif dataset_type == "UCSDped2" or dataset_type == "UCSDped1":
        train_folder = dataset_path + dataset_type + "/Train_jpg/"
        test_folder = dataset_path + dataset_type + "/Test_jpg/"
    print("---The training folder", train_folder)
    print("---The testing folder", test_folder)
    return train_folder, test_folder


def give_frame_trans():
    frame_trans = transforms.Compose([transforms.ToTensor()])
    return frame_trans