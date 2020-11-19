#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 04.03.20 at 10:34
This function is used to create the test data at different illumintations for Avenue
@author: li
"""
import cv2 as cv2
import numpy as np
import random
import os
import argparse


parser = argparse.ArgumentParser(description='Prepare dataset')
parser.add_argument('--option', type=str, help="Either extract data from videos or augment the saved data")
parser.add_argument('--datapath', type=str, help="Data directory")
parser.add_argument('--rain_type', type=str, help="original, heavy or torrential")
parser.add_argument('--bright', type=int, help="illumination")

def save_avenue_frame(use_str, path2read):
    """Extract frames from the Avenue dataset
    use_str: "training" or "testing"
    path2read: the path to read the videos from the Avenue dataset
    path2write: the path to save the frames from the Avenue_dataset
    """
#     path2read="/project_scratch/bo/anomaly_data/Avenue_play/Avenue/%s_videos" % use_str
#     path2write="/project_scratch/bo/anomaly_data/Avenue_play/Avenue/frames/%s" % use_str
    path2write = path2read + "/frames/%s" % use_str
    path2read = path2read + "%s_videos" % use_str
    print("Reading avenue dataset from ", path2read)
    print("Saving the frames in", path2write)
    if not os.path.exists(path2write):
        os.makedirs(path2write)
    path_child = sorted(os.listdir(path2read))
    tot_num = 0.0
    for iterr, single_child in enumerate(path_child[:1]):
        video_path = path2read + '/' + single_child
        s_video_name = single_child.split('.avi')[0]
        cap = cv2.VideoCapture(video_path)
        i = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite(path2write + '/%s_video_%s_frame_%05d.jpg' % (use_str, s_video_name, i), frame)
            i += 1
        cap.release()
        cv2.destroyAllWindows()
        tot_num += i
        print("There are %d frames for video %s" % (i, single_child))


def save_avenue_rain_or_bright(data_path, rain_type, aug, train_or_test, bright_space=0.2):
    """Save the augmented avenue dataset
    data_path: the path to read the Avenue frames: /project_scratch/bo/anomaly_data/
    rain_type: original, heavy, torrential
    aug: whether applying the augmentation, True or False
    train_or_test: training frames or testing frames, "training" or "testing"
    bright_space: the illumination of the augmented frames, float 
    """
    data_augment_path = data_path + 'Avenue/frames/%s_%s/' % (rain_type, train_or_test)
    create_folder(data_augment_path)
    print(data_augment_path)
    tr, tt, imshape, targshape = read_avenue_data(data_path)
    im_use = [tr if train_or_test is "training" else tt][0]
    time_step = [1 if rain_type is not "original" else 10][0]
    num_iter = int(np.ceil(np.shape(im_use)[0] / time_step))
    print("There are %d images" % len(im_use))
    print("There are %d iterations with %d frames per iteration" % (num_iter, time_step))
    if bright_space == 1.0 and rain_type is "original":
        aug = False
    if "train" in train_or_test:
        if aug is True:
            bright_space = [bright_space]
        else:
            bright_space = [1.0]
    else:
        bright_space = [bright_space]
    print(bright_space)
    for bright in bright_space:
        dir_sub = data_augment_path + 'bright_%.2f' % bright
        create_folder(dir_sub)
        print("----Finishing brightness level %.2f--------" % bright)
        for single_iter in range(num_iter):
            tr_sub = im_use[single_iter * time_step:(single_iter + 1) * time_step]
            im = load_im(tr_sub)
            if aug is True:
                if rain_type is not "original":
                    im_aug = add_rain(im, slant=10, rain_type=rain_type, bright_coefficient=bright)
                else:
                    im_aug = darken(im, darkness_coeff=bright)
            else:
                im_aug = im
            save_im(dir_sub, im_aug, tr_sub, train_test=train_or_test)
        print("save %d frames in total" % (time_step * (single_iter + 1)))



def create_folder(tds):
    if not os.path.exists(tds):
        os.makedirs(tds)


def load_im(im_path):
    im_tot = [cv2.imread(i) for i in im_path]
    return im_tot


def save_im(im_dir, im_aug, im_file, train_test="training"):
    if train_test is "training":
        im_dir_list = [im_dir+'/'+v.strip().split('/training/')[1] for v in im_file]
    elif train_test is "testing":
        im_dir_list = [im_dir+'/'+v.strip().split('/testing/')[1] for v in im_file]
    for single_dir, single_im in zip(im_dir_list, im_aug):
        single_im = (cv2.resize(single_im/255.0, dsize=(224, 128))*255.0).astype('uint8')
        cv2.imwrite(single_dir, single_im)


def read_avenue_data(model_mom):
    tr_tot = []
    path_mom = model_mom + "Avenue/frames/"
    for tr_or_tt in ["training", "testing"]:
        path = path_mom + tr_or_tt
        all_path = [v for v in os.listdir(path) if '.jpg' in v]
        all_path = sorted(all_path, key=lambda s: int(s.strip().split('frame_')[1].strip().split('.jpg')[0]))
        all_path = sorted(all_path, key=lambda s: int(s.strip().split('video_')[1].strip().split('_frame')[0]))
        all_path = [path + '/' + v for v in all_path if 'jpg' in v]
        if tr_or_tt is "training":
            tr_tot = all_path
        elif tr_or_tt is "testing":
            tt_tot = all_path
    print("there are %d training images and %d test images" % (np.shape(tr_tot)[0], np.shape(tt_tot)[0]))
    imshape = np.array([360, 640, 3])
    targshape = np.array([128, 224, 3])
    return tr_tot, tt_tot, imshape, targshape


def is_list(x):
    return type(x) is list


def change_light(image, coeff):
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)  # Conversion to HLS
    image_hls = np.array(image_hls, dtype=np.float64)
    image_hls[:, :, 1] = image_hls[:, :, 1] * coeff  # scale pixel values up or down for channel 1(Lightness)
    if coeff > 1:
        image_hls[:, :, 1][image_hls[:, :, 1] > 255] = 255  # Sets all values above 255 to 255
    else:
        image_hls[:, :, 1][image_hls[:, :, 1] < 0] = 0
    image_hls = np.array(image_hls, dtype=np.uint8)
    image_rgb = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)  # Conversion to RGB
    return image_rgb


def darken(image, darkness_coeff=-1.0):  # function to darken the image
    if is_list(image):
        image_rgb = []
        image_list = image
        for img in image_list:
            if darkness_coeff == -1:
                darkness_coeff_t = 1 - random.uniform(0, 1)
            else:
                darkness_coeff_t = darkness_coeff
            image_rgb.append(change_light(img, darkness_coeff_t))
    else:
        if darkness_coeff == -1:
            darkness_coeff_t = 1 - random.uniform(0, 1)
        else:
            darkness_coeff_t = darkness_coeff
        image_rgb = change_light(image, darkness_coeff_t)
    return image_rgb


def generate_random_lines(imshape, slant, drop_length, rain_type):
    drops = []
    area = imshape[0] * imshape[1]
    no_of_drops = area // 600

    if rain_type.lower() == 'drizzle':
        no_of_drops = area // 770
        drop_length = 10
    elif rain_type.lower() == 'heavy':
        drop_length = 30
    elif rain_type.lower() == 'torrential':
        no_of_drops = area // 500
        drop_length = 60

    for i in range(no_of_drops):  # If You want heavy rain, try increasing this
        if slant < 0:
            x = np.random.randint(slant, imshape[1])
        else:
            x = np.random.randint(0, imshape[1] - slant)
        y = np.random.randint(0, imshape[0] - drop_length)
        drops.append((x, y))
    return drops, drop_length


def rain_process(image, slant, drop_length, drop_color, drop_width, rain_drops, bright_coef):
    image_t = image.copy()
    for rain_drop in rain_drops:
        cv2.line(image_t, (rain_drop[0], rain_drop[1]), (rain_drop[0] + slant, rain_drop[1] + drop_length), drop_color,
                 drop_width)
    image = cv2.blur(image_t, (4, 4))  # rainy view are blurry
    brightness_coefficient = bright_coef  # rainy days are usually shady
    image_hls = hls(image)  # Conversion to HLS
    image_hls[:, :, 1] = image_hls[:, :, 1] * brightness_coefficient  # scale pixel values down for channel 1(Lightness)
    image_rgb = rgb(image_hls, 'hls')  # Conversion to RGB
    return image_rgb


def hls(image, src='RGB'):
    if is_list(image):
        image_hls = []
        image_list = image
        for img in image_list:
            eval('image_HLS.append(cv2.cvtColor(img,cv2.COLOR_' + src.upper() + '2HLS))')
    else:
        image_hls = eval('cv2.cvtColor(image,cv2.COLOR_' + src.upper() + '2HLS)')
    return image_hls


def rgb(image, src='BGR'):
    if is_list(image):
        image_rgb = []
        image_list = image
        for img in image_list:
            eval('image_RGB.append(cv2.cvtColor(img,cv2.COLOR_' + src.upper() + '2RGB))')
    else:
        image_rgb = eval('cv2.cvtColor(image,cv2.COLOR_' + src.upper() + '2RGB)')
    return image_rgb


def add_rain(image, slant=-1, drop_length=20, drop_width=1, drop_color=(200, 200, 200), rain_type='None',
             bright_coefficient=0.7):  # (200,200,200) a shade of gray
    slant_extreme = slant
    if is_list(image):
        image_rgb = []
        image_list = image
        imshape = image[0].shape
        if slant_extreme == -1:
            slant = np.random.randint(-10, 10)  # generate random slant if no slant value is given
        rain_drops, drop_length = generate_random_lines(imshape, slant, drop_length, rain_type)
        for img in image_list:
            output = rain_process(img, slant_extreme, drop_length, drop_color, drop_width, rain_drops,
                                  bright_coef=bright_coefficient)
            image_rgb.append(output)
    else:
        imshape = image.shape
        if slant_extreme == -1:
            slant = np.random.randint(-10, 10)  # generate random slant if no slant value is given
        rain_drops, drop_length = generate_random_lines(imshape, slant, drop_length, rain_type)
        output = rain_process(image, slant_extreme, drop_length, drop_color, drop_width, rain_drops,
                              bright_coef=bright_coefficient)
        image_rgb = output
    return image_rgb


def add_blur(image, x, y, hw, fog_coeff):
    overlay = image.copy()
    output = image.copy()
    alpha = 0.08 * fog_coeff
    rad = hw // 2
    point = (x + hw // 2, y + hw // 2)
    cv2.circle(overlay, point, int(rad), (255, 255, 255), -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output


if __name__ == "__main__":
    args = parser.parse_args()
    if args.option == "extract":
        save_avenue_frame("training", args.datapath)
        save_avenue_frame("testing", args.datapath)
    elif args.option == "augment":
        save_avenue_rain_or_bright(args.datapath, args.rain_type, True, "training", bright_space=args.bright)
        



