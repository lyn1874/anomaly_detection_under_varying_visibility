import numpy as np
import os
import time
import torch
import argparse
import cv2
from PIL import Image
import io
from sklearn import metrics
import matplotlib.pyplot as plt
from config import update_config
import input_utils
from utils import psnr_error
import sys
from models.unet import UNet

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--dataset_type', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--dataset_path', default='/project_scratch/bo/anomaly_data/', type=str)
parser.add_argument('--dataset_augment_type', default='lightning', type=str)
parser.add_argument('--dataset_augment_test_type', default='frames/testing/', type=str)
parser.add_argument('--version', default=0, type=int)
parser.add_argument('--trained_model', default=None, type=str, help='The pre-trained model to evaluate.')
parser.add_argument('--show_curve', action='store_true',
                    help='Show and save the psnr curve real-timely, this drops fps.')
parser.add_argument('--show_heatmap', action='store_true',
                    help='Show and save the difference heatmap real-timely, this drops fps.')
parser.add_argument('--ckpt_step', type=int)


imh, imw = 256, 256
def give_score(psnr_group, gt):
    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)
    for i in range(len(psnr_group)):
        distance = psnr_group[i]
        distance -= min(distance)  # distance = (distance - min) / (max - min)
        distance /= max(distance)

        scores = np.concatenate((scores, distance), axis=0)
        labels = np.concatenate((labels, gt[i][4:]), axis=0)  # Exclude the first 4 unpredictable frames in gt.

    assert scores.shape == labels.shape, \
        f'Ground truth has {labels.shape[0]} frames, but got {scores.shape[0]} detected frames.'

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
    auc = metrics.auc(fpr, tpr)
    print(f'AUC: {auc}\n')
    return auc


def val(cfg, model=None):
    if model:
        test_folder = cfg.test_folder
        print("The test folder", test_folder)
    else:
        model_path = '/project/bo/exp_data/FFP/%s_%d/' % (cfg.dataset_type, cfg.version)
        ckpt_path = model_path + "model-%d.pth" % cfg.ckpt_step    
        if cfg.dataset_augment_test_type != "frames/testing/" and "venue" in cfg.dataset_type:
            rain_type = str(cfg.dataset_augment_test_type.strip().split('_')[0])
            brightness = int(cfg.dataset_augment_test_type.strip().split('_')[-1])/10
            data_dir = cfg.dataset_path + "Avenue/frames/%s_testing/bright_%.2f/" % (rain_type, brightness)
            if not os.path.exists(data_dir):
                aug_data.save_avenue_rain_or_bright(cfg.dataset_path, rain_type, True, "testing", bright_space=brightness)
        else:
            data_dir = cfg.dataset_path + '/%s/%s/' % ("Avenue", cfg.dataset_augment_test_type)
            rain_type = "original"
            brightness = 1.0
        test_folder = data_dir
        orig_stdout = sys.stdout
        f = open(os.path.join(model_path, 'output_rain_%s_bright_%s.txt' % (rain_type, brightness)),'w')
        sys.stdout= f
        cfg.gt = np.load('/project/bo/anomaly_data/Avenue/gt_label.npy', allow_pickle=True)

    if model:  # This is for testing during training.
        generator = model
        generator.eval()
    else:
        generator = UNet(input_channels=12, output_channel=3).cuda().eval()
        generator.load_state_dict(torch.load(ckpt_path)['net_g'])
#         generator.load_state_dict(torch.load('weights/' + cfg.trained_model)['net_g'])
        print("The pre-trained generator has been loaded from", ckpt_path)
#         print(f'The pre-trained generator has been loaded from \'weights/{cfg.trained_model}\'.\n')
    videos = {}
    videos, video_string = input_utils.setup(test_folder, videos)

    fps = 0
    psnr_group = []

    if not model:
        if cfg.show_curve:
            fig = plt.figure("Image")
            manager = plt.get_current_fig_manager()
            manager.window.setGeometry(550, 200, 600, 500)
            # This works for QT backend, for other backends, check this ⬃⬃⬃.
            # https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
            plt.xlabel('frames')
            plt.ylabel('psnr')
            plt.title('psnr curve')
            plt.grid(ls='--')

            cv2.namedWindow('target frames', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('target frames', 384, 384)
            cv2.moveWindow("target frames", 100, 100)

        if cfg.show_heatmap:
            cv2.namedWindow('difference map', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('difference map', 384, 384)
            cv2.moveWindow('difference map', 100, 550)

    with torch.no_grad():
        for i, folder in enumerate(video_string):

            if not model:
                name = folder.split('/')[-1]
                fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

                if cfg.show_curve:
                    video_writer = cv2.VideoWriter(f'results/{name}_video.avi', fourcc, 30, cfg.img_size)
                    curve_writer = cv2.VideoWriter(f'results/{name}_curve.avi', fourcc, 30, (600, 430))

                    js = []
                    plt.clf()
                    ax = plt.axes(xlim=(0, len(dataset)), ylim=(30, 45))
                    line, = ax.plot([], [], '-b')

                if cfg.show_heatmap:
                    heatmap_writer = cv2.VideoWriter(f'results/{name}_heatmap.avi', fourcc, 30, cfg.img_size)

            psnrs = []
            dataset = input_utils.test_dataset(videos[folder]['frame'], [imh, imw])
            print("Start video %s with %d frames...................." % (folder, len(dataset)))
            psnrs = []
            for j, clip in enumerate(dataset):
                input_np = clip[0:12, :, :]
                target_np = clip[12:15, :, :]
                input_frames = torch.from_numpy(input_np).unsqueeze(0).cuda()
                target_frame = torch.from_numpy(target_np).unsqueeze(0).cuda()
             
                G_frame = generator(input_frames)
                test_psnr = psnr_error(G_frame, target_frame).cpu().detach().numpy()
                psnrs.append(float(test_psnr))

                if not model:
                    if cfg.show_curve:
                        cv2_frame = ((target_np + 1) * 127.5).transpose(1, 2, 0).astype('uint8')
                        js.append(j)
                        line.set_xdata(js)  # This keeps the existing figure and updates the X-axis and Y-axis data,
                        line.set_ydata(psnrs)  # which is faster, but still not perfect.
                        plt.pause(0.001)  # show curve

                        cv2.imshow('target frames', cv2_frame)
                        cv2.waitKey(1)  # show video

                        video_writer.write(cv2_frame)  # Write original video frames.

                        buffer = io.BytesIO()  # Write curve frames from buffer.
                        fig.canvas.print_png(buffer)
                        buffer.write(buffer.getvalue())
                        curve_img = np.array(Image.open(buffer))[..., (2, 1, 0)]
                        curve_writer.write(curve_img)

                    if cfg.show_heatmap:
                        diff_map = torch.sum(torch.abs(G_frame - target_frame).squeeze(), 0)
                        diff_map -= diff_map.min()  # Normalize to 0 ~ 255.
                        diff_map /= diff_map.max()
                        diff_map *= 255
                        diff_map = diff_map.cpu().detach().numpy().astype('uint8')
                        heat_map = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)

                        cv2.imshow('difference map', heat_map)
                        cv2.waitKey(1)

                        heatmap_writer.write(heat_map)  # Write heatmap frames.

                torch.cuda.synchronize()
                end = time.time()
                if j > 1:  # Compute fps by calculating the time used in one completed iteration, this is more accurate.
                    fps = 1 / (end - temp)
                temp = end
#                 print(f'\rDetecting: [{i + 1:02d}] {j + 1}/{len(dataset)}, {fps:.2f} fps.', end='')

            psnr_group.append(np.array(psnrs))

            if not model:
                if cfg.show_curve:
                    video_writer.release()
                    curve_writer.release()
                if cfg.show_heatmap:
                    heatmap_writer.release()

    print('\nAll frames were detected, begin to compute AUC.')
    
    auc = give_score(psnr_group, cfg.gt)
    if not model:
        sys.stdout = orig_stdout
        f.close()

    return auc


if __name__ == '__main__':
    args = parser.parse_args()
    test_cfg = update_config(args, mode='test')
    test_cfg.print_cfg()
    val(test_cfg)

    