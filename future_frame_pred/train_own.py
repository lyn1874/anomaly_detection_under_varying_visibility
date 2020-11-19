import os
from glob import glob
import cv2
import time
import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import random
import sys

from utils import *
from losses import *
import input_utils 
from models.unet import UNet
from models.pix2pix_networks import PixelDiscriminator
from models.liteFlownet import lite_flownet as lite_flow
from config import update_config
from models.flownet2.models import FlowNet2SD
from evaluate_own import val

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--dataset_type', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--iters', default=40000, type=int, help='The total iteration number.')
parser.add_argument('--resume', default=None, type=str,
                    help='The pre-trained model to resume training with, pass \'latest\' or the model name.')
parser.add_argument('--save_interval', default=1000, type=int, help='Save the model every [save_interval] iterations.')
parser.add_argument('--val_interval', default=1000, type=int,
                    help='Evaluate the model every [val_interval] iterations, pass -1 to disable.')
parser.add_argument('--show_flow', default=False, action='store_true',
                    help='If True, the first batch of ground truth optic flow could be visualized and saved.')
parser.add_argument('--flownet', default='lite', type=str, help='lite: LiteFlownet, 2sd: FlowNet2SD.')
parser.add_argument("--dataset_path", default="/project_scratch/bo/anomaly_data/", type=str)
parser.add_argument("--dataset_augment_type", default="frames/training/", type=str)
parser.add_argument("--dataset_augment_test_type", default="frames/testing/", type=str)
parser.add_argument("--version", default=0, type=int)

args = parser.parse_args()

args.model_dir = '/project/bo/exp_data/FFP/%s_%d/' % (args.dataset_type, args.version)
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)


orig_stdout = sys.stdout
f = open(os.path.join(args.model_dir, 'log.txt'),'w')
sys.stdout= f

train_cfg = update_config(args, mode='train')
train_cfg.print_cfg()
    
for arg in vars(train_cfg):
    print(arg, getattr(train_cfg, arg))

train_cfg.gt = np.load('/project/bo/anomaly_data/Avenue/gt_label.npy', allow_pickle=True)

generator = UNet(input_channels=12, output_channel=3).cuda()
discriminator = PixelDiscriminator(input_nc=3).cuda()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=train_cfg.g_lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=train_cfg.d_lr)

if train_cfg.resume:
    generator.load_state_dict(torch.load(train_cfg.resume)['net_g'])
    discriminator.load_state_dict(torch.load(train_cfg.resume)['net_d'])
    optimizer_G.load_state_dict(torch.load(train_cfg.resume)['optimizer_g'])
    optimizer_D.load_state_dict(torch.load(train_cfg.resume)['optimizer_d'])
    print(f'Pre-trained generator and discriminator have been loaded.\n')
else:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    print('Generator and discriminator are going to be trained from scratch.\n')

assert train_cfg.flownet in ('lite', '2sd'), 'Flow net only supports LiteFlownet or FlowNet2SD currently.'
if train_cfg.flownet == '2sd':
    flow_net = FlowNet2SD()
    flow_net.load_state_dict(torch.load('models/flownet2/FlowNet2-SD.pth')['state_dict'])
else:
    flow_net = lite_flow.Network()
    flow_net.load_state_dict(torch.load('models/liteFlownet/network-default.pytorch'))

flow_net.cuda().eval()  # Use flow_net to generate optic flows, so set to eval mode.

adversarial_loss = Adversarial_Loss().cuda()
discriminate_loss = Discriminate_Loss().cuda()
gradient_loss = Gradient_Loss(3).cuda()
flow_loss = Flow_Loss().cuda()
intensity_loss = Intensity_Loss().cuda()

tr_folder, tt_folder = input_utils.give_data_folder(args.dataset_type, 
                                                    args.dataset_path,
                                                    args.dataset_augment_type,
                                                    args.dataset_augment_test_type)
train_cfg.train_folder = tr_folder
train_cfg.test_folder = tt_folder
frame_trans = input_utils.give_frame_trans()
train_dataset = input_utils.train_dataset(tr_folder, frame_trans, 256, 256, 4, 1)
print("There are images", len(train_dataset))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_cfg.batch_size, 
                              shuffle=True, num_workers=4, drop_last=True)

tb_folder = args.model_dir + "/tensorboard/"
if not os.path.exists(tb_folder):
    os.makedirs(tb_folder)
writer = SummaryWriter(tb_folder+"bs_%d" % train_cfg.batch_size)
# writer = SummaryWriter(f'tensorboard_log/{train_cfg.dataset}_bs{train_cfg.batch_size}')
start_iter = int(train_cfg.resume.split('_')[-1].split('.')[0]) if train_cfg.resume else 0
training = True
generator = generator.train()
discriminator = discriminator.train()

try:
    step = start_iter
    while training:
        for clips, flow_strs in train_dataloader:
            input_frames = clips[:, 0:12, :, :].cuda()  # (n, 12, 256, 256)
            target_frame = clips[:, 12:15, :, :].cuda()  # (n, 3, 256, 256)
            input_last = input_frames[:, 9:12, :, :].cuda()  # use for flow_loss
#             # pop() the used frame index, this can't work in train_dataset.__getitem__ because of multiprocessing.
#             for index in indice:
#                 train_dataset.all_seqs[index].pop()
#                 if len(train_dataset.all_seqs[index]) == 0:
#                     train_dataset.all_seqs[index] = list(range(len(train_dataset.videos[index]) - 4))
#                     random.shuffle(train_dataset.all_seqs[index])
            G_frame = generator(input_frames)
            if train_cfg.flownet == 'lite':
                gt_flow_input = torch.cat([input_last, target_frame], 1)
                pred_flow_input = torch.cat([input_last, G_frame], 1)
                # No need to train flow_net, use .detach() to cut off gradients.
                flow_gt = flow_net.batch_estimate(gt_flow_input, flow_net).detach()
                flow_pred = flow_net.batch_estimate(pred_flow_input, flow_net).detach()
            else:
                gt_flow_input = torch.cat([input_last.unsqueeze(2), target_frame.unsqueeze(2)], 2)
                pred_flow_input = torch.cat([input_last.unsqueeze(2), G_frame.unsqueeze(2)], 2)

                flow_gt = (flow_net(gt_flow_input * 255.) / 255.).detach()  # Input for flownet2sd is in (0, 255).
                flow_pred = (flow_net(pred_flow_input * 255.) / 255.).detach()

            if train_cfg.show_flow:
                flow = np.array(flow_gt.cpu().detach().numpy().transpose(0, 2, 3, 1), np.float32)  # to (n, w, h, 2)
                for i in range(flow.shape[0]):
                    aa = flow_to_color(flow[i], convert_to_bgr=False)
                    path = train_cfg.train_data.split('/')[-3] + '_' + flow_strs[i]
                    cv2.imwrite(f'images/{path}.jpg', aa)  # e.g. images/avenue_4_574-575.jpg
                    print(f'Saved a sample optic flow image from gt frames: \'images/{path}.jpg\'.')

#-----------------------COmment from here to 
#             inte_l = intensity_loss(G_frame, target_frame)
#             grad_l = gradient_loss(G_frame, target_frame)
#             fl_l = flow_loss(flow_pred, flow_gt)
#             g_l = adversarial_loss(discriminator(G_frame))
#             G_l_t = 1. * inte_l + 1. * grad_l + 2. * fl_l + 0.05 * g_l

#             # Train discriminator
#             # When training discriminator, don't train generator, so use .detach() to cut off gradients.
#             D_l = discriminate_loss(discriminator(target_frame), discriminator(G_frame.detach()))
#             optimizer_D.zero_grad()
#             D_l.backward()
#             optimizer_D.step()

#             # Train generator
#             optimizer_G.zero_grad()
#             G_l_t.backward()
#             optimizer_G.step()
#-------------------------------Here

            inte_l = intensity_loss(G_frame, target_frame)
            grad_l = gradient_loss(G_frame, target_frame)
            fl_l = flow_loss(flow_pred, flow_gt)
            g_l = adversarial_loss(discriminator(G_frame))
            G_l_t = 1. * inte_l + 1. * grad_l + 2. * fl_l + 0.05 * g_l

            # When training discriminator, don't train generator, so use .detach() to cut off gradients.
            D_l = discriminate_loss(discriminator(target_frame), discriminator(G_frame.detach()))

            # https://github.com/pytorch/pytorch/issues/39141
            # torch.optim optimizer now do inplace detection for module parameters since PyTorch 1.5
            # If I do this way:
            # ----------------------------------------
            # optimizer_D.zero_grad()
            # D_l.backward()
            # optimizer_D.step()
            # optimizer_G.zero_grad()
            # G_l_t.backward()
            # optimizer_G.step()
            # ----------------------------------------
            # The optimizer_D.step() modifies the discriminator parameters inplace.
            # But these parameters are required to compute the generator gradient for the generator.

            # Thus I should make sure no parameters are modified before calling .step(), like this:
            # ----------------------------------------
            # optimizer_G.zero_grad()
            # G_l_t.backward()
            # optimizer_G.step()
            # optimizer_D.zero_grad()
            # D_l.backward()
            # optimizer_D.step()
            # ----------------------------------------

            # Or just do .step() after all the gradients have been computed, like the following way:
            optimizer_D.zero_grad()
            D_l.backward()
            optimizer_G.zero_grad()
            G_l_t.backward()
            optimizer_D.step()
            optimizer_G.step()

            torch.cuda.synchronize()
            time_end = time.time()
            if step > start_iter:  # This doesn't include the testing time during training.
                iter_t = time_end - temp
            temp = time_end

            if step != start_iter:
                if step % 20 == 0:
                    time_remain = (train_cfg.iters - step) * iter_t
                    eta = str(datetime.timedelta(seconds=time_remain)).split('.')[0]
                    psnr = psnr_error(G_frame, target_frame)
                    lr_g = optimizer_G.param_groups[0]['lr']
                    lr_d = optimizer_D.param_groups[0]['lr']

                    print(f"[{step}]  inte_l: {inte_l:.3f} | grad_l: {grad_l:.3f} | fl_l: {fl_l:.3f} | "
                          f"g_l: {g_l:.3f} | G_l_total: {G_l_t:.3f} | D_l: {D_l:.3f} | psnr: {psnr:.3f} | "
                          f"iter: {iter_t:.3f}s | ETA: {eta} | lr: {lr_g} {lr_d}")

                    save_G_frame = ((G_frame[0] + 1) / 2)
                    save_G_frame = save_G_frame.cpu().detach()[(2, 1, 0), ...]
                    save_target = ((target_frame[0] + 1) / 2)
                    save_target = save_target.cpu().detach()[(2, 1, 0), ...]

                    writer.add_scalar('psnr/train_psnr', psnr, global_step=step)
                    writer.add_scalar('total_loss/g_loss_total', G_l_t, global_step=step)
                    writer.add_scalar('total_loss/d_loss', D_l, global_step=step)
                    writer.add_scalar('G_loss_total/g_loss', g_l, global_step=step)
                    writer.add_scalar('G_loss_total/fl_loss', fl_l, global_step=step)
                    writer.add_scalar('G_loss_total/inte_loss', inte_l, global_step=step)
                    writer.add_scalar('G_loss_total/grad_loss', grad_l, global_step=step)
                    writer.add_scalar('psnr/train_psnr', psnr, global_step=step)

                if step % int(train_cfg.iters / 100) == 0:
                    writer.add_image('image/G_frame', save_G_frame, global_step=step)
                    writer.add_image('image/target', save_target, global_step=step)

                if step % train_cfg.save_interval == 0:
                    model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                                  'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
                    torch.save(model_dict, args.model_dir + "model-%02d.pth" % step)                    
                    print("Already saved:", args.model_dir + "model-%02d.pth" % step)

                if step % train_cfg.val_interval == 0 or step == 1:
                    auc = val(train_cfg, model=generator)
                    print(step, "auc score", auc)
                    writer.add_scalar('results/auc', auc, global_step=step)
                    generator.train()

            step += 1
            if step > train_cfg.iters:
                training = False
                model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                              'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
                torch.save(model_dict, args.model_dir + "model-%02d.pth" % step)
                sys.stdout = orig_stdout
                f.close()
                break


except KeyboardInterrupt:
    print(f'\nStop early, model saved: \'latest_{train_cfg.dataset}_{step}.pth\'.\n')

    if glob(f'weights_2/latest*'):
        os.remove(glob(f'weights_2/latest*')[0])

    model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                  'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
    torch.save(model_dict, args.model_dir + "model-%02d.pth" % step)
