import warnings
warnings.filterwarnings("ignore")

import os
import torch
import torch.optim as optim
import time
import argparse

import config as cfg
from models import __models__
from models.loss import getloss
from utils.checkpoint import *
from utils.utils import *
from utils.visualize import *
from data.dataloader import load_data
from test import valid

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

def train(data_train, data_valid, net, writer):

    criterion = getloss(cfg.loss)
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=0.0, betas=(0.9, 0.999))

    step, start = 0, time.time()
    psnr_max, is_best = -1, False

    if cfg.ckpt:
        print('Resume from checkpoint: %s' % cfg.ckpt)
        load_state_ckpt(cfg.ckpt, net)  

    print("Start training: mode=%s, lr=%f, x_nc=%d, ft_nc=%d" % (cfg.mode, cfg.lr, cfg.input_nc_x, cfg.input_nc_ft))
    for epoch in range(0, cfg.epoch_num):    
        # 1. training
        loss_epoch = 0
        t_epoch = time.time()
        for data_idx, data_batch in enumerate(data_train):
            t = time.time()
            step += 1
            x_ft = data_batch[0].permute(cfg.permu).cuda()  # (b,h,w,c)->(b,c,h,w)
            x, ft = x_ft[:, :cfg.input_nc_x, :, :], x_ft[:, cfg.input_nc_x:, :, :]  # (b,c1,h,w),(b,c2,h,w)
            y_gt = data_batch[1].permute(cfg.permu).cuda()  # (b,3,h,w)
            if torch.max(x_ft) > cfg.inf:
                continue

            optimizer.zero_grad()
            y = net([x, ft])
            y_gt_cropped = crop_like(y_gt, y)
            loss = criterion(y, y_gt_cropped)
            loss.backward()
            clip_gradient(optimizer, 1.0)
            optimizer.step()           

            if step % 100 == 0:
                add_scalar_summary(writer, step, cfg.name, loss.item(), stage="train")

            print('[%d/%d][%d/%d], type: %s, loss: %f, time: %.3fs' % (
                data_idx, len(data_train), epoch, cfg.epoch_num, cfg.type, loss.item(), time.time()-t))

            loss_epoch += loss.item()
        print('First epoch time is %f:' %(time.time()-t_epoch))
        print('[%d/%d], type: %s, loss_epoch: %f' % (epoch, cfg.epoch_num, cfg.type, loss_epoch))

        # 2. testing, we use PSNR as metric
        with torch.no_grad():
            psnr_val = valid(data_valid, net, writer, epoch)     

        # 3. save checkpoints
        if psnr_val > psnr_max:
            is_best = True
            psnr_max = psnr_val
        else:
            is_best = False

        if (epoch + 1) % cfg.save_epoch == 0:
            save_checkpoint(cfg.save, {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'psnr_val': psnr_val,
                'optimizer': optimizer.state_dict()
            }, is_best)        

    print("Finish training: mode=%s\n" % args.mode, flush=True)
    print("Take %.3f seconds.\n" % (time.time() - start), flush=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action='store_true', help='reset folder')
    args = parser.parse_args()
    if args.reset:
        os.system('rm -rf' + cfg.save)
    os.makedirs(cfg.save, exist_ok=True)

    print('Load data', flush=True)
    data_train, data_valid = load_data(cfg, True), load_data(cfg, False)
    print('data length: ', len(data_train), len(data_valid))
    
    print("Build networks.", flush=True)
    net = __models__[cfg.mode.lower()](cfg)    
    net = torch.nn.DataParallel(net).cuda().train()
    print(net)

    writer = create_writer(cfg.name)
    shutil.copyfile('config.py', os.path.join(cfg.save, 'config.py'))

    train(data_train, data_valid, net, writer)