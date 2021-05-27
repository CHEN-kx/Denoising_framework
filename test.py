import torch
import torch.nn.functional as F

import config as cfg

from utils.metrics import *
from utils.utils import *
from utils.visualize import *
from models import __models__
from data.dataloader import load_data

def valid(data_valid, net, writer, epoch):
    net.eval()
    scale = 1.0
    psnrs = AverageMeter()
    ssims = AverageMeter()
    mrses = AverageMeter()    

    for _, (data_batch, data_name) in enumerate(data_valid):
        x_ft = data_batch[0].permute(cfg.permu).cuda()  # (b,h,w,c)->(b,c,h,w)
        x, ft = x_ft[:, :cfg.input_nc_x, :, :], x_ft[:, cfg.input_nc_x:, :, :]  # (b,c1,h,w),(b,c2,h,w)
        y_gt = data_batch[1].permute(cfg.permu).cuda()  # (b,3,h,w)

        x_ft = F.interpolate(x_ft, scale_factor=scale, mode='bicubic')
        x = F.interpolate(x, scale_factor=scale, mode='bicubic')
        ft = F.interpolate(ft, scale_factor=scale, mode='bicubic')
        y_gt = F.interpolate(y_gt, scale_factor=scale, mode='bicubic')

        y = net([x, ft])
        y_gt_cropped = crop_like(y_gt, y)

        # add image summary
        if cfg.type == 'diffuse':
            albedo = data_batch[2].permute(cfg.permu).cuda()
            albedo = F.interpolate(albedo, scale_factor=scale, mode='bicubic')
            albedo_cropped = crop_like(albedo, y)
            x_cropped = crop_like(x_ft[:, :3, :, :], y)
            x_img = x_cropped * (albedo_cropped + cfg.eps)
            y_img = y * (albedo_cropped + cfg.eps)
            y_gt_img = y_gt_cropped * (albedo_cropped + cfg.eps)
        else:
            x_cropped = crop_like(x_ft[:, :3, :, :], y)
            x_img = torch.exp(x_cropped) - 1.0
            y_img = torch.exp(y) - 1.0
            y_gt_img = torch.exp(y_gt_cropped) - 1.0

        img_summary = torch.cat([x_img[0], y_img[0], y_gt_img[0]], dim=1)  # (c,h,w)->(c,3h,w)
        save_image(cfg.name, data_name[0].replace(" ", ""), img_summary.cpu(), normalize=True, permute=True)

        y_img = tensor2img(y_img[0])
        y_gt_img = tensor2img(y_gt_img[0])
        psnr_k = calc_psnr(y_img, y_gt_img)
        ssim_k = calc_ssim(y_img, y_gt_img)
        mrse_k = calc_mrse(y_img, y_gt_img)
        psnrs.update(psnr_k, n=x.size(0))
        ssims.update(ssim_k, n=x.size(0))
        mrses.update(mrse_k, n=x.size(0))
        print(data_name, "psnr_k=", psnr_k, "ssim_k=", ssim_k)

    add_scalar_summary(writer, epoch, 'psnr', psnrs.avg, stage='valid')
    add_scalar_summary(writer, epoch, 'ssim', ssims.avg, stage='valid')
    add_scalar_summary(writer, epoch, 'mrse', mrses.avg, stage='valid')
    print('[%d/%d], type: %s, psnr: %f, ssim: %f, mrse: %f' %
          (epoch, cfg.epoch_num, cfg.type, psnrs.avg, ssims.avg, mrses.avg))

    net.train()
    return psnrs.avg

if __name__ == '__main__':
    print('Load data', flush=True)
    data_valid = load_data(False)
    print('data length: ', len(data_valid))
    
    print("Build networks.", flush=True)
    net = __models__[cfg.mode.lower()](cfg)    
    net = torch.nn.DataParallel(net).cuda().train()
    print(net)

    writer = create_writer(cfg.name)
    torch.cuda.set_device(cfg.gpu)

    valid(data_valid, net, writer, epoch=0)