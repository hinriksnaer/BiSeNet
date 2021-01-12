#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import random
import logging
import time
import itertools
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from barbar import Bar
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from lib.models import model_factory
from configs import cfg_factory
from lib.cityscapes_cv2 import get_data_loader
from tools.evaluate import eval_model
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg

# apex
has_apex = True
try:
    from apex import amp, parallel
except ImportError:
    has_apex = False


## fix all random seeds
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1,)
    parse.add_argument('--port', dest='port', type=int, default=44554,)
    parse.add_argument('--model', dest='model', type=str, default='bisenetv1',)
    parse.add_argument('--finetune-from', type=str, default=None,)
    return parse.parse_args()

args = parse_args()
cfg = cfg_factory[args.model]

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)

def set_model():
    net = model_factory[cfg.model_type](cfg.n_classes)

    if cfg.model_type == 'hardnet':
        net.apply(weights_init)
        pretrained_path='./hardnet_weights/hardnet_petite_base.pth'
        weights = torch.load(pretrained_path)
        net.base.load_state_dict(weights)

    if not args.finetune_from is None:
        net.load_state_dict(torch.load(args.finetune_from, map_location='cpu'))
    if cfg.use_sync_bn: net = set_syncbn(net)
    net.cuda()
    net.train()
    criteria_pre = OhemCELoss(0.7)
    criteria_aux = [OhemCELoss(0.7) for _ in range(cfg.num_aux_heads)]
    return net, criteria_pre, criteria_aux

def set_syncbn(net):
    if has_apex:
        net = parallel.convert_syncbn_model(net)
    else:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    return net


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': 0},
            {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr': cfg.lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=cfg.lr_start,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    return optim

def set_meters():
    time_meter = TimeMeter(cfg.max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(cfg.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters


def train():
    logger = logging.getLogger()

    is_dist = False

    ## dataset
    dl = get_data_loader(
            cfg.im_root, cfg.train_im_anns,
            cfg.ims_per_gpu, cfg.scales, cfg.cropsize,
            cfg.max_iter, mode='train', distributed=is_dist)

    valid = get_data_loader(
        cfg.im_root, cfg.val_im_anns,
            cfg.ims_per_gpu, cfg.scales, cfg.cropsize,
            cfg.max_iter, mode='val', distributed=is_dist
    )

    ## model
    net, criteria_pre, criteria_aux = set_model()
    print(net)
    print(f'n_parameters: {sum(p.numel() for p in net.parameters())}')
    ## optimizer
    optim = set_optimizer(net)

    ## fp16
    if has_apex:
        opt_level = 'O1' if cfg.use_fp16 else 'O0'
        net, optim = amp.initialize(net, optim, opt_level=opt_level)

    ## meters
    if 'bisenet' in cfg.model_type:
        time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()
    if 'hardnet' in cfg.model_type:
        time_meter, loss_meter, _, _ = set_meters()

    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=cfg.max_iter, warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

    best_validation = np.inf

    best_miou = 0

    for i in range(cfg.n_epochs):
        ## train loop
        for it, (im, lb) in enumerate(Bar(dl)):

            net.train()

            im = im.cuda()
            lb = lb.cuda()

            lb = torch.squeeze(lb, 1)

            optim.zero_grad()
            if 'bisenet' in cfg.model_type:
                logits, *logits_aux = net(im)
            else:
                logits = net(im)

            loss_pre = criteria_pre(logits, lb)

            if 'bisenet' in cfg.model_type:
                loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
                loss = loss_pre + sum(loss_aux)
            else:
                loss = loss_pre


            loss.backward()
            optim.step()
            torch.cuda.synchronize()
            lr_schdr.step()

            time_meter.update()
            loss_meter.update(loss.item())
            if 'bisenet' in cfg.model_type:
                loss_pre_meter.update(loss_pre.item())
                _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]
            
            del im
            del lb

        ## print training log message
        lr = lr_schdr.get_lr()
        lr = sum(lr) / len(lr)

        if 'bisenet' in cfg.model_type:
            print_log_msg(
                i, cfg.max_iter, lr, time_meter, loss_meter,
                loss_pre_meter, loss_aux_meters)
        else:
            print_log_msg(
                i, cfg.max_iter, lr, time_meter, loss_meter)


        heads, mious = eval_model(net, 1, cfg.im_root, cfg.test_im_anns, cfg.n_classes, cfg.cropsize)
        logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))

        if best_miou < mious[0]:
            print('new best performance, storing model')
            best_miou = mious[0]
            state = net.state_dict()
            torch.save(state,  osp.join(cfg.respth, 'best_validation.pth'))

            print('best_miou')
            print(best_miou)



        ##validation loop
        validation_loss = []
        '''
        for it, (im, lb) in enumerate(Bar(valid)):

            net.eval()

            im = im.cuda()
            lb = lb.cuda()

            lb = torch.squeeze(lb, 1)

            with torch.no_grad():
                logits, *logits_aux = net(im)

                if 'bisenet' in cfg.model_type:
                    logits, *logits_aux = net(im)
                else:
                    logits = net(im)

                loss_pre = criteria_pre(logits, lb)

                if 'bisenet' in cfg.model_type:
                    loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
                    loss = loss_pre + sum(loss_aux)
                else:
                    loss = loss_pre
                    
                validation_loss.append(loss.item())

            del im
            del lb
        ## print training log messag
        validation_loss = sum(validation_loss)/len(validation_loss)
        print(f'Validation loss: {validation_loss}')

        if best_validation > validation_loss:
            print('new best performance, storing model')
            best_validation = validation_loss
            state = net.state_dict()
            torch.save(state,  osp.join(cfg.respth, 'best_validation.pth'))
        '''

    ## dump the final model and evaluate the result
    save_pth = osp.join(cfg.respth, 'model_final.pth')
    logger.info('\nsave models to {}'.format(save_pth))
    state = net.state_dict()

    torch.save(state, save_pth)

    logger.info('\nevaluating the final model')
    torch.cuda.empty_cache()
    heads, mious = eval_model(net, 2, cfg.im_root, cfg.test_im_anns)
    logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))

    return


def main():



    torch.cuda.set_device(args.local_rank)
    
    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)

    cfg.respth = cfg.respth + '/' + cfg.model_type + '/' + datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
    os.makedirs(cfg.respth)

    setup_logger('{}-train'.format(cfg.model_type), cfg.respth)
    train()


if __name__ == "__main__":
    main()
