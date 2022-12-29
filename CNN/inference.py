import torch, time, os, random, sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io as sio

import warnings
warnings.filterwarnings("ignore")

import argparse


    
def parse_args():
    parser = argparse.ArgumentParser(description= 'MTL for NYUv2')
    parser.add_argument('--data_root', default="/home/geethu.jacob/workspace_shared/MTL/DATA/RLW/NYU/",
                        help='data root', type=str) 
    parser.add_argument('--gpu_id', default='0', help='gpu_id') 
    parser.add_argument('--model', default='DMTL', type=str, help='DMTL, MTAN')
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')

    return parser.parse_args()

params = parse_args()
from create_dataset import NYUv2
from backbone import build_model
from utils import *

    

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id
if params.model in ['DMTL']:
    batch_size = 8
elif params.model in ['MTAN']:
    batch_size = 4
    
nyuv2_train_set = NYUv2(root=params.data_root, mode='trainval', augmentation=params.aug)
nyuv2_test_set = NYUv2(root=params.data_root, mode='test', augmentation=False)

nyuv2_test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True)


   
model = build_model(dataset='NYUv2', model=params.model).cuda()
checkpoint = torch.load('result/MTLnet_model_{}_best.pth.tar'.format(params.model))
model.load_state_dict(checkpoint['state_dict'])
task_num = len(model.tasks)

from tqdm import trange, tqdm
tasks = model.tasks


print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30')

avg_cost = torch.zeros(24)
cost = torch.zeros(24)

model.eval()
conf_mat = ConfMatrix(model.class_nb)
with torch.no_grad():  # operations inside don't track history
    val_dataset = iter(nyuv2_test_loader)
    val_batch = len(nyuv2_test_loader)
    for k in range(val_batch):
        val_data, val_label, val_depth, val_normal = val_dataset.next()
        val_data, val_label = val_data.cuda(non_blocking=True), val_label.long().cuda(non_blocking=True)
        val_depth, val_normal = val_depth.cuda(non_blocking=True), val_normal.cuda(non_blocking=True)

        val_pred = model(val_data)
        val_loss = [model_fit(val_pred[0], val_label, 'semantic'),
                     model_fit(val_pred[1], val_depth, 'depth'),
                     model_fit(val_pred[2], val_normal, 'normal')]

        conf_mat.update(val_pred[0].argmax(1).flatten(), val_label.flatten())

        cost[12] = val_loss[0].item()
        cost[15] = val_loss[1].item()
        cost[16], cost[17] = depth_error(val_pred[1], val_depth)
        cost[18] = val_loss[2].item()
        cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(val_pred[2], val_normal)
        avg_cost[12:] += cost[12:] / val_batch

    # compute mIoU and acc
    avg_cost[13], avg_cost[14] = conf_mat.get_metrics()

#         scheduler.step()
e_t = time.time()
print('TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
    .format(avg_cost[12], avg_cost[13],
            avg_cost[14], avg_cost[15], 
            avg_cost[16], avg_cost[17], 
            avg_cost[18], avg_cost[19], 
            avg_cost[20], avg_cost[21], 
            avg_cost[22], avg_cost[23]))


