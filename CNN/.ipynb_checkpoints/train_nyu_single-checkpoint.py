import torch, time, os, random, sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io as sio
from model import config
from backbone import build_model
from utils import *

from create_dataset import NYUv2

from util.weighting import weight_update
args=config.Args()

import argparse
data_path = "/home/geethu.jacob/workspace_shared/MTL/DATA/RLW/NYU/"

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
        
def parse_args():
    parser = argparse.ArgumentParser(description= 'MTL for NYUv2')
    parser.add_argument('--data_root', default=data_path, help='data root', type=str) 
    parser.add_argument('--gpu_id', default='0', help='gpu_id') 
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--random_distribution', default='normal', type=str, 
                        help='normal, random_normal, uniform, dirichlet, Bernoulli, Bernoulli_1')
    parser.add_argument('--weighting', default='EW', type=str, help='EW, RLW')
    parser.add_argument('--task', default='normal', type=str, help='task: segmentation, depth, normal')
    parser.add_argument('--out', default='result', help='Directory to output the result')

    return parser.parse_args()

params = parse_args()
print(params)
def save_checkpoint(state, is_best, checkpoint=params.out, filename='best.pth.tar'):
    filepath = os.path.join(checkpoint, 'segnet_weight_{}_'.format(params.task) + filename)
    torch.save(state, filepath)

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id

batch_size = 8
# elif params.model in ['DMTL', 'Cross_Stitch']:
#     batch_size = 4
# elif params.model in ['MTAN', 'NDDRCNN']:
#     batch_size = 4
    
nyuv2_train_set = NYUv2(root=params.data_root, mode='trainval', augmentation=params.aug)
nyuv2_test_set = NYUv2(root=params.data_root, mode='test', augmentation=False)

nyuv2_test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True)


nyuv2_train_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True)


        
model = build_model(dataset='NYUv2', args=args, tasks=[params.task],
                    weighting=params.weighting, 
                    random_distribution=params.random_distribution).cuda()
# print(model)
task_num = len(model.tasks)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
if params.task == 'segmentation':
    print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC\n')
elif params.task == 'depth':
    print('LOSS FORMAT: DEPTH_LOSS ABS_ERR REL_ERR\n')
elif params.task == 'normal':
    print('LOSS FORMAT: NORMAL_LOSS MEAN MED <11.25 <22.5 <30\n')
total_epoch = 300
best_loss = 100
train_batch = len(nyuv2_train_loader)
avg_cost = torch.zeros([total_epoch, 24])

lambda_weight = torch.ones([task_num, total_epoch, train_batch]).cuda()
# if params.save_grad:
g_norm = np.zeros([task_num, total_epoch, train_batch])
g_cos = np.zeros([task_num, total_epoch, train_batch])
for epoch in range(total_epoch):
    s_t = time.time()
    cost = torch.zeros(24)

    # iteration for all batches
    model.train()
    train_dataset = iter(nyuv2_train_loader)
    conf_mat = ConfMatrix(model.class_nb)
    for batch_index in range(train_batch):
        train_data, train_label, train_depth, train_normal = train_dataset.next()
        train_data, train_label = train_data.cuda(non_blocking=True), train_label.long().cuda(non_blocking=True)
        train_depth, train_normal = train_depth.cuda(non_blocking=True), train_normal.cuda(non_blocking=True)
        train_pred = model(train_data)
        if params.task == 'segmentation':
            loss = model_fit(train_pred[0], train_label, 'semantic')
            cost[0] = loss.item()
            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())
        elif params.task == 'depth':
            
            loss = model_fit(train_pred[0], train_depth, 'depth')
            cost[4], cost[5] = depth_error(train_pred[0], train_depth)
            cost[3] = loss.item()
        elif params.task == 'normal':
            loss = model_fit(train_pred[0], train_normal, 'normal')
            cost[6] = loss.item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred[0], train_normal)

        loss_train = loss

        batch_weight = weight_update(params.weighting, loss_train, model, optimizer, epoch, 
                                     batch_index, task_num, clip_grad=False, scheduler=None, 
                                     random_distribution=params.random_distribution, 
                                     avg_cost=avg_cost[:,0:7:3])
        
        avg_cost[epoch, :12] += cost[:12] / train_batch
    # compute mIoU and acc
    if params.task == 'segmentation':
        avg_cost[epoch, 1], avg_cost[epoch, 2] = conf_mat.get_metrics()
    if params.task == 'segmentation':
        loss_index = avg_cost[epoch,0]
    elif params.task == 'depth':
        loss_index = avg_cost[epoch,3]
    elif params.task == 'normal':
        loss_index = avg_cost[epoch,6]
    isbest = loss_index < best_loss
    print(loss_index, best_loss, isbest)   
    if isbest:
        print(params)
        best_loss = loss_index
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }, isbest)
        print('Saving best model......... \n')
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
                if params.task == 'segmentation':
                    val_loss = model_fit(val_pred[0], val_label, 'semantic')
                    cost[12] = val_loss.item()
                elif params.task == 'depth':
                    
                    val_loss = model_fit(val_pred[0], val_depth, 'depth')
                    cost[16], cost[17] = depth_error(val_pred[0], val_depth)
                    cost[15] = val_loss.item()
                elif params.task == 'normal':
                    val_loss = model_fit(val_pred[0], val_normal, 'normal')
                    cost[18] = val_loss.item()
                    cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(val_pred[0], val_normal)
                

                conf_mat.update(val_pred[0].argmax(1).flatten(), val_label.flatten())

                avg_cost[epoch, 12:] += cost[12:] / val_batch

            # compute mIoU and acc
            avg_cost[epoch, 13], avg_cost[epoch, 14] = conf_mat.get_metrics()

        e_t = time.time()
        if params.task == 'segmentation':
            print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f}'
              ' TEST: {:.4f} {:.4f} {:.4f}'
              .format(epoch, avg_cost[epoch, 0], avg_cost[epoch, 1], avg_cost[epoch, 2], 
                    avg_cost[epoch, 12], avg_cost[epoch, 13], avg_cost[epoch, 14]))
        elif params.task == 'depth':
            print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f}'
              ' TEST: {:.4f} {:.4f} {:.4f}'
              .format(epoch, avg_cost[epoch, 3], avg_cost[epoch, 4], avg_cost[epoch, 5], 
                    avg_cost[epoch, 15], avg_cost[epoch, 16], avg_cost[epoch, 17]))
        elif params.task == 'normal':
            print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
              ' TEST: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'
              .format(epoch, avg_cost[epoch, 6], avg_cost[epoch, 7], avg_cost[epoch, 8], 
                      avg_cost[epoch, 9], avg_cost[epoch, 10], avg_cost[epoch, 11],
                      avg_cost[epoch, 18], avg_cost[epoch, 19], avg_cost[epoch, 20], 
                      avg_cost[epoch, 21], avg_cost[epoch, 22], avg_cost[epoch, 23]))
