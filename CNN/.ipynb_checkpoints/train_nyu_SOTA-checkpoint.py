import torch, time, os, random, sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io as sio
from backbone import build_model
from utils import *

from create_dataset import NYUv2

sys.path.append('../utils')
from util.weighting import weight_update

import argparse

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


class Weight(torch.nn.Module):
    def __init__(self):
        super(Weight, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor([1.0, 1.0, 1.0]))
Weights = Weight().cuda()

class transformer(torch.nn.Module):
    def __init__(self):
        super(transformer, self).__init__()
        self.conv1 = torch.nn.Conv2d(2048, 2048, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(64, 64, 1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, inputs):
        results = []
        results.append(self.conv1(inputs))
        return results

def parse_args():
    parser = argparse.ArgumentParser(description= 'MTL for NYUv2')
    parser.add_argument('--data_root', default="/home/geethu.jacob/workspace_shared/MTL/DATA/RLW/NYU/", 
                        help='data root', type=str) 
    parser.add_argument('--gpu_id', default='0', help='gpu_id') 
    parser.add_argument('--model', default='MTAN', type=str, help='DMTL, MTAN, Cross_Stitch')
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--random_distribution', default='normal', type=str, 
                        help='normal, random_normal, uniform, dirichlet, Bernoulli, Bernoulli_1')
    parser.add_argument('--weighting', default='RLW', type=str, 
                        help='EW, RLW, UW, dwa, KD_MTL, gradnorm')
    return parser.parse_args()

params = parse_args()
print(params)

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id

if params.model in ['DMTL']:
    batch_size = 8
elif params.model in ['DMTL', 'Cross_Stitch']:
    batch_size = 4
elif params.model in ['MTAN', 'NDDRCNN']:
    batch_size = 4
    
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


        
model = build_model(dataset='NYUv2', model=params.model, 
                    weighting=params.weighting, 
                    random_distribution=params.random_distribution).cuda()
task_num = len(model.tasks)
opt = []
opt += model.parameters()
if params.weighting=='UW' or params.weighting=='gradnorm':
    opt += [Weights.weights]
if params.weighting=='KD_MTL':
    single_model = {}
    transformers = {}
    # need to run train_nyu_single.py and save models for all tasks before using this 
    for i in range(task_num):
        single_model[i] = build_model(dataset='NYUv2', model='DMTL', 
                    weighting='EW', tasks=[model.tasks[i]],
                    random_distribution=params.random_distribution).cuda()
        checkpoint = torch.load('{}segnet_weight_{}_best.pth.tar'.format('result/',model.tasks[i]))
        single_model[i].load_state_dict(checkpoint['state_dict'])
        transformers[i] = transformer().cuda()
    par = []
    for i in range(task_num):
        par += transformers[i].parameters()

    transformer_optimizer = optim.Adam(par, lr=1e-4, weight_decay=5e-4)   


optimizer = optim.Adam(opt, lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30')
total_epoch = 200
best_loss = 100
train_batch = len(nyuv2_train_loader)
avg_cost = torch.zeros([total_epoch, 24])

lambda_weight = torch.ones([task_num, total_epoch]).cuda()
dist_loss=None
# if params.save_grad:
g_norm = np.zeros([task_num, total_epoch, train_batch])
g_cos = np.zeros([task_num, total_epoch, train_batch])
batch_weight = torch.ones(task_num).cuda()
for epoch in range(total_epoch):
    s_t = time.time()
    cost = torch.zeros(24)
    T=1.0
    # apply Dynamic Weight Average
    if params.weighting == 'dwa':
        if epoch == 0 or epoch == 1:
            lambda_weight[:, epoch] = 1.0
        else:
            w_1 = avg_cost[epoch - 1, 0] / avg_cost[epoch - 2, 0]
            w_2 = avg_cost[epoch - 1, 3] / avg_cost[epoch - 2, 3]
            w_3 = avg_cost[epoch - 1, 6] / avg_cost[epoch - 2, 6]
            lambda_weight[0, epoch] = 3 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
            lambda_weight[1, epoch] = 3 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
            lambda_weight[2, epoch] = 3 * np.exp(w_3 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
    # iteration for all batches
    model.train()
    train_dataset = iter(nyuv2_train_loader)
    conf_mat = ConfMatrix(model.class_nb)
    for batch_index in range(train_batch):
        train_data, train_label, train_depth, train_normal = train_dataset.next()
        train_data, train_label = train_data.cuda(non_blocking=True), train_label.long().cuda(non_blocking=True)
        train_depth, train_normal = train_depth.cuda(non_blocking=True), train_normal.cuda(non_blocking=True)
        
        train_pred = model(train_data)
        feat_s = model.get_final_features(train_data)
        train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                      model_fit(train_pred[1], train_depth, 'depth'),
                      model_fit(train_pred[2], train_normal, 'normal')]
        loss_train = torch.zeros(3).cuda()
        for i in range(3):
            loss_train[i] = train_loss[i]
        if epoch ==0 and batch_index == 0:
            initial_loss_train = loss_train
        optimizer.zero_grad() 
        if params.weighting == 'KD_MTL':
            transformer_optimizer.zero_grad()
            dist_loss = []
            for i in range(task_num):
                with torch.no_grad():
                     feat_ti = single_model[i].get_final_features(train_data)
                feat_ti0 = feat_ti.detach()
                feat_ti0 = feat_ti0 / (feat_ti0.pow(2).sum(1) + 1e-6).sqrt().view(feat_ti0.size(0), 
                                                                                  1, feat_ti0.size(2),
                                                                                  feat_ti0.size(3))            
                feat_si = transformers[i](feat_s)
                feat_si0 = feat_si[0] / (feat_si[0].pow(2).sum(1) + 1e-6).sqrt().view(feat_si[0].size(0), 
                                                                                1, feat_si[0].size(2),
                                                                                feat_si[0].size(3))
                dist_0 = (feat_si0 - feat_ti0).pow(2).sum(1).mean()
                dist_loss.append(dist_0)
            lambda_ = [1, 1, 2]
            dist_loss = sum(dist_loss[i] * lambda_[i] for i in range(task_num))
        batch_weight = weight_update(params.weighting, loss_train, model, optimizer, epoch, 
                                     batch_index, task_num, clip_grad=False, scheduler=None, 
                                     random_distribution=params.random_distribution, 
                                     avg_cost=avg_cost[:,0:7:3], Weights=Weights,
                                     lambda_weight=lambda_weight,
                                     initial_loss_train=initial_loss_train, 
                                     dist_loss=dist_loss)
        if params.weighting == 'KD_MTL':
            transformer_optimizer.step()
        # accumulate label prediction for every pixel in training images
        conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

        cost[0] = train_loss[0].item()
        cost[3] = train_loss[1].item()
        cost[4], cost[5] = depth_error(train_pred[1], train_depth)
        cost[6] = train_loss[2].item()
        cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred[2], train_normal)
        avg_cost[epoch, :12] += cost[:12] / train_batch

    # compute mIoU and acc
    avg_cost[epoch, 1], avg_cost[epoch, 2] = conf_mat.get_metrics()
    loss_index = (avg_cost[epoch, 0] + avg_cost[epoch, 3] + avg_cost[epoch, 6]) / 3.0
    isbest = loss_index < best_loss
    # evaluating test data
    if isbest:
        print(params)
        best_loss = loss_index
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
                avg_cost[epoch, 12:] += cost[12:] / val_batch

            # compute mIoU and acc
            avg_cost[epoch, 13], avg_cost[epoch, 14] = conf_mat.get_metrics()

        scheduler.step()
        e_t = time.time()
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} ||'
            'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} || {:.4f}'
            .format(epoch, avg_cost[epoch, 0], avg_cost[epoch, 1], avg_cost[epoch, 2], avg_cost[epoch, 3],
                    avg_cost[epoch, 4], avg_cost[epoch, 5], avg_cost[epoch, 6], avg_cost[epoch, 7], avg_cost[epoch, 8],
                    avg_cost[epoch, 9], avg_cost[epoch, 10], avg_cost[epoch, 11], avg_cost[epoch, 12], avg_cost[epoch, 13],
                    avg_cost[epoch, 14], avg_cost[epoch, 15], avg_cost[epoch, 16], avg_cost[epoch, 17], avg_cost[epoch, 18],
                    avg_cost[epoch, 19], avg_cost[epoch, 20], avg_cost[epoch, 21], avg_cost[epoch, 22], avg_cost[epoch, 23], e_t-s_t))
