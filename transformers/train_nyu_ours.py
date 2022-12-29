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
args=config.Args()


from util.weighting import weight_update

import argparse

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def isnan(num):
    if math.isnan(num):
        return 10
    else:
        return min(num,10)


class layer(torch.nn.Module):
    def __init__(self,num_layers):
        super(layer,self).__init__()
        
        self.sigma1=[]
        self.sigma2=[]
        self.sigma3=[]
        self.num_layers = num_layers
        for i in range(num_layers):
            self.sigma1.append(torch.nn.Parameter(torch.rand(1, requires_grad=True,device='cuda')))
        self.sigma1=nn.ParameterList(self.sigma1)
        for i in range(num_layers):
            self.sigma2.append(torch.nn.Parameter(torch.rand(1, requires_grad=True,device='cuda')))
        self.sigma2=nn.ParameterList(self.sigma2)
        for i in range(num_layers):
            self.sigma3.append(torch.nn.Parameter(torch.rand(1, requires_grad=True,device='cuda')))
        self.sigma3=nn.ParameterList(self.sigma3)
        
    def get_parameters(self):
        print("sigma1",torch.Tensor(self.sigma1))
        print("sigma2",torch.Tensor(self.sigma2))
        print("sigma3",torch.Tensor(self.sigma3))

    
def parse_args():
    parser = argparse.ArgumentParser(description= 'MTL for NYUv2')
    parser.add_argument('--data_root', default="/home/geethu.jacob/workspace_shared/MTL/DATA/RLW/NYU/",
                        help='data root', type=str) 
    parser.add_argument('--gpu_id', default='0', help='gpu_id') 
    parser.add_argument('--T', default=0.1, type=float, help='temperature for OTW')
    parser.add_argument('--model', default='ViT', type=str, help='ViT, Cross_Stitch')
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--warmup', default=True, help='warmup')
    parser.add_argument('--weighting', default='OTW', type=str, help='EW, OTW')
    parser.add_argument('--AFD', default=True, help='AFD component')
    parser.add_argument('--out', default='result', help='Directory to output the result')

    return parser.parse_args()

params = parse_args()
print(params)

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id

if params.model in ['ViT']:
    batch_size = 8
elif params.model in ['Cross_Stitch']:
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

def save_checkpoint(state, is_best, checkpoint=params.out, filename='best.pth.tar'):
    filepath = os.path.join(checkpoint, 'MTLnet_model_{}_'.format(params.model) + filename)
    torch.save(state, filepath)

single_model = {}
single_optimizer = {}
mtl = layer(12)    
model = build_model(dataset='NYUv2', args=args, tasks=['segmentation','depth','normal'],
                    weighting=params.weighting).cuda()
task_num = len(model.tasks)
opt = []
opt += model.parameters()
from tqdm import trange, tqdm
tasks = model.tasks

par = []
for i in range(3):
    single_model[tasks[i]] = build_model(dataset='NYUv2', args=args, tasks=[tasks[i]],
                                         weighting=params.weighting).cuda()
    
    single_optimizer[tasks[i]] = optim.Adam(single_model[tasks[i]].parameters(), lr=1e-4, weight_decay=1e-5)
    single_model[tasks[i]].train()   
    
if params.warmup==True:
    for i in range(task_num):  
        
        
        print('STARTING WARMUP FOR TASK ', i, '..........................')
        
        train_batch = len(nyuv2_train_loader)
        for epoch in trange(5, desc='Epoch ', leave=True):
            # iteration for all batches
            
            train_dataset = iter(nyuv2_train_loader)
            for batch_index in trange(train_batch, desc='batch ', leave=False): 
                train_data, train_label, train_depth, train_normal = train_dataset.next()
                train_data, train_label = train_data.cuda(non_blocking=True), train_label.long().cuda(non_blocking=True)
                train_depth, train_normal = train_depth.cuda(non_blocking=True), train_normal.cuda(non_blocking=True)

                train_pred = single_model[tasks[i]](train_data)
                if tasks[i] == 'segmentation':
                    loss = model_fit(train_pred[0], train_label, 'semantic')
                elif tasks[i] == 'depth':
                    loss = model_fit(train_pred[0], train_depth, 'depth')
                elif tasks[i] == 'normal':
                    loss = model_fit(train_pred[0], train_normal, 'normal')

                loss_train = loss
                single_optimizer[tasks[i]].zero_grad()
                loss = loss_train
                loss = torch.sum(loss)
                loss.backward()
                single_optimizer[tasks[i]].step()
                
optimizer = optim.Adam(opt, lr=1e-4, weight_decay=1e-5)
AFD_optimizer = optim.Adam(list(mtl.parameters()), lr=1e-3, weight_decay=5e-4) 
print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30')
total_epoch = 300
best_loss = 100
train_batch = len(nyuv2_train_loader)
avg_cost = torch.zeros([total_epoch, 24])

avg_cost_MTL = torch.zeros([total_epoch, task_num])
avg_cost_single = torch.zeros([total_epoch, task_num])
lambda_weight = torch.ones([task_num, total_epoch]).cuda()
dist_loss=None

batch_weight = torch.ones(task_num).cuda()
for epoch in range(total_epoch):
    
    s_t = time.time()
    cost = torch.zeros(24)
    T = params.T
    # iteration for all batches
    model.train()
    train_dataset = iter(nyuv2_train_loader)
    conf_mat = ConfMatrix(model.class_nb)
    if epoch == 0:
        lambda_weight[:, epoch] = 1.0
    else:
        if epoch == 1:
            w_1 = min(avg_cost_MTL[epoch-1, 0] / avg_cost_single[epoch-1, 0],1)
            w_2 = min(avg_cost_MTL[epoch-1, 1] / avg_cost_single[epoch-1, 1],1)
            w_3 = min(avg_cost_MTL[epoch-1, 2] / avg_cost_single[epoch-1, 2],1)

        else:
            w_1 = avg_cost_MTL[epoch - 1, 0] / avg_cost_single[epoch-1, 0]
            w_2 = avg_cost_MTL[epoch - 1, 1] / avg_cost_single[epoch-1, 1]
            w_3 = avg_cost_MTL[epoch - 1, 2] / avg_cost_single[epoch-1, 2]
            print(w_1,w_2,w_3)
        term1 = isnan(np.exp(w_1 / T));term2 = isnan(np.exp(w_2 / T));term3 = isnan(np.exp(w_3 / T))
        lambda_weight[0, epoch] = task_num*term1 / (term1 + term2 + term3)
        lambda_weight[1, epoch] = task_num*term2 / (term1 + term2 + term3)
        lambda_weight[2, epoch] = task_num*term3 / (term1 + term2 + term3)
    for batch_index in range(train_batch):
        train_data, train_label, train_depth, train_normal = train_dataset.next()
        train_data, train_label = train_data.cuda(non_blocking=True), train_label.long().cuda(non_blocking=True)
        train_depth, train_normal = train_depth.cuda(non_blocking=True), train_normal.cuda(non_blocking=True)
        
        train_pred = model(train_data)
        feat_s = model.get_features(train_data)
        
        train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                      model_fit(train_pred[1], train_depth, 'depth'),
                      model_fit(train_pred[2], train_normal, 'normal')]
        loss_train = torch.zeros(task_num).cuda()
        single_loss_train = torch.zeros(task_num).cuda()
        single_pred = single_model['segmentation'](train_data)
        single_loss_train[0] = model_fit(single_pred[0], train_label, 'semantic')
        single_pred = single_model['depth'](train_data)
        single_loss_train[1] = model_fit(single_pred[0], train_depth, 'depth')
        single_pred = single_model['normal'](train_data)
        single_loss_train[2] = model_fit(single_pred[0], train_normal, 'normal')

        for i in range(task_num):
            loss_train[i] = train_loss[i]
        optimizer.zero_grad() 
        
        for i in range(task_num):
            single_optimizer[tasks[i]].zero_grad()
        if params.weighting == 'EW':
            batch_weight = torch.ones(task_num).cuda()
        if params.weighting == 'OTW':
            for i in range(task_num):
                batch_weight[i] = lambda_weight[i, epoch]
        loss = torch.sum(loss_train*batch_weight)

        if params.AFD: 
            AFD_optimizer.zero_grad()                        
            feat_ti=[]
            for i in range(task_num):           
                feat_ti.append(single_model[tasks[i]].get_features(train_data))
            feat_si = feat_s
            for j in range(12):
                dist_0 = 0.0
                temp_t = []
                for i in range(task_num):

                    if i==0:
                        sig = mtl.sigma1
                    elif i==1:
                        sig = mtl.sigma2
                    elif i==2:
                        sig = mtl.sigma3 
                    feat_ti0 = feat_ti[i][j,:,1:].detach()
                    feat_ti0 = feat_ti0 / (feat_ti0.pow(2).sum(1) + 1e-6).sqrt().view(feat_ti0.size(0), 1,
                                                                                   feat_ti0.size(2)) 
                    temp_t.append(sig[j]*feat_ti0)
                feat_ti0 = sum(temp_t)


                feat_si0 = feat_si[j][:, task_num:]
                feat_si0 = feat_si0 / (feat_si0.pow(2).sum(1) + 1e-6).sqrt().view(feat_si0.size(0), 1, 
                                                                                    feat_si0.size(2))

                dist_0 += (feat_si0 - feat_ti0).pow(2).sum(1).mean()
            dist_loss = dist_0
            
            loss+=dist_loss
        loss.backward()
        if epoch<100:
            single_loss = torch.sum(single_loss_train)
            single_loss.backward()
            for i in range(task_num):
                single_optimizer[tasks[i]].step()
        optimizer.step()
        AFD_optimizer.step()
        
        conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())
        cost[0] = train_loss[0].item()
        cost[3] = train_loss[1].item()
        cost[4], cost[5] = depth_error(train_pred[1], train_depth)
        cost[6] = train_loss[2].item()
        cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred[2], train_normal)
        avg_cost[epoch, :12] += cost[:12] / train_batch
        
        avg_cost_MTL[epoch, 0] += train_loss[0].item() / train_batch 
        avg_cost_single[epoch, 0] += single_loss_train[0].item() / train_batch
        
        avg_cost_MTL[epoch, 1] += train_loss[1].item() / train_batch 
        avg_cost_single[epoch, 1] += single_loss_train[1].item() / train_batch
        
        avg_cost_MTL[epoch, 2] += train_loss[2].item() / train_batch 
        avg_cost_single[epoch, 2] += single_loss_train[2].item() / train_batch
    # compute mIoU and acc
    avg_cost[epoch, 1], avg_cost[epoch, 2] = conf_mat.get_metrics()
    loss_index = (avg_cost[epoch, 0] + avg_cost[epoch, 3] + avg_cost[epoch, 6]) / 3.0
    isbest = loss_index < best_loss
    # evaluating test data
    if isbest:
        print(params)
        best_loss = loss_index
        save_checkpoint({
            'state_dict': model.state_dict()
        }, isbest)
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

#         scheduler.step()
        e_t = time.time()
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} ||'
            'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} || {:.4f}'
            .format(epoch, avg_cost[epoch, 0], avg_cost[epoch, 1], avg_cost[epoch, 2], avg_cost[epoch, 3],
                    avg_cost[epoch, 4], avg_cost[epoch, 5], avg_cost[epoch, 6], avg_cost[epoch, 7], avg_cost[epoch, 8],
                    avg_cost[epoch, 9], avg_cost[epoch, 10], avg_cost[epoch, 11], avg_cost[epoch, 12], avg_cost[epoch, 13],
                    avg_cost[epoch, 14], avg_cost[epoch, 15], avg_cost[epoch, 16], avg_cost[epoch, 17], avg_cost[epoch, 18],
                    avg_cost[epoch, 19], avg_cost[epoch, 20], avg_cost[epoch, 21], avg_cost[epoch, 22], avg_cost[epoch, 23], e_t-s_t))

