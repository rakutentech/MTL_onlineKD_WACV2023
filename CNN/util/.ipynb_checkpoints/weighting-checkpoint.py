

import torch, random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def weight_update(weighting, loss_train, model, optimizer, epoch, batch_index, task_num,
                  clip_grad=False, scheduler=None, mgda_gn='l2', 
                  random_distribution=None, avg_cost=None, Weights=None, lambda_weight=None,
                 initial_loss_train=None, dist_loss=None):
    """
    weighting: weight method (EW, RLW)
    random_distribution: using in random (uniform, normal, random_normal, dirichlet, Bernoulli, Bernoulli_1)
    """
    batch_weight = torch.ones(task_num).cuda()
    optimizer.zero_grad()
    if weighting == 'EW':
        batch_weight = torch.ones(task_num).cuda()
    elif weighting == 'UW':
        for i in range(task_num):
            batch_weight[i] = torch.exp(-Weights.weights[i])
    elif weighting == 'dwa' or weighting == 'OTW':   
        for i in range(task_num):
            batch_weight[i] = lambda_weight[i, epoch]
    elif weighting == 'gradnorm':
        norms = []
        # compute gradient w.r.t. last shared conv layer's parameters
        W = model.backbone.layer4[2].conv3.weight
        for i in range(task_num):
            gygw = torch.autograd.grad(loss_train[i], W, retain_graph=True)
            norms.append(torch.norm(torch.mul(Weights.weights[i], gygw[0])))
        norms = torch.stack(norms)
        task_loss = loss_train

        loss_ratio = task_loss.data / initial_loss_train.data
        inverse_train_rate = loss_ratio / loss_ratio.mean()
        mean_norm = norms.mean()
        alpha = 1.5
        constant_term = mean_norm.data * (inverse_train_rate ** alpha)
        grad_norm_loss = (norms - constant_term).abs().sum()
        w_grad = torch.autograd.grad(grad_norm_loss, Weights.weights)[0]
        for i in range(task_num):
            batch_weight[i] = Weights.weights[i].data
        
    elif weighting == 'RLW' and random_distribution is not None:
        if random_distribution == 'uniform':
            batch_weight = F.softmax(torch.rand(task_num).cuda(), dim=-1)
        elif random_distribution == 'normal':
            batch_weight = F.softmax(torch.randn(task_num).cuda(), dim=-1)
        elif random_distribution == 'dirichlet':
            # https://en.wikipedia.org/wiki/Dirichlet_distribution#Random_number_generation
            alpha = 1
            gamma_sample = [random.gammavariate(alpha, 1) for _ in range(task_num)]
            dirichlet_sample = [v / sum(gamma_sample) for v in gamma_sample]
            batch_weight = torch.Tensor(dirichlet_sample).cuda()
        elif random_distribution == 'random_normal':
            batch_weight = F.softmax(torch.normal(model.random_normal_mean, model.random_normal_std).cuda(), dim=-1)
        elif random_distribution == 'Bernoulli':
            while True:
                w = torch.randint(0, 2, (task_num,))
                if w.sum()!=0:
                    batch_weight = w.cuda()
                    break
        elif len(random_distribution.split('_'))==2 and random_distribution.split('_')[0]=='Bernoulli':
            w = random.sample(range(task_num), k=int(random_distribution.split('_')[1]))
            batch_weight = torch.zeros(task_num).cuda()
            batch_weight[w] = 1.
        else:
            raise('no support {}'.format(random_distribution))
            
    loss = loss_train*batch_weight
    if weighting == 'UW':
        loss +=Weights.weights
    loss = torch.sum(loss)
    if weighting == 'KD_MTL':
        loss +=dist_loss
#         optimizer.zero_grad()
    loss.backward()
    
    if clip_grad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if weighting == 'gradnorm':
        Weights.weights.grad = torch.zeros_like(Weights.weights.data)
        Weights.weights.grad.data = w_grad.data
    optimizer.step()
    
    if weighting == 'gradnorm':
        Weights.weights.data = task_num * Weights.weights.data / Weights.weights.data.sum()
    
    if scheduler is not None:
        scheduler.step()
    if weighting != 'EW' and batch_weight is not None and (batch_index+1) % 200 == 0:
        print('{} weight: {}'.format(weighting, batch_weight.cpu()))
    return batch_weight