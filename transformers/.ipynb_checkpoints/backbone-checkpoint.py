import torch, sys, copy
import torch.nn as nn
import torch.nn.functional as F


from model import config
# sys.path.append('../utils')
from util.basemodel import BaseModel
from model.factory import create_model

def build_model(dataset, args, weighting, tasks = None, random_distribution=None):
    model = MTL_Vit(dataset=dataset, args=args, weighting=weighting, tasks = tasks, random_distribution=random_distribution)
    
#     if model == 'DMTL':
#         model = DeepLabv3(dataset=dataset, weighting=weighting, tasks = tasks, random_distribution=random_distribution)
#     elif model == 'MTAN':
#         model = MTANDeepLabv3(dataset=dataset, weighting=weighting, tasks = tasks, random_distribution=random_distribution)
#     elif model == 'NDDRCNN':
#         model = NDDRCNN(dataset=dataset, weighting=weighting, random_distribution=random_distribution)
#     elif model == 'Cross_Stitch':
#         model = Cross_Stitch(dataset=dataset, weighting=weighting, random_distribution=random_distribution)
    return model

class MTL_Vit(BaseModel):
    def __init__(self, dataset='NYUv2', args=None, weighting=None, tasks=None, random_distribution=None):
        if dataset == 'NYUv2':
            self.class_nb = 13
            self.tasks = tasks
            if tasks is None:
                self.tasks = ['segmentation', 'depth', 'normal']
            self.num_out_channels = {'segmentation': 13, 'depth': 1, 'normal': 3}
        elif dataset == 'CityScape':
            self.class_nb = 7
            self.tasks = tasks
            if tasks is None:
                self.tasks = ['segmentation', 'depth']
            self.num_out_channels = {'segmentation': 7, 'depth': 1}
        else:
            raise('No support {} dataset'.format(dataset))
        self.task_num = len(self.tasks)
        super(MTL_Vit, self).__init__(task_num=self.task_num,
                                        weighting=weighting,
                                        random_distribution=random_distribution)
        cfg = config.load_config()  
        ##################################### Model Configuration #######################################################
        model_cfg = cfg["model"][args.backbone]
        if "mask_transformer" in args.decoder:
            decoder_cfg = cfg["decoder"]["mask_transformer"]
        else:
            decoder_cfg = cfg["decoder"][args.decoder]
        model_cfg["image_size"] = (args.input_height, args.input_width)
        model_cfg["backbone"] = args.backbone
        model_cfg["dropout"] = args.dropout
        model_cfg["drop_path_rate"] = args.drop_path
        decoder_cfg["name"] = args.decoder
        model_cfg["decoder"] = decoder_cfg
        model_cfg["normalization"] = args.normalization    
        model_cfg["n_cls"] = self.class_nb

        ###################################### Single models #########################################
        self.single_models = {}
        orig_model_cfg = copy.deepcopy(model_cfg)
        orig_args = copy.deepcopy(args)
        if 'segmentation' in self.tasks:
            args.tasks = ['seg']
            args.weight = [1,0,0]
            self.single_models['seg'] = create_model(model_cfg,args)
        if 'depth' in self.tasks:
            args.tasks = ['depth']
            args.weight = [0,1,0]

            self.single_models['depth'] = create_model(model_cfg,args)
        if 'normal' in self.tasks:
            args.tasks = ['sn']
            args.weight = [0,0,1]
            self.single_models['sn'] = create_model(model_cfg,args)
        self.args = args    
        orig_args.tasks = self.tasks
        self.model = create_model(orig_model_cfg,orig_args)
       
#         ch = [256, 512, 1024, 2048]       
#         self.all_stages = ['conv', 'layer1_without_conv', 'layer2', 'layer3', 'layer4']
#         self.backbone = ResnetDilated(resnet.__dict__['resnet50'](pretrained=True))
#         self.decoders = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])
        
    def forward(self, x):
#         img_size  = x.size()[-2:]
        B, _, H, W = x.shape
#         print(x.shape)
#         x = self.backbone(x)
#         self.rep = x
#         if self.rep_detach:
#             for tn in range(self.task_num):
#                 self.rep_i[tn] = self.rep.detach().clone()
#                 self.rep_i[tn].requires_grad = True
        out = [0 for _ in self.tasks]
        x = self.model.encoder.get_final_features(x)
        
        x = x[:, len(self.tasks):]
       
        for i, t in enumerate(self.tasks):
            if t in ['segmentation', 'segment_semantic']:
                out[i] = self.model.segmentation_head(x,H,W)
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'depth':       
                
                _, out[i] = self.model.depth_head(x,H,W) 
            if t == 'normal':
                out[i] = self.model.sn_head(x,H,W)
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
                
                
        return out
    
    def get_share_params(self):
        return self.backbone.parameters()
        
    def get_features(self,x):
        x = self.model.encoder.get_all_features(x) 
        return x
    
    def get_final_features(self,x):
        x = self.model.encoder.get_final_features(x)   
#         p=[]
#         for stage in self.backbone.:
#             x = self.backbone.forward_stage(x,stage)
#         p=x
        return x
    

