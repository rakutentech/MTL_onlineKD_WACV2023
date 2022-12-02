## Online Knowledge distillation for multi task learning
 
This repository contains the source code of the paper, Online Knowledge distillation for multi task learning, authored by Geethu Miriam Jacob, Vishal Agarwal and Bjorn Stenger, published in WACV 2023.



## TODO

Upload pretrained weights (WIP)

Upload inference.py (WIP)



## Environment
The code runs in torch version 1.7.0 and torchvision version 0.8.0

Install the required packages using 
`pip install -r requirements.txt`

Alternatively, use the Dockerfile provided.



## Preparing Data

| Datasets                                         | How to get it?                                               | Comments                                                     |
| ------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| NYUv2                                            | Download from [here](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0) (288x384, 8.4G) | Pre-processed by [mtan](https://github.com/lorenmt/mtan)     |
| CityScapes                                       | Download from [here](https://drive.google.com/uc?id=1WrVMA_UZpoj7voajf60yIVaS_Ggl0jrH&export=download) (128x256, 2.2G) | Pre-processed by [adashare](https://github.com/sunxm2357/AdaShare)     |



## Experiments


For training single models, change argument 'task' and run
`python train_nyu_single.py`


For training models on DWA, UW, Gradnorm, RLW, KD-MTL, change argument 'weighting', 'model' , 'data_path' and run
`python train_nyu_SOTA.py`

For baseline MTL model, change argument 'weighting' as 'EW' and run 
`python train_nyu_SOTA.py`


For training our model, change 'data_path' argument and run
`python train_nyu_ours.py`



## Pretrained models
Single models
[Segmentation]()
[Depth]()
[Surface normal]()


MTL model:
[CNN model]()
[Transformer model]()

For inference of our MTL model, run
`python inference.py`



## Acknowledgement

We would like to thank the authors that release the public repositories as follow (listed in no particular order):  

[mtan,dwa](https://github.com/lorenmt/mtan), 
[crossstitch](https://github.com/lorenmt/mtan)
[rlw](https://openreview.net/forum?id=OdnNBNIdFul)
[uw](https://github.com/yaringal/multi-task-learning-example)
[gradnorm](https://github.com/hosseinshn/GradNorm)
[KDMTL](https://github.com/VICO-UoE/KD4MTL)



