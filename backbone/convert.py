# code from ruotian luo
# https://github.com/ruotianluo/pytorch-faster-rcnn

# weights download from
# https://github.com/jcjohnson/pytorch-vgg
import torch
from torch.utils.model_zoo import load_url
from torchvision import models

sd = torch.load("/Users/ruox/Documents/master/FasterRCNN/implementation_youtube/backbone/vgg16-00b39a1b.pth")
print(sd.keys())
sd['classifier.0.weight'] = sd['classifier.1.weight']
sd['classifier.0.bias'] = sd['classifier.1.bias']
del sd['classifier.1.weight']
del sd['classifier.1.bias']

sd['classifier.3.weight'] = sd['classifier.4.weight']
sd['classifier.3.bias'] = sd['classifier.4.bias']
del sd['classifier.4.weight']
del sd['classifier.4.bias']

import  os
# speicify the path to save
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
torch.save(sd, "vgg16_caffe.pth")