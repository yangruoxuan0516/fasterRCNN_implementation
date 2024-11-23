from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import xml.etree.ElementTree as ET
import torchvision
import random

'''
VOC2007-test
    └──VOC2007/
        ├── Annotations
            ├── 000001.xml
            ├── 000002.xml
            ├── ...
            └── 009963.xml
        ├── ImageSets
            └── Layout
                └── test.txt
        └── JPEGImages
            ├── 000001.jpg
            ├── 000002.jpg
            ├── ...
            └── 009963.jpg

VOC2007-trainval
    └──VOC2007/
        ├── Annotations
            ├── 000005.xml
            ├── 000007.xml
            ├── ...
            └── 009961.xml
        ├── ImageSets
            └── Layout
                ├── train.txt
                ├── trainval.txt
                └── val.txt
        └── JPEGImages
            ├── 000005.jpg
            ├── 000007.jpg
            ├── ...
            └── 009961.jpg
'''

class VOCDataset(Dataset):
    # def __init__(self, data_path='/Users/ruox/Documents/master/FasterRCNN/', split='trainval'):
    def __init__(self, data_path='/home/infres/ryang-23/FasterRCNN-PyTorch', split='trainval'):
        # set pathes
        if split == 'trainval':
            self.split = 'trainval'
            # data_path = os.path.join(data_path, 'VOC2007-trainval')
            data_path = os.path.join(data_path, 'VOC2007')
        elif split == 'test':
            self.split = 'test'
            data_path = os.path.join(data_path, 'VOC2007-test')
        # data_path = os.path.join(data_path, 'VOC2007')
        self.data_path = data_path
        self.annopath = os.path.join(self.data_path, 'Annotations', '%s.xml')
        self.imgpath = os.path.join(self.data_path, 'JPEGImages', '%s.jpg')
        self.imgsetpath = os.path.join(self.data_path, 'ImageSets', 'Main', '%s.txt')

        # get image ids
        self.ids = list()
        for line in open(self.imgsetpath % self.split):
            self.ids.append(line.strip())

        # set label classes, and its index mapping
        classes = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]
        classes = sorted(classes)
        classes = ['background'] + classes
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}

        # load images and annotations
        img_annotations = dict()
        for img_id in self.ids:
            img_annotation = dict()
            annotation = ET.parse(self.annopath % img_id).getroot()
            # get targets (pairs of label and bbox)
            targets = list()
            for obj in annotation.findall('object'):
                if obj.find('difficult').text == '0':
                    target = dict()
                    target['label'] = self.label2idx[obj.find('name').text]
                    bbox = [int(obj.find('bndbox/xmin').text) - 1,
                            int(obj.find('bndbox/ymin').text) - 1,
                            int(obj.find('bndbox/xmax').text) - 1,
                            int(obj.find('bndbox/ymax').text) - 1]
                    target['bbox'] = bbox
                    targets.append(target)
            img_annotation['targets'] = targets
            # get image size
            size = annotation.find('size')
            img_annotation['height'] = int(size.find('height').text)
            img_annotation['width'] = int(size.find('width').text)
            
            img_annotations[img_id] = img_annotation
        self.img_annotations = img_annotations
            

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        # get image id
        img_id = self.ids[index]

        # get image
        img = Image.open(self.imgpath % img_id)
        to_flip = False
        if self.split == 'train' and random.random() < 0.5:
            to_flip = True
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        img = torchvision.transforms.ToTensor()(img)

        # get annotation
        annotations = self.img_annotations[img_id]
        targets = dict()
        targets['bboxes'] = torch.as_tensor([target['bbox'] for target in annotations['targets']])
        targets['labels'] = torch.as_tensor([target['label'] for target in annotations['targets']])

        if to_flip:
            for idx, box in enumerate(targets['bboxes']):
                x1, y1, x2, y2 = box
                w = x2-x1
                im_w = img.shape[-1]
                x1 = im_w - x1 - w
                x2 = x1 + w
                targets['bboxes'][idx] = torch.as_tensor([x1, y1, x2, y2])


        return img_id, img, targets