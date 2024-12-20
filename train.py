import torch
import os
import sys
import tqdm
from torch.utils.data import DataLoader

from dataset import voc
from model.fasterRCNN import FasterRCNN

import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import MultiStepLR

import numpy as np
import random
from infer import evaluate_map

from config.config import config

def train():

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = FasterRCNN(device).to(device)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # optimizer = torch.optim.SGD(lr=0.001,
    #                             params=filter(lambda p: p.requires_grad,
    #                                           model.parameters()),
    #                             weight_decay=5E-4,
    #                             momentum=0.9)
    # scheduler = MultiStepLR(optimizer, milestones=[9,12], gamma=0.1)
    
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': 0.001 * 2, 'weight_decay': 0}]
            else:
                params += [{'params': [value], 'lr': 0.001, 'weight_decay': 0.0005}]
    optimizer = torch.optim.SGD(params, momentum=0.9)


    # save training result
    save_path = os.path.join(os.getcwd(), 'result/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    losses = []  # Track loss over time
    losses_rpns = []  # Track RPN loss over time
    losses_frcnns = []  # Track FRCNN loss over time

    # avg_losses = []
    # avg_losses_rpns = []
    # avg_losses_frcnns = []

    # load dataset
    train_dataset = voc.VOCDataset(split='trainval')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
        

    # acc_steps = 1
    # step_count = 1

    best_map = 0.0

    num_epochs = config.TRAIN_EPOCHS_NUM
    for epoch in range(num_epochs):
        print("\n======Epoch: ", epoch+1, "======")
        model.train()
        optimizer.zero_grad()
        train_bar = tqdm.tqdm(train_loader, file = sys.stdout, ncols=120)
        for one_batch in train_bar:

            # get a new batch, which is a single pair of image and target here
            img_id, img, target = one_batch

            # forward and get the loss
            img = img.float().to(device)
            target['bboxes'] = target['bboxes'].float().to(device)
            target['labels'] = target['labels'].long().to(device)
            rpn_output, frcnn_output = model(img, target, img_id)
            rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
            frcnn_loss = frcnn_output['frcnn_classification_loss'] + frcnn_output['frcnn_localization_loss']

            # print the four losses
            # print("RPN Classification Loss: ", rpn_output['rpn_classification_loss'])
            # print("RPN Localization Loss: ", rpn_output['rpn_localization_loss'])
            # print("FRCNN Classification Loss: ", frcnn_output['frcnn_classification_loss'])
            # print("FRCNN Localization Loss: ", frcnn_output['frcnn_localization_loss'])
            loss = rpn_loss + frcnn_loss

            losses.append(loss.item())
            losses_rpns.append(rpn_loss.item())
            losses_frcnns.append(frcnn_loss.item())

            # loss = loss / acc_steps
            loss.backward()
            # if step_count % acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            # step_count += 1
            
            train_bar.desc = "train epoch[{}/{}] rpn cls{:.3f} loc{:.3f} roi cls{:.3f} loc{:.3f}".format(epoch+1, num_epochs, rpn_output['rpn_classification_loss'], rpn_output['rpn_localization_loss'], frcnn_output['frcnn_classification_loss'], frcnn_output['frcnn_localization_loss'])

        # print the mean loss of this epoch
        # optimizer.step()
        # optimizer.zero_grad()
        # scheduler.step()
        loss_output = ''
        loss_output += 'Epoch: {}\n'.format(epoch+1)
        loss_output += 'Loss: {}\n'.format(sum(losses) / len(losses))
        loss_output += 'RPN Loss: {}\n'.format(sum(losses_rpns) / len(losses_rpns))
        loss_output += 'FRCNN Loss: {}\n'.format(sum(losses_frcnns) / len(losses_frcnns))
        print(loss_output)

        # avg_losses.append(sum(losses) / len(losses))
        # avg_losses_rpns.append(sum(losses_rpns) / len(losses_rpns))
        # avg_losses_frcnns.append(sum(losses_frcnns) / len(losses_frcnns))
        # losses = []
        # losses_rpns = []
        # losses_frcnns = []

        # compute mAP every 3 epochs
        # if epoch % 1 == 0:

        

        if epoch == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
            best_map = evaluate_map()
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            
        else:
            torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
            map = evaluate_map()

            if map > best_map:
                best_map = map
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))

        if epoch == 9:
            model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pth')))
            # optimizer = torch.optim.SGD(lr=0.0001,
            #                             params=filter(lambda p: p.requires_grad,
            #                                         model.parameters()),
            #                             weight_decay=5E-4,
            #                             momentum=0.9)
            params = []
            for key, value in dict(model.named_parameters()).items():
                if value.requires_grad:
                    if 'bias' in key:
                        params += [{'params': [value], 'lr': 0.0001 * 2, 'weight_decay': 0}]
                    else:
                        params += [{'params': [value], 'lr': 0.0001, 'weight_decay': 0.0005}]
            optimizer = torch.optim.SGD(params, momentum=0.9)
            
        if epoch == 14:
            break



    # print("LOSSES: ", avg_losses)
    # print("LOSSES_RPNS: ", avg_losses_rpns)
    # print("LOSSES_FRCNNS: ", avg_losses_frcnns)

    # save the model
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))


if __name__ == '__main__':
    train()








