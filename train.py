import torch
import os
import sys
import tqdm
from torch.utils.data import DataLoader

from dataset import voc
from model import fasterRCNN
import yaml

import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import MultiStepLR

def train():
    # Load the YAML configuration
    with open('config/voc.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = fasterRCNN.FasterRCNN().to(device)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    optimizer = torch.optim.SGD(lr=0.001,
                                params=filter(lambda p: p.requires_grad,
                                              model.parameters()),
                                weight_decay=5E-4,
                                momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[12,16], gamma=0.1)
    
    # save training result
    save_path = os.path.join(os.getcwd(), 'result/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    # Initialize real-time plot
    # plt.ion()  # Interactive mode on
    # fig, ax = plt.subplots()
    losses = []  # Track loss over time
    losses_rpns = []  # Track RPN loss over time
    losses_frcnns = []  # Track FRCNN loss over time


    # load dataset
    train_dataset = voc.VOCDataset(split='trainval')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
        
    num_epochs = config['train_param']['num_epochs']
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        train_bar = tqdm.tqdm(train_loader, file = sys.stdout, ncols=100)
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
            loss = rpn_loss + frcnn_loss

            losses.append(loss.item())
            losses_rpns.append(rpn_loss.item())
            losses_frcnns.append(frcnn_loss.item())

            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # id = int(id[0])
            # tqdm.tqdm.write(f"Processed ID: {id}")
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1, num_epochs, loss)

            # Update plot in real-time
            # ax.clear()
            # ax.plot(losses, label="Training Loss") 
            # ax.plot(losses_rpns, label="RPN Loss")
            # ax.plot(losses_frcnns, label="FRCNN Loss")
            # ax.set_xlabel("Iterations")
            # ax.set_ylabel("Loss")
            # ax.legend()
            # plt.draw()
            # plt.pause(0.01)  # Pause to allow the plot to update
        loss_output = ''
        loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
        loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
        loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
        loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
        print(loss_output)
    # Close interactive plotting mode
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    train()








