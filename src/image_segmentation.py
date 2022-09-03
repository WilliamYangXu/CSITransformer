#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib as plt
import torch
import os
import torch.nn as nn
import random
import time

from src.loss import NMSE_cuda, NMSELoss, CosSimilarity, rho
from src.backbone import Csi_Net, CS_Net, Csi_Transformer_Net, Csi_CNN_Transformer_Net
from src.data import load_data
import matplotlib.pyplot as plt

gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def channel_visualization(image):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest', origin='upper')
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.show()

SEED = 42
print("seeding everything...")
seed_everything(SEED)
print("initializing parameters...")


class model_trainer():

    def __init__(self,
                 epochs,
                 net,
                 feedbackbits=128,
                 batch_size=32,
                 learning_rate=1e-3,
                 lr_decay_freq=30,
                 lr_decay=0.1,
                 best_loss=100,
                 num_workers=0,
                 print_freq=100,
                 train_test_ratio=0.8):

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_decay_freq = lr_decay_freq
        self.lr_decay = lr_decay
        self.best_loss = best_loss
        self.num_workers = num_workers
        self.print_freq = print_freq
        self.train_test_ratio = train_test_ratio
        # parameters for data
        self.feedback_bits = feedbackbits
        self.img_height = 16
        self.img_width = 32
        self.img_channels = 2

        self.model = eval(net)(self.feedback_bits)
        self.x_label = []
        self.y_label = []
        self.ys_label = []
        self.t_label = []

        if len(gpu_list.split(',')) > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()  # model.module
        else:
            self.model = self.model.cuda()

        self.criterion = NMSELoss(reduction='mean')  # nn.MSELoss()
        self.criterion_test = NMSELoss(reduction='sum')
        # self.criterion_rho = CosSimilarity(reduction='mean')
        # self.criterion_test_rho = CosSimilarity(reduction='sum')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # train_loader, test_loader, train_dataset, test_dataset, \
        # train_shuffle_loader, test_shuffle_loader, train_shuffle_dataset, test_shuffle_dataset

        self.train_loader, self.test_loader, self.train_dataset,        self.test_dataset, self.train_shuffle_loader, self.test_shuffle_loader,        self.train_shuffle_dataset, self.test_shuffle_dataset =             load_data('/filepath',shuffle = True)

    def model_save(self,encoderPATH, decoderPATH):
        print('Saving model...')

        try:
            torch.save({'state_dict': self.model.encoder.state_dict(), }, '/filepath')
        except:
            torch.save({'state_dict': self.model.module.encoder.state_dict(), }, '/filepath')

        try:
            torch.save({'state_dict': self.model.decoder.state_dict(), }, '/filepath')
        except:
            torch.save({'state_dict': self.model.module.decoder.state_dict(), }, '/filepath')
#         print('Model saved!')
        self.best_loss = self.average_loss

    def model_train(self):

        for epoch in range(self.epochs):
            print('========================')
            print('lr:%.4e' % self.optimizer.param_groups[0]['lr'])
            # train model
            self.model.train()
   
            # decay lr
            if epoch % self.lr_decay_freq == 0 and epoch > 0:
                self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] * self.lr_decay

            # training...
            for i, input in enumerate(self.train_loader):
                input = input.cuda()  # input [batch=32,2,16,32]
                output = self.model(input)
                loss = self.criterion(output, input)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if i % self.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Loss {loss:.4f}\t'.format(
                        epoch, i, len(self.train_loader), loss=loss.item()))
            self.model.eval()

            # evaluating...
            self.total_loss = 0
            self.total_rho = 0
            start = time.time()
            with torch.no_grad():

                for i, input in enumerate(self.test_loader):
                    
                    input = input.cuda()
                    output = self.model(input)
                    self.total_loss += self.criterion_test(output, input).item()
                    # self.total_rho += self.criterion_rho(output,input).item()
                    #print(rho(output,input), type(rho(output,input)))
                    self.total_rho += (rho(output,input))
                    
                end = time.time()
                t = end - start
                self.average_loss = self.total_loss / len(self.test_dataset)
                self.average_rho = self.total_rho / len(list(enumerate(self.test_loader)))
                self.x_label.append(epoch)
                self.y_label.append(self.average_loss)
                self.t_label.append(t)
                print('NMSE %.4f ρ %.3f time %.3f' % (self.average_loss,self.average_rho, t))

        for i, input in enumerate(self.test_loader): # visualize one sample
            if i == 3: # set shuffle = False to ensure the same sample each time
                ones = torch.ones(32,32)
                image1 = input[0].view(32,32)
                image1 = ones - image1
                image1 = image1.numpy()
                channel_visualization(image1)
                input = input.cuda()
                output = self.model(input)
                output = output.cpu()
                image2 = output[0].view(32,32)
                image2 = ones - image2
                image2 = image2.detach().numpy()
                channel_visualization(image2)

        return self.x_label, self.y_label, sum(self.t_label)/len(self.t_label) # , self.ys_label

