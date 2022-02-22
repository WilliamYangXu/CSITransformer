import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import os
import torch.nn as nn
import random
from src.data import load_data
from src.image_segmentation import model_trainer

gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 42
seed_everything(SEED)

def train_all(bits):
    print("="*30)
    print("Encoder: CNN; Decoder: CNN")
    print("compressed codeword bits: {}".format(bits))
    agent3 = model_trainer(epochs=40, net="Csi_Net",feedbackbits=bits)
    x3, agent3_NMSE, t3 = agent3.model_train()
    print("Csi_Net")
    print(agent3_NMSE)
    print("average time used is:", t3)
    plt.plot(x3, agent3_NMSE, label="cnn")

    print("="*30)
    print("Encoder: transformer; Decoder: transformer")
    print("compressed codeword bits: {}".format(bits))
    agent1 = model_trainer(epochs=40, net="Csi_Transformer_Net",feedbackbits=bits)
    x1, agent1_NMSE, t1 = agent1.model_train()
    print("Csi_Transformer_Net")
    print(agent1_NMSE)
    print("average time used is:", t1)
    plt.plot(x1, agent1_NMSE, label="Csi_Transformer_Net")
    
    print("="*30)
    print("Encoder: CNN; Decoder: transformer")
    print("compressed codeword bits: {}".format(bits))
    agent2 = model_trainer(epochs=40, net="Csi_CNN_Transformer_Net",feedbackbits=bits)
    x2, agent2_NMSE, t2 = agent2.model_train()
    print("Csi_CNN_Transformer_Net")
    print(agent2_NMSE)
    print("average time used is:", t2)
    plt.plot(x2, agent2_NMSE, label="Csi_CNN_Transformer_Net")

    print("="*30)
    print("Encoder: Random Projection; Decoder: CNN")
    print("compressed codeword bits: {}".format(bits))
    agent4 = model_trainer(epochs=40, net="CS_Net",feedbackbits=bits)
    x4, agent4_NMSE= agent4.model_train()
    print("CS_Net")
    print(agent4_NMSE)
    plt.plot(x4, agent4_NMSE, label="CS_Net")

    print(x2)
    plt.show()

    plt.plot(x1, agent1_NMSE, label="Csi_Transformer_Net")
    plt.plot(x2, agent2_NMSE, label="Csi_CNN_Transformer_Net")
    plt.plot(x3, agent3_NMSE, label="CNN")
    plt.plot(x4, agent4_NMSE, label="Random Projection")
    plt.xlabel("Number of Epochs")
    plt.ylabel("NMSE")
    plt.show()

# train_all(256)
train_all(128)
# train_all(64)
# train_all(32)
