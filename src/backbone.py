#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes)),
            ('LeakyReLU', nn.LeakyReLU(negative_slope=0.3, inplace=False))
        ]))

class ConvBN_linear(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN_linear, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))

class ResBlock(nn.Module):

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(ConvBN(ch, 8, 3))
            resblock_one.append(ConvBN(8, 16, 3))
            resblock_one.append(ConvBN_linear(16, ch, 3))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x

class PatchEmbed(nn.Module):

    def __init__(self, H=16, W=32, patch_size=4, in_chans=2, embed_dim=32):
        super().__init__()
        num_patches = H * W / patch_size ** 2
        self.img_size = [H, W]
        self.patch_size = [patch_size, patch_size]
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class FixedPositionalEncoding(nn.Module):
    
    def __init__(self, embedding_dim, max_length=5000):
        super(FixedPositionalEncoding, self).__init__()
        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x

class TransformerEncoder(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, feedforward_dim):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear_1 = torch.nn.Linear(embed_dim, feedforward_dim)
        self.linear_2 = torch.nn.Linear(feedforward_dim, embed_dim)
        self.layernorm_1 = torch.nn.LayerNorm(embed_dim)
        self.layernorm_2 = torch.nn.LayerNorm(embed_dim)

    def forward(self, x_in):
        attn_out, _ = self.attn(x_in, x_in, x_in)
        x = self.layernorm_1(x_in + attn_out)
        ff_out = self.linear_2(torch.nn.functional.relu(self.linear_1(x)))
        x = self.layernorm_2(x + ff_out)
        return x

class Csi_Encoder(nn.Module):

    def __init__(self, feedback_bits):
        super(Csi_Encoder, self).__init__()

        self.convban = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN_linear(1, 2, 1)),
        ]))
        self.fc = nn.Linear(2048, int(feedback_bits))

    def forward(self, x_in):
        x_in = x_in.view(32,1,32,32)
        out = self.convban(x_in)
        out = out.view(32,-1)
        out = self.fc(out)
        return out

class Csi_Decoder(nn.Module):

    def __init__(self, feedback_bits):
        super(Csi_Decoder, self).__init__()

        self.feedback_bits = feedback_bits
        self.fc = nn.Linear(int(feedback_bits), 1024)
        decoder = OrderedDict([
            ("decoder1",ResBlock(1)),
            ('LeakyReLU', nn.LeakyReLU(negative_slope=0.3, inplace=False)),
            ("decoder2",ResBlock(1))
        ])
        self.decoder_feature = nn.Sequential(decoder)
        self.out_cov = ConvBN_linear(1,1,3)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = x
        out = self.fc(out)
        out = out.view(32, 1, -1, 32)
        out = self.decoder_feature(out)
        out = self.out_cov(out)
        out = self.sig(out)
        out = out.view(32,2,16,32)
        return out

class Csi_Attention_Encoder(nn.Module):

    def __init__(self, feedback_bits):
        super(Csi_Attention_Encoder, self).__init__()
        
        # with positional encoding
        # self.patch_embedding = nn.Sequential(OrderedDict([
        #     ("patch_embedding", PatchEmbed(H=16, W=32, patch_size=4, in_chans=2, embed_dim=32))
        # ]))
        # self.positional_encoding = nn.Sequential(OrderedDict([
        #     ("positional_encoding", FixedPositionalEncoding(32,32))
        # ]))
        # self.transformer_layer =  nn.Sequential(OrderedDict([
        #     ("transformer_encoder1", TransformerEncoder(32,8,0,512)) # after [32, 512, 32]
        # ])) # for added positional encoding
        
        # without positional encoding
        self.conv_layer = ConvBN_linear(1,2,1)
        self.transformer_layer = nn.Sequential(OrderedDict([

                ("transformer_encoder1", TransformerEncoder(64,8,0,512))
            ])) # without positional encoding
        self.fc = nn.Linear(2048, int(feedback_bits))

    def forward(self, x_in):

        # with pos encoding
        ##x_in = self.patch_embedding(x_in)
        ##x_in = self.positional_encoding(x_in)
        # without pos encoding
        x_in = x_in.view(32,1,32,32)
        x_in = self.conv_layer(x_in)

        x_in = x_in.view(32,32,64)
        out = self.transformer_layer(x_in)
        #out = out.contiguous().view(32,-1) with pos encoding
        out = out.contiguous().view(-1, 2048) # without pos encoding
        out = self.fc(out)
        return out

class Csi_Attention_Decoder(nn.Module):

    def __init__(self, feedback_bits):
        super(Csi_Attention_Decoder, self).__init__()

        self.feedback_bits = feedback_bits
        self.fc = nn.Linear(int(feedback_bits), 2048)
        decoder = OrderedDict([
            ("transformer_decoder1",TransformerEncoder(64,8,0,feedforward_dim=128))
        ])
        self.decoder_feature = nn.Sequential(decoder)
        self.conv_linear = ConvBN_linear(2,1,1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = x
        out = self.fc(out) 
        out = out.view(32, -1, 64)
        out = self.decoder_feature(out)
        out = out.view([32,2,32,32])
        out = self.conv_linear(out)
        out = self.sig(out)
        out = out.view(32,2,16,32)
        return out

class Csi_Net(nn.Module):

    def __init__(self, feedback_bits):
        super(Csi_Net, self).__init__()
        self.encoder = Csi_Encoder(feedback_bits)
        self.decoder = Csi_Decoder(feedback_bits)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out

class Csi_Transformer_Net(nn.Module):

    def __init__(self, feedback_bits):
        super(Csi_Transformer_Net, self).__init__()
        self.encoder = Csi_Attention_Encoder(feedback_bits)
        self.decoder = Csi_Attention_Decoder(feedback_bits)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out

class CS_Net(nn.Module):

    def __init__(self, feedback_bits):
        super(CS_Net, self).__init__()
        self.A = np.random.uniform(low=-0.5, high=0.5, size=(1024, feedback_bits))
        self.A = torch.from_numpy(self.A)
        self.A = self.A.float().cuda()
        self.decoder = Csi_Decoder(feedback_bits)

    def forward(self, x):
        
        x = x.view(32, -1)
        out = x @ self.A
        out = out.cuda()
        out = self.decoder(out)
        return out

class Csi_CNN_Transformer_Net(nn.Module):

    def __init__(self, feedback_bits):
        super(Csi_CNN_Transformer_Net, self).__init__()
        self.encoder = Csi_Encoder(feedback_bits)
        self.decoder = Csi_Attention_Decoder(feedback_bits)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out

