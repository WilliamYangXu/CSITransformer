#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse

def NMSE_cuda(x, x_hat):
    x_real = x[:, 0, :, :].view(len(x), -1) - 0.5
    x_imag = x[:, 1, :, :].view(len(x), -1) - 0.5
    x_hat_real = x_hat[:, 0, :, :].contiguous().view(len(x_hat), -1) - 0.5
    x_hat_imag = x_hat[:, 1, :, :].contiguous().view(len(x_hat), -1) - 0.5
    power = torch.sum(x_real ** 2 + x_imag ** 2, axis=1)
    mse = torch.sum((x_real - x_hat_real) ** 2 + (x_imag - x_hat_imag) ** 2, axis=1)
    nmse = mse / power
    return nmse

class NMSELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x, x_hat)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse)
        else:
            nmse = torch.sum(nmse)
        return nmse

def rho(x, x_hat):
    x_real = x[:, 0, :, :].view(len(x), -1) - 0.5
    x_imag = x[:, 1, :, :].view(len(x), -1) - 0.5
    x_hat_real = x_hat[:, 0, :, :].contiguous().view(len(x_hat), -1) - 0.5
    x_hat_imag = x_hat[:, 1, :, :].contiguous().view(len(x_hat), -1) - 0.5

    cos = nn.CosineSimilarity(dim=1, eps=0)
    out_real = cos(x_real,x_hat_real)
    out_imag = cos(x_imag, x_hat_imag)
    result_real = out_real.sum() / len(out_real)
    reault_imag = out_imag.sum() / len(out_imag)
    return 0.5*(result_real + reault_imag)

class CosSimilarity(nn.Module):
    def __init__(self, reduction='sum'):
        super(CosSimilarity, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        cos = rho(x, x_hat)
        # if self.reduction == 'mean':
        #     cos = torch.mean(cos)
        # else:
        #     cos = torch.sum(cos)
        return cos

