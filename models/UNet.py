import torch
import torch.nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self,n_classes):
        self.conv1 = nn.Conv2d(1,64,kernel_size )

    def forward(self, x):
        pass
