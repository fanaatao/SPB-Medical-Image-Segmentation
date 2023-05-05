import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvRNNCell(nn.Module):
    def __init__(self, in_channels, shape, num_filter, kernel_size=3):
        super(ConvRNNCell, self).__init__()
        padding = (kernel_size - 1) // 2
        self._shape = shape
        self._num_filter = num_filter
        self.conv_1 = nn.Conv2d(in_channels + num_filter, num_filter, kernel_size=1, padding=0, bias=True)
        self.conv_2 = nn.Conv2d(in_channels + num_filter, num_filter, kernel_size=kernel_size, padding=padding,bias=True)
	
    def forward(self, x, hidden):
        """forward process of ConvRNNCell"""
        combind = torch.cat((x, hidden), 1)
        reset_gate = F.sigmoid(self.conv_1(combind))
        
        reset_hidden = reset_gate * hidden
        
        combin_reset = torch.cat((x, reset_hidden), dim=1)
        new_hidden = F.relu(self.conv_2(combin_r