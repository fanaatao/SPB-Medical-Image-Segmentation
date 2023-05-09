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
        new_hidden = F.relu(self.conv_2(combin_reset))  # TODO: applying tanh maybe a better choice
        return new_hidden

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self._num_filter, self._shape[0], self._shape[1])).cuda(0)



class ConvRNN(nn.Module):
    def __init__(self, in_channels, shape, num_filter, kernel_size=3, num_layer=1, bidirectional=True):
        super(ConvRNN, self).__init__()
        self._in_channels = in_channels
        self._shape = shape
        self._num_filter = num_filter
        self._kernel_size = kernel_size
        self._num_layer = num_layer
        self._bidirectional = bidirectional
        self._padding = (self._kernel_size - 1) // 2
        self._cell_list = None
        self._forward_cell_list = None
        self._backward_cell_list = None

        if self._bidirectional:
            forward_cell_list = []
            backward_cell_list = []

            for idx in range(self._num_layer):
                forward_cell_list.append(
                    ConvRNNCell(self._in_channels, self._shape, self._num_filter, self._kernel_size))
                backward_cell_list.append(
                    ConvRNNCell(self._in_channels, self._shape, self._num_filter, self._kernel_size))
            self._forward_cell_list = nn.ModuleList(forward_cell_list)
            self._backward_cell_list = nn.ModuleList(backward_cell_list)
        else:
            cell_list = []
            for idx in range(self._num_layer):
                cell_list.append(ConvRNNCell(self._in_channels, self._shape, self._num_filter, self._kernel_size))
            self._cell_list = nn.ModuleList(cell_list)

    def forward(self, x, hidden_state):
        curr_forward_x = x.transpose(0, 1)
        curr_backward_x = x.transpose(0, 1)

        forward_hidden_state, backward_hidden_state = hidden_state

        seq_len = curr_forward_x.size(0)

        if self._b