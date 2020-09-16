import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Parameter
#from utility import *
#from scaled_softplus import *
# Creating the Network

class ContinuousLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ContinuousLSTMCell, self).__init__()
        self.linear = nn.Linear(input_dim, 7 * hidden_dim)
        self.linear_hidden = nn.Linear(hidden_dim, 7 * hidden_dim)
        #self.scaled_softplus = ScaledSoftplus(1)
        self.nonlinearity = nn.Softplus()

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.linear_hidden.reset_parameters()
        #self.nonlinearity.reset_parameters()

    def copy_params(self, weights):
        with torch.no_grad():
            self.linear.weight.copy_(weights['w_ih'])
            self.linear_hidden.weight.copy_(weights['w_hh'])

    def forward(self, input_, hidden):
        #cbarx base rate #ox output gate #deltx decay
        (h_t, c_t, cbarx) = hidden
        gates = self.linear(input_) + self.linear_hidden(h_t)


                
        i, f, c, o, ibar, fbar, decay = gates.chunk(7, gates.dim() - 1) 
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        c = torch.tanh(c)
        o = torch.sigmoid(o)
        ibar = torch.sigmoid(ibar)
        fbar = torch.sigmoid(fbar)


        _c = (f * c_t) + (i * c)
        _cbar = (fbar * cbarx) + (ibar * c)
        _delt = self.nonlinearity(decay)
        
        return (_c, _cbar, _delt, o)

    
