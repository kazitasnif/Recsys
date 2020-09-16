import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ctlstm_cell import * 
from torch.nn.utils.rnn import PackedSequence
from utility import *
#


class CTLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(CTLSTM, self).__init__()
        self.input_dim = input_size
        self.hidden_size = hidden_size
        #self.device = device
        self.lstm = ContinuousLSTMCell(input_size, hidden_size)
        #self.reset_parameters()

    def reset_parameters(self, output_only = False):
        self.lstm.reset_parameters()




    def forward(self, input_, states, duration):
            if(states is None):
                states = self.init_hidden(input_.size(0), input_.device)
            steps = range(duration.size(1))
            output = []
            input_ = input_.permute(1, 0, 2)
  
            for i in steps:
                (cx, cbarx, deltx, ox) = states
                (h_t, c_t) = get_ctlstm_hidden(states, duration[:, i])
                #output.append(h_t)
                states = self.lstm(input_[i], (h_t, c_t, cbarx))
                #(h_t, c_t) = get_ctlstm_hidden(states, torch.zeros(duration[:, i].shape))
                output.append(states)
           
            output = tuple(torch.cat(o, 0).permute(1, 0, 2)
                           for o in zip(*output))

            #print(output.shape)
            return output

    def init_hidden(self, batch_size, device='cpu'):
        
        #not stacked at this point
        return (torch.zeros(1, batch_size, self.hidden_size, dtype=torch.float,
                           device=device), 
                torch.zeros(1, batch_size, self.hidden_size, dtype=torch.float,
                           device=device), 
                torch.zeros(1, batch_size, self.hidden_size, dtype=torch.float,
                           device=device), 
                torch.zeros(1, batch_size, self.hidden_size, dtype=torch.float,
                           device=device))
     


 




        
        
        
        
