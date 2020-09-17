import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from utility import *
from torch.nn import Parameter
from torch.nn import NLLLoss
from numpy.random import uniform
from torch.distributions.uniform import Uniform
from torch.nn.utils.rnn import PackedSequence

class IntensityLoss(nn.Module):
    def __init__(self, hidden_dim, nsamples=10, eps = 1e-8):
        super(IntensityLoss, self).__init__()
        self.nsamples = nsamples
        self.intensity_out = nn.Linear(hidden_dim,1)
        self.nonlinearity = nn.Softplus()
        self.eps = eps


    def getIntensity(self, duration, states):
        #print(states[0].shape)
        (h_t, c_t) = get_ctlstm_hidden(states, duration)
        #print(h_t.shape)
        intensity = self.nonlinearity(self.intensity_out(h_t))
        return intensity


    def intensityUpperBound(self, states):
     (cx, cbarx, deltx, ox) = states
     weight = self.intensity_out.weight[0]
     #print(weight.shape)
     c_cbar_diff = (cx - cbarx)[0]
     #print(c_cbar_diff.shape)
     decreasing_index = ((((c_cbar_diff < 0) & (weight < 0)).float()) + 
                        (((c_cbar_diff > 0) & (weight > 0)).float()))
     c_t = cbarx + (c_cbar_diff * decreasing_index)
     h_t = ox * torch.tanh(c_t)
     
     intensity_upper_bound = self.nonlinearity(self.intensity_out(h_t))

     #print(intensity_upper_bound.reshape(-1).shape)
     #print(intensity_upper_bound.shape)
     return intensity_upper_bound.reshape(-1)


    def sample(self, states):
        #print('sampling')
        upper_bound = self.intensityUpperBound(states)
        #print(upper_bound.shape)
        predicted_times = []

        for i in range(len(upper_bound)):
            t = 0 
            while True:
                delta = (torch.empty(1, device=states[0].device).exponential_(upper_bound[i].item()))
                u = (torch.empty(1, device=states[0].device).uniform_())
                t = t + delta.item()
                #print(states[0][:, 0, :].shape)
                #print(self.getIntensity(t, (states[0][:, i, :], states[1][:, i, :], states[2][:, i, :], states[3][:, i, :])))
                if (u.item() * upper_bound[i].item()) <= (self.getIntensity(t, (states[0][:, i, :], states[1][:, i, :], states[2][:, i, :], states[3][:, i, :])).item()):
                    break
            
            predicted_times.append(t) 

        #print(predicted_times)
        return predicted_times
          


    def integrate(self, duration, states):
        #t_prev_np = t_prev.cpu().data.numpy()
        #t_next_np = t_next.cpu().data.numpy()
        sample_count = 0
        loss = torch.zeros(duration.shape).to(duration.device)

        uniform_sampler = Uniform(torch.zeros(duration.shape).to(duration.device), duration)
        weight = (duration / self.nsamples)
        while(sample_count < self.nsamples):
            t = (uniform_sampler.sample()).view(-1, 1)
            
            loss += (self.getIntensity(t, states)).reshape(-1) * weight  
            sample_count = sample_count + 1
        
        #loss = torch.sum(loss)
        #assert(not np.isnan(loss.item()))
        #print(loss.shape)
        return loss
    
    def forward(self, duration, states):

        nll_loss = -1.0 * torch.log(self.getIntensity(duration, states) + self.eps).reshape(-1)
        #print(nll_loss.shape)
        #nll_loss = torch.sum(nll_loss, 0)
        #print(nll_loss.shape)
        nll_loss = nll_loss + self.integrate(duration, states)
        return nll_loss




        
            
 
    
   
