import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from utility import *
from torch.nn import Parameter
from torch.nn import NLLLoss
from scaled_softplus import * 
from numpy.random import uniform
from torch.distributions.uniform import Uniform
from torch.nn.utils.rnn import PackedSequence

class IntensityLoss(nn.Module):
    def __init__(self, intensity_criterion, nsamples=10):
        super(MonteCarloIntegral, self).__init__()
        self.intensity_criterion = intensity_criterion
        self.nsamples = nsamples

    def integrate(self, duration, states):
        #t_prev_np = t_prev.cpu().data.numpy()
        #t_next_np = t_next.cpu().data.numpy()
        sample_count = 0
        loss = torch.zeros(duration.shape).to(duration.device)
        uniform_sampler = Uniform(torch.zeros(duration.shape).to(duration.device), duration)
        while(sample_count < self.nsamples):
            t = (uniform_sampler.sample()).view(-1, 1)
            weight = (duration / self.nsamples)
            loss += (self.intensity_criterion.getIntensity(states, duration)) * weight  
            sample_count = sample_count + 1
        
        loss = torch.sum(loss)
        assert(not np.isnan(loss.item()))
        return loss
    
    def forward(self):
        
            
 
    
   