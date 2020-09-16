import torch

def get_ctlstm_hidden(states, time_diff):
    (cx, cbarx, deltx, ox) = states
    c_t = cbarx + ((cx - cbarx) * torch.exp(-1.0 * deltx * time_diff))
    h_t = ox * torch.tanh(c_t)
    return (h_t, c_t)