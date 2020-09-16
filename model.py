import torch
import torch.nn as nn
import torch.nn.functional as F

#from torch.autograd import 

from modules import Embed, Intra_RNN, Inter_RNN, Time_Loss
from intensity_loss import IntensityLoss
from utility import *

class RecommenderModel:

    def __init__(self, dims, params, flags, datahandler, tester, time_threshold, device='cpu', ctlstm=False):
        self.dims = dims
        self.params = params
        self.flags = flags
        self.datahandler = datahandler
        self.tester = tester
        self.time_threshold = time_threshold
        self.device = device
        self.ctlstm = ctlstm
        self.init_model()

    def init_model(self):
        #initialize lists to contain the parameters in two sub-nets
        inter_intra_params = []
        time_params = []

        #setting up embedding matrices
        self.item_embed = Embed(self.dims["N_ITEMS"], self.dims["EMBEDDING_DIM"], item=True)
        if(self.device.startswith("cuda")):
            self.item_embed = self.item_embed.cuda()
        inter_intra_params += list(self.item_embed.parameters())

        if(self.flags["context"]):
            self.time_embed = Embed(self.dims["TIME_RESOLUTION"], self.dims["TIME_HIDDEN"], item=False)
            if(self.device.startswith("cuda")):
                self.time_embed = self.time_embed.cuda()
            inter_intra_params += list(self.time_embed.parameters())
            self.user_embed = Embed(self.dims["N_USERS"], self.dims["USER_HIDDEN"], item=False)
            if(self.device.startswith("cuda")):
                self.user_embed = self.user_embed.cuda()
            inter_intra_params += list(self.user_embed.parameters())

        #setting up models with optimizers
        self.inter_rnn = Inter_RNN(self.dims["INTER_INPUT_DIM"], self.dims["INTER_HIDDEN"], self.params["dropout"], device=self.device, ctlstm=self.ctlstm)
        if(self.device.startswith("cuda")):
            self.inter_rnn = self.inter_rnn.cuda()
        inter_intra_params += list(self.inter_rnn.parameters())

        self.intra_rnn = Intra_RNN(self.dims["EMBEDDING_DIM"], self.dims["INTRA_HIDDEN"], self.dims["N_ITEMS"], self.params["dropout"])
        if(self.device.startswith("cuda")):
            self.intra_rnn = self.intra_rnn.cuda()
        inter_intra_params += list(self.intra_rnn.parameters())

        #setting up linear layers for the time loss, first recommendation loss and inter RNN
        if(self.flags["temporal"]):
            if(not self.ctlstm):
                self.time_linear = nn.Linear(self.dims["INTER_HIDDEN"],1)
                if(self.device.startswith("cuda")):
                    self.time_linear = self.time_linear.cuda()
                time_params += [{"params": self.time_linear.parameters(), "lr":0.1*self.params["lr"]}]

            self.first_linear = nn.Linear(self.dims["INTER_HIDDEN"],self.dims["N_ITEMS"])
            if(self.device.startswith("cuda")):
                self.first_linear = self.first_linear.cuda()
        """
        self.intra_linear = nn.Linear(self.dims["INTER_HIDDEN"],self.dims["INTRA_HIDDEN"])
        self.intra_linear = self.intra_linear.cuda()
        inter_intra_params += list(self.intra_linear.parameters())
        """
        #setting up time loss model
        if(self.flags["temporal"]):
            self.time_loss_func = Time_Loss()

            if(self.ctlstm):
                self.intensity_loss = IntensityLoss(self.dims["INTER_HIDDEN"])
            else:
                self.intensity_loss = None
            if(self.device.startswith("cuda")):
                self.time_loss_func = self.time_loss_func.cuda()
                self.intensity_loss = self.intensity_loss.cuda()
            time_params += [{"params": self.time_loss_func.parameters() if not self.ctlstm else self.intensity_loss.parameters(), "lr": 0.1*self.params["lr"]}]

        #setting up optimizers
        self.inter_intra_optimizer = torch.optim.Adam(inter_intra_params, lr=self.params["lr"])
        if(self.flags["temporal"]):
            self.time_optimizer = torch.optim.Adam(time_params, lr=self.params["lr"])
            self.first_rec_optimizer = torch.optim.Adam(self.first_linear.parameters(), lr=self.params["lr"])

    #CUSTOM CROSS ENTROPY LOSS(Replace as soon as pytorch has implemented an option for non-summed losses)
    #https://github.com/pytorch/pytorch/issues/264
    def masked_cross_entropy_loss(self, y_hat, y):
        logp = -F.log_softmax(y_hat, dim=1)
        logpy = torch.gather(logp,1,y.view(-1,1))
        #mask = torch.tensor(y.data.float().sign().view(-1,1))
        mask = y.clone().detach().data.float().sign().view(-1, 1)
        logpy = logpy*mask
        return logpy.view(-1)

    def get_w(self):
        return self.time_loss_func.get_w()

    #step function implementing the equantion:
    #exp(time+w*t + exp(time)-exp(time+w*t)/w) = exp_time_w*exp((exp_time-exp_time_w)/w)
    @staticmethod
    def step_val(t, time_exp, w, dt): 
        time_w_exp = time_exp*torch.exp(t*w)
        exp_2 = torch.exp((time_exp-time_w_exp)/w)
        prob = time_w_exp*exp_2
        return t*prob

    #simpson numerical integration with higher resolution in the first 100 hours
    def time_prediction(self, time, w):
        #integration settings
        #integration_count += 1
        precision = 3000
        T = 700 #time units
        part1 = 100
        part2 = 600
        if(self.flags["use_day"]):
            T = T/24
            part1 = part1/24
            part2 = part2/24

        #moving data structures to the  for efficiency
        T = torch.tensor([T], dtype = torch.float, device=self.device)
        dt1 = torch.tensor([part1/precision], dtype=torch.float, device=self.device)
        dt2 = torch.tensor([part2/precision], dtype=torch.float, device=self.device)
        part1 = torch.tensor([part1], dtype=torch.float, device=self.device)
        
        #integration loops
        time_exp = torch.exp(time)
        time_preds1 = self.step_val(part1,time_exp, w, dt1)
        time_preds2 = self.step_val(T,time_exp, w, dt2) + time_preds1
        for i in range(1,precision//2):#high resolution loop
            t = (2*i-1)*dt1
            time_preds1 += 4*self.step_val(t,time_exp, w, dt1)
            time_preds1 += 2*self.step_val(t+dt1,time_exp, w, dt1)
        time_preds1 += 4*self.step_val(part1-dt1,time_exp, w, dt1)
        for i in range(1,precision//2):#rough resolution loop
            t = (2*i-1)*dt2 + part1
            time_preds2 += 4*self.step_val(t,time_exp, w, dt2)
            time_preds2 += 2*self.step_val(t+dt2,time_exp, w, dt2)
        time_preds2 += 4*self.step_val(T-dt2,time_exp,w,dt2)

        #division moved to the end for efficiency
        time_preds1 *= dt1/3
        time_preds2 *= dt2/3

        return time_preds1+time_preds2

        #scedule updater
    def update_loss_settings(self, epoch_nr):
        if(not self.flags["temporal"]):
            return
        else:
            if(epoch_nr == 0):
                self.params["ALPHA"] = 1.0
                self.params["BETA"] = 0.0
                self.params["GAMMA"] = 0.0
            if(epoch_nr == 4):
                self.params["ALPHA"] = 0.0
                self.params["BETA"] = 1.0
            if(epoch_nr == 8):
                self.params["BETA"] = 0.0
                self.params["GAMMA"] = 1.0
            if(epoch_nr == 10):
                self.params["ALPHA"] = 0.5
                self.params["GAMMA"] = 0.5
            if(epoch_nr == 11):
                self.params["BETA"] = 0.5
                self.params["GAMMA"] = 0.0
            if(epoch_nr == 12):
                self.params["ALPHA"] = 0.45
                self.params["BETA"] = 0.45
                self.params["GAMMA"] = 0.1
            if(self.flags["freeze"]):
                if(epoch_nr == 21):
                    self.flags["train_all"] = False
                    self.flags["train_first"] = False
                    self.params["ALPHA"] = 1.0
                    self.params["BETA"] = 0.0
                    self.params["GAMMA"] = 0.0
                if(epoch_nr == 24):
                    self.flags["train_first"] = True
                    self.flags["train_time"] = False
                    self.params["ALPHA"] = 0.0
                    self.params["GAMMA"] = 1.0
        return
    def predict_from_intensity(self):
        pass
    def train_mode(self):
        self.intra_rnn.train()
        self.inter_rnn.train()
        return

    def eval_mode(self):
        self.intra_rnn.eval()
        self.inter_rnn.eval()
        return

    #move batch data to cuda tensors
    def process_batch_inputs(self, items, session_reps, sess_time_reps, user_list, session_durations):
        sessions = (torch.tensor(session_reps, dtype=torch.float, device=self.device))
        #print(sessions.type())
        items = (torch.tensor(items, dtype=torch.long, device=self.device))
        sess_gaps = (torch.tensor(sess_time_reps, dtype=torch.long, device=self.device))
        session_durations = (torch.tensor(session_durations, dtype=torch.float, device=self.device))
        users = (torch.tensor(user_list.tolist(), dtype=torch.long, device=self.device))
  
        return items, sessions, sess_gaps, users, session_durations

    def process_batch_targets(self, item_targets, time_targets, first_rec_targets):
        item_targets = (torch.tensor(item_targets, dtype=torch.long, device=self.device))
        time_targets = (torch.tensor(time_targets, dtype=torch.torch.float, device=self.device)) 
        first = (torch.tensor(first_rec_targets, dtype=torch.long, device=self.device))
        return item_targets, time_targets, first

    def train_on_batch(self, items, session_reps, sess_time_reps, user_list, item_targets, time_targets, first_rec_targets, session_lengths, session_rep_lengths, session_durations):
        #print(self.time_threshold)
        #zero gradients before each epoch
        self.inter_intra_optimizer.zero_grad()
        if(self.flags["temporal"]):
            self.time_optimizer.zero_grad()
            self.first_rec_optimizer.zero_grad()

        #get batch from datahandler and turn into s
        X, S, S_gaps, U, S_durations = self.process_batch_inputs(items, session_reps, sess_time_reps, user_list, session_durations)
        #print(S_durations)
        #print(S_gaps)

        #print(S_gaps)

        #print(S.type())
        Y, T_targets, First_targets = self.process_batch_targets(item_targets, time_targets, first_rec_targets)
        #print(Y.shape)
        #print(T_targets)
        #print(First_targets.shape)

        if(self.flags["context"]):
            #get embedded times
            embedded_S_gaps = self.time_embed(S_gaps)

            #get embedded user
            embedded_U = self.user_embed(U)
            embedded_U = embedded_U.unsqueeze(1)
            embedded_U = embedded_U.expand(embedded_U.size(0), embedded_S_gaps.size(1), embedded_U.size(2))
        #print(session_lengths)
        #print(session_rep_lengths)

        #get the index of the last session representation of each user by subtracting 1 from each lengths, move to  for efficiency
        rep_indicies = (torch.tensor(session_rep_lengths, dtype=torch.long, device=self.device)) - 1
        #print(rep_indicies)
        #get initial hidden state of inter gru layer and call forward on the module
        inter_hidden = self.inter_rnn.init_hidden(S.size(0))
        #print(inter_hidden.shape)
        if(self.flags["context"]):
            #print(S.type())
            #print(embedded_S_gaps.type())
            #print(embedded_U.type())
            #print(S)
            #print(embedded_S_gaps)
            if(not self.ctlstm):
                inter_last_hidden = self.inter_rnn(torch.cat((S, embedded_S_gaps, embedded_U),2), inter_hidden, rep_indicies, S_durations)
                #print(inter_last_hidden.shape)
            else:
                inter_last_states = self.inter_rnn(torch.cat((S, embedded_S_gaps, embedded_U),2), inter_hidden, rep_indicies, S_durations)
                inter_last_hidden = get_ctlstm_hidden(inter_last_states, T_targets)[0]
                #print(inter_last_hidden.shape)



            #print(inter_last_hidden)
        else:
            if(not self.ctlstm):
                inter_last_hidden = self.inter_rnn(S, inter_hidden, rep_indicies, S_durations)
            else:
                inter_last_states = self.inter_rnn(S, inter_hidden, rep_indicies, S_durations)
                inter_last_hidden = get_ctlstm_hidden(inter_last_states, T_targets)[0]



        #if(self.ctlstm):

           # self.intensity_loss.intensityUpperBound(inter_last_states)
        #    time_predictions = self.intensity_loss.sample(inter_last_states)
        #    print(len(time_predictions))
        #get time scores and first prediction scores from the last hidden state of the inter RNN
        if(self.flags["temporal"]):
            if(not self.ctlstm):
                times = self.time_linear(inter_last_hidden).squeeze()
            #print(times.shape)
            first_predictions = self.first_linear(inter_last_hidden).squeeze()

        #get item embeddings
        embedded_X = self.item_embed(X)
        #print(embedded_X.shape)

        #create average pooling session representation using the item embeddings and the lenght of each sequence
        lengths = (torch.tensor(session_lengths, dtype=torch.float, device=self.device).view(-1,1)) #reshape the lengths in order to broadcast and use it for division
        
        sum_X = embedded_X.sum(1)
        mean_X = sum_X.div(lengths)
        #print(mean_X.data.shape)
        #print(mean_X.shape)

        #subtract 1 from the lengths to get the index of the last item in each sequence
        lengths = lengths.long()-1

        #call forward on the inter RNN
        recommendation_output, hidden_out = self.intra_rnn(embedded_X, inter_last_hidden, lengths)

        #store the new session representation based on the current scheme
        if(self.flags["use_hidden"]):
            self.datahandler.store_user_session_representations(hidden_out.data[0], user_list, time_targets)
        else:
            self.datahandler.store_user_session_representations(mean_X.data, user_list, time_targets)

        # LOSSES
        #prepare tensors for recommendation loss evaluation
        reshaped_Y = Y.view(-1)
        reshaped_rec_output = recommendation_output.view(-1,self.dims["N_ITEMS"]) #[SEQLEN*BATCHSIZE,N_items]

        #calculate recommendation losses
        reshaped_rec_loss = self.masked_cross_entropy_loss(reshaped_rec_output, reshaped_Y)
        #get mean losses based on actual number of valid events in batch
        sum_loss = reshaped_rec_loss.sum(0)
        divident = (torch.tensor([sum(session_lengths)], dtype=torch.float, device=self.device))
        mean_loss = sum_loss/divident

        if(self.flags["temporal"]):
            first_loss = self.masked_cross_entropy_loss(first_predictions, First_targets)
            sum_first_loss = first_loss.sum(0)
            mean_first_loss = sum_first_loss/embedded_X.size(0)


            #calculate the time loss
            #print(T_targets.shape)
            #print(T_targets[0])
            if(not self.ctlstm):
                time_loss = self.time_loss_func(times, T_targets, self.params["EPSILON"])
                #print(time_loss.shape)
            else:
                #print(inter_last_states)
                time_loss = self.intensity_loss(T_targets, inter_last_states)
            #print(time_loss)
            #mask out "fake session time-gaps" from time loss
            #print(self.time_threshold)
            #print(T_targets.data)
            mask = (T_targets.data.ge(self.time_threshold).float())
            #print(mask)
            time_loss = time_loss*mask

            #find number of non-ignored time gaps                                                                                                                           
            non_zero_count = 0
            for sign in mask.data:
                if (sign != 0):
                    non_zero_count += 1
            time_loss_divisor = (torch.tensor([max(non_zero_count,1)], dtype=torch.float, device=self.device))
            mean_time_loss = time_loss.sum(0)/time_loss_divisor

            #calculate gradients
            combined_loss = self.params["ALPHA"]*mean_time_loss + self.params["BETA"]*mean_loss + self.params["GAMMA"]*mean_first_loss
            combined_loss.backward()

            #update parameters through BPTT, options for freezing parts of the network
            if(self.flags["train_time"]):
                self.time_optimizer.step()
            if(self.flags["train_first"]):
                self.first_rec_optimizer.step()
            if(self.flags["train_all"]):
                self.inter_intra_optimizer.step()
        else:
            mean_loss.backward()
            self.inter_intra_optimizer.step()
        return mean_loss.data[0]

    def predict_on_batch(self, items, session_reps, sess_time_reps, user_list, item_targets, time_targets, first_rec_targets, session_lengths, session_rep_lengths, time_error, session_durations):
        #get batch from datahandler and turn into s
        X, S, S_gaps, U, S_durations = self.process_batch_inputs(items, session_reps, sess_time_reps, user_list, session_durations)

        #get embedded times
        if(self.flags["context"]):
            #get embedded times
            embedded_S_gaps = self.time_embed(S_gaps)

            #get embedded user
            embedded_U = self.user_embed(U)
            embedded_U = embedded_U.unsqueeze(1)
            embedded_U = embedded_U.expand(embedded_U.size(0), embedded_S_gaps.size(1), embedded_U.size(2))

        #get the index of the last session representation of each user by subtracting 1 from each lengths, move to  for efficiency
        rep_indicies = (torch.tensor(session_rep_lengths, dtype=torch.long, device=self.device)) - 1

        #get initial hidden state of inter gru layer and call forward on the module
        inter_hidden = self.inter_rnn.init_hidden(S.size(0))
        if(self.flags["context"]):
            #inter_last_hidden = self.inter_rnn(torch.cat((S, embedded_S_gaps, embedded_U),2), inter_hidden, rep_indicies)

            if(not self.ctlstm):
                inter_last_hidden = self.inter_rnn(torch.cat((S, embedded_S_gaps, embedded_U),2), inter_hidden, rep_indicies, S_durations)
                #print(inter_last_hidden.shape)
            else:
                inter_last_states = self.inter_rnn(torch.cat((S, embedded_S_gaps, embedded_U),2), inter_hidden, rep_indicies, S_durations)
                inter_last_hidden = get_ctlstm_hidden(inter_last_states, torch.zeros(len(time_targets), device=self.device))[0]
        else:
            if(not self.ctlstm):
                inter_last_hidden = self.inter_rnn(S, inter_hidden, rep_indicies, S_durations)
            else:
                inter_last_states = self.inter_rnn(S, inter_hidden, rep_indicies, S_durations)
                inter_last_hidden = get_ctlstm_hidden(inter_last_states, torch.zeros(len(time_targets), device = self.device))[0]


        #get time scores and first prediction scores from the last hidden state of the inter RNN
        if(self.flags["temporal"]):


            if(not self.ctlstm):
                times = self.time_linear(inter_last_hidden).squeeze()
            first_predictions = self.first_linear(inter_last_hidden).squeeze()

            #calculate time error if this is desired
            if(time_error):
                if(self.ctlstm):
                    time_predictions = self.intensity_loss.sample(inter_last_states)
                else:
                    w = self.time_loss_func.get_w()
                    time_predictions = self.time_prediction(times.data, w.data)


                self.tester.evaluate_batch_time(time_predictions, time_targets)

        #get item embeddings
        embedded_X = self.item_embed(X)

        #create average pooling session representation using the item embeddings and the lenght of each sequence
        lengths = (torch.tensor(session_lengths, dtype=torch.float, device=self.device).view(-1,1)) #reshape the lengths in order to broadcast and use it for division
        sum_X = embedded_X.sum(1)
        mean_X = sum_X.div(lengths)



        #subtract 1 from the lengths to get the index of the last item in each sequence
        lengths = lengths.long()-1

        #call forward on the inter RNN
        recommendation_output, hidden_out = self.intra_rnn(embedded_X, inter_last_hidden, lengths)

        #store the new session representation based on the current scheme
        if(self.flags["use_hidden"]):
            self.datahandler.store_user_session_representations(hidden_out.data[0], user_list, time_targets)
        else:
            self.datahandler.store_user_session_representations(mean_X.data, user_list, time_targets)
        
        if(self.flags["temporal"]):
            k_values, k_predictions = torch.topk(torch.cat((first_predictions.unsqueeze(1),recommendation_output),1), self.params["TOP_K"])
        else:
            k_values, k_predictions = torch.topk(recommendation_output, self.params["TOP_K"])
        return k_predictions
    

    
