import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import onehot_from_logits, categorical_sample

class BasePolicy(nn.Module):
    """
    Base policy network
    """
    def __init__(self, sa_sizes, hidden_dim=64,onehot_dim=0, nonlin=F.leaky_relu,
                 norm_in=True, partial_obs=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(BasePolicy, self).__init__()
        self.partial_obs = partial_obs
        self.nagent = len(sa_sizes)

        self.state_enc = nn.ModuleList()
        self.commun = nn.ModuleList()
        self.actor = nn.ModuleList()

        for sdim, adim in sa_sizes:
            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('enc_bn', nn.BatchNorm1d(sdim,
                                                            affine=False))
            state_encoder.add_module('enc_fc',nn.Linear(sdim, hidden_dim))
            state_encoder.add_module("enc_n", nn.LeakyReLU())
            self.state_enc.append(state_encoder)
            #encode the state

            commun = AMUCell(hidden_dim, hidden_dim)
            self.commun.append(commun)
            # the vortex module

            actor = nn.Sequential()
            actor.add_module("act_fc",nn.Linear(hidden_dim, adim))
            self.actor.append(actor)
            # the output layer
        self.shared_modules = [self.commun]

        self.nonlin = nonlin

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps):
        states = [s for s, h, c in inps]
        h_prevs = [h for s, h, c in inps]
        c_prevs = [c for s, h, c in inps]
        hiddens = [state_enc(state) for state_enc, state in zip(self.state_enc, states)]

        latents = []  # store the results of commun
        #new_hidden = []
        new_cell = []  # the new cell state for each agent
        agents = range(self.nagent)
        for i in agents:
            hidden,cell = self.commun[i].forward(hiddens[i], h_prevs[i], c_prevs[i])
            new_cell.append(cell)
            latents.append(hidden)
        if self.partial_obs:
            for i in agents:
                for k in [3,2,1]:
                    ind = int(states[i][0][-k])
                    latents[i],_ = self.commun[ind].forward(hiddens[ind].detach(),latents[i],c_prevs[ind])
        else:
            for i in agents:
                #hidden, cell = self.commun[i].forward(hiddens[i], h_prevs[i], c_prevs[i])
                j = i + 1
                #new_cell.append(cell)
                if j == len(agents):
                    j = 0
                while True:
                    latents[i], _ = self.commun[j].forward(hiddens[j].detach(), latents[i], c_prevs[j])
                    j += 1
                    if j == len(agents):
                        j = 0
                    if i == j:
                        #new_hidden.append(late)
                        break

        out = [actor(latent) for actor, latent in zip(self.actor, latents)]

        return out, latents,new_cell



class DiscretePolicy(BasePolicy):
    """
    Policy Network for discrete action spaces
    """
    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, inps, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False):
        outs,new_h,new_c = super(DiscretePolicy, self).forward(inps)
        retss = []
        for out in outs:
            probs = F.softmax(out, dim=1)
            on_gpu = next(self.parameters()).is_cuda
            if sample:
                int_act, act = categorical_sample(probs, use_cuda=on_gpu)
            else:
                act = onehot_from_logits(probs)
            rets = [act]
            if return_log_pi or return_entropy:
                log_probs = F.log_softmax(out, dim=1)
            if return_all_probs:
                rets.append(probs)
            if return_log_pi:
                # return log probability of selected action
                rets.append(log_probs.gather(1, int_act))
            if regularize:
                rets.append([(out**2).mean()])
            if return_entropy:
                rets.append(-(log_probs * probs).sum(1).mean())
            retss.append(rets)
        if len(retss[0]) == 1:
            new_retss = []
            for ret in retss:
                new_retss.append(ret[0])
            return new_retss, new_h, new_c
        return retss,new_h,new_c

class AMUCell(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(AMUCell, self).__init__()
        self.Wxr = nn.Linear(input_size, hidden_size)
        self.Whr = nn.Linear(hidden_size, hidden_size)
        self.Wxz = nn.Linear(input_size, hidden_size)
        self.Whz = nn.Linear(hidden_size, hidden_size)

        self.Wx = nn.Linear(input_size, hidden_size)
        self.Wh = nn.Linear(hidden_size, hidden_size)

        self.Wa = nn.Linear(input_size+hidden_size,hidden_size)
        self.Woc = nn.Linear(2*hidden_size,hidden_size)
    def forward(self, x, h_prev, c_prev):
        Wxr = self.Wxr(x)
        Whr = self.Whr(c_prev)
        r = torch.sigmoid(Wxr + Whr)

        Wxz = self.Wxz(x)
        Whz = self.Whz(c_prev)
        z = torch.sigmoid(Wxz + Whz)

        Wx = self.Wx(x)
        Wh = self.Wh(r * c_prev)
        c_temp = torch.tanh(Wx + Wh)

        new_c = (1 - z) * c_prev + z *c_temp

        a = F.softmax(torch.sigmoid(self.Wa(torch.cat((x,h_prev),dim=1))))

        new_h = torch.tanh(self.Woc(torch.cat((a*new_c,h_prev),dim=1)))


        return new_h, new_c