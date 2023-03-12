import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain


class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AttentionCritic, self).__init__()
        #assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        #self.attend_heads = attend_heads
        self.action_encoders = nn.ModuleList()
        #self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        #self.state_encoders = nn.ModuleList()
        # iterate over agents
        for sdim, adim in sa_sizes:
            idim = sdim + adim
            odim = adim
            encoder = nn.Sequential()
            encoder.add_module('enc_fc1', nn.Linear(adim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.action_encoders.append(encoder)
            #self.critic_encoders.append(encoder)

            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)


        #attend_dim = hidden_dim // attend_heads
        #self.key_extractors = nn.ModuleList()
        #self.selector_extractors = nn.ModuleList()
        #self.value_extractors = nn.ModuleList()
        #for i in range(attend_heads):
            #self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            #self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            #self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
                                                                #attend_dim),
                                                       #nn.LeakyReLU()))

        self.shared_modules = []

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

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        if agents is None:
            agents = range(self.nagents)
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        a_encodings = [self.action_encoders[a_i](actions[a_i]) for a_i in agents]
        all_rets = []
        for i, a_i in enumerate(agents):
            agent_rets = []
            critic_in = torch.cat((a_encodings[i], states[i]), dim=1)
            all_q = self.critics[a_i](critic_in)
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets
