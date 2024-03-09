import torch
from torch import nn
from torch import distributions
import torch.nn.functional as F
import hyperparams as hp
import numpy as np


hp = hp.HyperParameters()

if hp.nlin == 'elu':
    activation = F.elu
elif hp.nlin == 'tanh':
    activation = torch.tanh
else:
    activation = torch.nn.ReLU()


def check_is_terminal(layer, terminal):
    b_size = layer.shape[1]

    if terminal is not None:
        layer = layer * (1. - terminal).reshape(1, b_size, 1)
    return layer


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, continuous_action_space,
                 trainable_std_dev, init_log_std_dev=None):
        super().__init__()
        self.action_dim = action_dim

        logit_hidden_dim = int(np.sqrt((hp.hidden_size * action_dim)))

        '''Place holders for hidden layers'''
        self.z_hidden_layer = None
        self.a_hidden_layer = None

        self.z_rnn = nn.GRU(state_dim + action_dim,
                            hp.hidden_size,
                            num_layers=hp.recurrent_layers)

        self.layer_phi_z = nn.Linear(hp.hidden_size, logit_hidden_dim)

        # self.a_rnn = nn.GRU(action_dim,3*action_dim)
        # self.layer_phi_a = nn.Linear(3*action_dim, 3*action_dim)
        # self.layer_phi_pi = nn.Linear(3*action_dim + hp.hidden_size,
        # 							logit_hidden_dim)

        '''Definitions for the Policy Head'''
        self.layer_policy_logits = nn.Linear(logit_hidden_dim, action_dim)

        self.continuous_action_space = continuous_action_space
        self.log_std_dev = nn.Parameter(init_log_std_dev * torch.ones(action_dim, dtype=torch.float),
                                        requires_grad=trainable_std_dev)
        self.covariance_eye = torch.eye(action_dim).unsqueeze(0)


    def get_z_init_state(self, batch_size, device):
        self.z_hidden_layer = torch.zeros(hp.recurrent_layers,
                                          batch_size, hp.hidden_size).to(device)

    def get_a_init_state(self, batch_size, device):
        self.a_hidden_layer = torch.zeros(hp.recurrent_layers,
                                          batch_size, 3 * self.action_dim).to(device)

    def forward(self, state, a_ini, terminal=None):

        batch_size = state.shape[1]
        device = state.device

        '''Computing z_{1:t-1}'''

        if self.z_hidden_layer is None or batch_size != self.z_hidden_layer.shape[1]:
            self.get_z_init_state(batch_size, device)

        self.z_hidden_layer = check_is_terminal(self.z_hidden_layer, terminal)

        # z_prev = self.z_hidden_layer.detach().clone()
        # phi_z_prev = activation(self.layer_phi_zprev(z_prev[-1]))

        _, self.z_hidden_layer = self.z_rnn(torch.cat(tensors=(state, a_ini), dim=-1),
                                            self.z_hidden_layer)

        pi_state = activation(self.layer_phi_z(self.z_hidden_layer[-1, :]))

        '''Policy head'''
        policy_logits_out = self.layer_policy_logits(pi_state)

        if self.continuous_action_space:
            cov_matrix = self.covariance_eye.to(device).expand(batch_size, self.action_dim,
                                                               self.action_dim) * torch.exp(self.log_std_dev.to(device))
            ''' We define the distribution on the CPU since otherwise 
            operations fail with CUDA illegal memory access error.'''
            policy_dist = torch.distributions.multivariate_normal.MultivariateNormal(policy_logits_out.to("cpu"),
                                                                                     cov_matrix.to("cpu"))
        else:
            policy_dist = distributions.Categorical(F.softmax(policy_logits_out, dim=1).to("cpu"))
        return policy_dist, self.z_hidden_layer[-1, :]


class Critic_Coordinator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        logit_hidden_dim = int(np.sqrt((hp.hidden_size * action_dim)))

        '''Defining place holders for hidden layers'''
        self.hidden_layer = None

        self.z_rnn = nn.GRU(state_dim + action_dim,
                            hp.hidden_size,
                            num_layers=hp.recurrent_layers)
        self.layer_phi_z = nn.Linear(hp.hidden_size, logit_hidden_dim)

        self.layer_value = nn.Linear(logit_hidden_dim, 1)

    def get_init_state(self, batch_size, device):
        self.hidden_layer = torch.zeros(hp.recurrent_layers,
                                        batch_size, hp.hidden_size).to(device)

    def forward(self, state, a_ini, terminal=None):
        batch_size = state.shape[1]
        device = state.device

        if self.hidden_layer is None or batch_size != self.hidden_layer.shape[1]:
            self.get_init_state(batch_size, device)

        self.hidden_layer = check_is_terminal(self.hidden_layer, terminal)

        _, self.hidden_layer = self.z_rnn(torch.cat(tensors=(state, a_ini), dim=-1),
                                          self.hidden_layer)

        phi_z = activation(self.layer_phi_z(self.hidden_layer[-1]))

        value_out = self.layer_value(phi_z)

        return value_out


class Psi(nn.Module):
    def __init__(self, ais_dim, init_log_std_dev=None):
        super().__init__()

        self.layer_r_pred = nn.Linear(ais_dim, 1)
        self.layer_mu = nn.Linear(ais_dim, ais_dim)
        self.layer_weights = nn.Linear(ais_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ais):
        r_pred = activation(self.layer_r_pred(ais))
        next_ais = activation(self.layer_mu(ais))
        weights = self.sigmoid(self.layer_weights(ais))
        return r_pred, next_ais, weights









