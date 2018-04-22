import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import pdb
import numpy as np

## ref: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

class Policy(nn.Module):
    def __init__(self, num_actions, hidden_size, gamma):
        super(Policy, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_controller = nn.GRUCell(num_actions, hidden_size)
        self.rnn_decoder = nn.Linear(hidden_size, num_actions)

        self.gamma = gamma
        self.reset()

    def reset(self):
        self.rewards = []
        self.saved_log_probs = []
        self.hx = Variable(torch.zeros(1, self.hidden_size), requires_grad=False)
        if torch.cuda.is_available():
            self.hx = self.hx.cuda()

    def forward(self, x):
        self.hx = self.rnn_controller(x, self.hx)
        action_scores = self.rnn_decoder(self.hx)
        return F.softmax(action_scores, dim=1)

    def select_action(self, state):
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.data[0]

    def finish_episode(self, baseline):

        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = Variable(torch.Tensor(rewards), requires_grad=False)
        if torch.cuda.is_available():
            rewards = rewards.cuda()
        #rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * (reward-baseline))
        policy_loss = torch.cat(policy_loss).sum()

        del self.rewards[:]
        del self.saved_log_probs[:]
        self.reset()
        return policy_loss
