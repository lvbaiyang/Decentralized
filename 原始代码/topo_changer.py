import torch
import numpy as np
from torch import nn
from torch.optim import Adam
import heapq


class TopoChanger:

    def __init__(self, n_agent, states_dim):
        self.n_agent = n_agent
        self.states_dim = states_dim
        self.nets = []
        for i in range(self.n_agent):
            self.nets.append(nn.Sequential(
                nn.Linear(self.states_dim[i], 128),
                nn.Tanh(),
                nn.Linear(128, sum(self.states_dim))
            ))

        self.optims = [Adam(self.nets[i].parameters())
                       for i in range(self.n_agent)]

    def train(self, n_agent_states):
        all_states = torch.cat(n_agent_states, dim=1)
        for i in range(self.n_agent):
            predict_states = self.nets[i](n_agent_states[i])
            loss = torch.mean((predict_states - all_states) ** 2)
            loss.backward()
            self.optims[i].step()
            self.optims[i].zero_grad()

    def get_new_topo(self, n_agent_states, n_neighbor):
        topo = []
        for i in range(self.n_agent):
            predict_states = self.nets[i](n_agent_states[i])
            cur = 0
            losses = []
            for j in range(self.n_agent):
                loss = torch.mean(
                    (predict_states[:, cur: cur+self.states_dim[j]] - n_agent_states[j]) ** 2)
                cur += self.states_dim[j]
                losses.append([loss, j])

            neighbors = [j for l, j in heapq.nlargest(n_neighbor, losses)]
            topo.extend([(i, n) for n in neighbors])
        topo_np = np.zeros([self.n_agent, self.n_agent])
        for i, j in topo:
            topo_np[i, j] = 1
            topo_np[i, i] = 1
            topo_np[j, j] = 1
            # topo_np[j, i] = 1
        topo_np /= topo_np.sum(axis=1, keepdims=True)
        # print(topo_np)
        return torch.from_numpy(topo_np).float()
