from topo_changer import TopoChanger
import torch
from torch import nn
from math import pi, log
from torch.optim import Adam
from collections import deque
from random import sample
import random
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem

class StateFeature(nn.Module):

    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim

        self.feature = nn.Sequential(
            nn.Linear(self.state_dim, 36),
            nn.Tanh(),
            nn.Linear(36, 36),
            nn.Tanh()
        )

    def forward(self, states):
        return self.feature(states)


class LocallyConnected(nn.Module):

    def __init__(self, topo, n_agent, states_dim, actions_dim, global_dim):
        super().__init__()
        self.topo = topo
        self.n_agent = n_agent
        self.states_dim = states_dim
        self.actions_dim = actions_dim
        self.global_dim = global_dim
        self.states_feature_fc, self.actions_layer, self.global_layer = self._create_state_features()
        self.topo = np.zeros([n_agent, n_agent])
        attenion_vars = [att.weight for atts in self.attentions_layer for att in atts] + [
            att.bias for atts in self.attentions_layer for att in atts]
        self.attention_optim = Adam(attenion_vars)
        for i, j in topo:
            self.topo[i, j] = 1
            self.topo[i, i] = 1
            self.topo[j, j] = 1
            self.topo[j, i] = 1
        self.dec = 0.5

        self.topo /= 5
        # self.topo = np.ones([self.n_agent, self.n_agent])/6
        # self.topo = np.eye(self.n_agent)
        self.topo = torch.from_numpy(self.topo).float()
        self.topo_changer = TopoChanger(self.n_agent, self.states_dim)

    def _create_state_features(self):
        states_feature_fc = []
        for i in range(self.n_agent):
            states_feature_fc.append(
                StateFeature(self.states_dim[i])
            )
            self.add_module("state_layer_%d" % i, states_feature_fc[-1])
        actions_layer = []
        attentions_layer = []
        for i in range(self.n_agent):
            actions_layer.append(nn.Linear(36, self.actions_dim[i]))
            actions_layer[-1].weight.mul(0.1)
            actions_layer[-1].bias.mul(0.1)
            self.add_module("action_layer_%d" % i, actions_layer[-1])

        value_layer = []
        for i in range(self.n_agent):
            attentions_layer.append([])
            value_layer.append([])
            for j in range(self.n_agent):
                attentions_layer[i].append(nn.Linear(36, 10))
                self.add_module("attention_layer_%d_%d" %
                                (i, j), attentions_layer[i][j])
                value_layer[i].append(nn.Linear(36, 36))
                self.add_module("value_layer_%d_%d" %
                                (i, j), value_layer[i][j])
        self.attentions_layer = attentions_layer
        self.value_layer = value_layer

        global_layer = nn.Linear(self.global_dim, 36)

        return states_feature_fc, actions_layer, global_layer

    def forward(self, states, global_state):
        # batch = states[0].size(0) > 1
        batch = False
        if batch and ep % 5 == 1:
            n_neighbor = 1
            if ep > 200:
                n_neighbor = 2
            if ep > 1000:
                n_neighbor = 3
            topo_np = self.topo_changer.get_new_topo(states, n_neighbor)
            self.topo = self.topo * 0.99 + 0.01 * topo_np
        if batch:
            self.topo_changer.train(states)

        states_feature = [self.states_feature_fc[i](
            states[i]) for i in range(self.n_agent)]

        if batch and ep < 1200 and ep % 5 == 1:
            attention_values = []
            for i in range(self.n_agent):
                i_attention = self.attentions_layer[i][i](
                    states_feature[i])
                attention_values.append([])
                for j in range(self.n_agent):
                    attention_values[i].append(torch.exp(0.005 * torch.sum(i_attention.detach(
                    ) * self.attentions_layer[i][j](states_feature[j].detach()), dim=1)))
                attention_values[i] = torch.stack(attention_values[i], dim=1)

            attention_values = torch.stack(attention_values, dim=1)
            attention_values /= attention_values.sum(dim=2, keepdim=True)

            # print(attention_values)
            torch.mean((attention_values - self.topo) ** 2).backward()
            self.attention_optim.step()
            self.attention_optim.zero_grad()
        # 1. PDA
        # 2. Attention-PDA
        # 3. Attention

        # 1. ss
        # 2. sa
        # 3. sj

        topo = []
        states_communication = []
        for i in range(self.n_agent):
            i_attention = self.attentions_layer[i][i](
                states_feature[i])
            topo.append([])
            value = []
            for j in range(self.n_agent):
                topo[i].append(torch.exp(
                    # 0.03
                    0.05 * torch.sum(i_attention * self.attentions_layer[i][j](states_feature[j]), dim=1)))
                value.append(self.value_layer[i][j](states_feature[j]))
            topo[i] = torch.stack(topo[i], dim=1)
            # topo[i][:] = torch.ones(self.n_agent)
            topo[i][:] = self.topo[i]
            # topo[i] /= topo[i].sum(dim=1, keepdim=True)
            # print(topo[i])
            # topo[i] = (1 - self.dec) * topo[i] + self.topo[i] * self.dec
            # topo[i] /= 2
            states_communication.append(topo[i].view(
                [-1, 1, self.n_agent]).bmm(torch.stack(value, dim=1)).view([-1, 36]))

        states_communication = torch.stack(states_communication, dim=2)

        # topo = torch.zeros([states[0].size(0), self.n_agent, self.n_agent])
        # topo = 0 * topo + self.topo
        # states_feature = torch.stack(states_feature, dim=1)

        # states_communication = topo.bmm(states_feature).transpose(1, 2)
        global_feature = self.global_layer(global_state)
        # print(global_state)
        actions = []
        for i in range(self.n_agent):
            # actions.append(torch.tanh(self.actions_layer[i](torch.cat([states_communication[:, :, i], global_feature], dim=1))))
            actions.append(torch.tanh(self.actions_layer[i](
                states_communication[:, :, i] + global_feature)))
        return torch.cat(actions, dim=1)


class FullConnected(nn.Module):

    def __init__(self, topo, n_agents, input_dims, out_dims, global_dim):
        super().__init__()
        self.input_dims = input_dims
        self.out_dims = out_dims
        self.global_dim = global_dim
        self.agent_cnt = len(self.input_dims)
        self._create_variables()

    def _create_variables(self):
        self.ff_layer1 = nn.Linear(sum(self.input_dims) + self.global_dim, 64)
        self.ff_layer2 = nn.Linear(64, 64)
        self.ff_out_layer = nn.Linear(64, sum(self.out_dims))

    def forward(self, inputs, global_input=None):
        ff = torch.cat(inputs + [global_input], dim=1)
        ff = torch.tanh(self.ff_layer1(ff))
        ff = torch.tanh(self.ff_layer2(ff))
        ff = self.ff_out_layer(ff)
        return torch.tanh(ff)


class Actor(nn.Module):

    def __init__(self, topo, state_dims, act_dims, global_dim, state_split_func, action_split_func):
        super().__init__()
        self.network = LocallyConnected(
            topo, len(state_dims), state_dims, act_dims, global_dim)
        self.state_dim = sum(state_dims) + global_dim
        self.act_dim = sum(act_dims)
        self.state_split_func = state_split_func
        self.action_split_func = action_split_func

        self.log_std = torch.zeros(sum(act_dims)) - 0.7
        # self.log_std[0] = -10

    def forward(self, states):
        states = torch.Tensor(states).view([-1, self.state_dim])
        # print(states)

        local_states = self.state_split_func(states)
        mu = self.network(local_states[:-1], local_states[-1])
        std = torch.exp(self.log_std)

        return mu, std, self.log_std

    @staticmethod
    def log_density(x, mu, std, log_std):
        var = std.pow(2)
        log_density = -(x - mu).pow(2) / (2 * var) \
                      - 0.5 * log(2 * pi) - log_std
        return log_density.sum(1, keepdim=True)

    @staticmethod
    def kl_divergence(new_actor, old_actor, states):
        mu, std, logstd = new_actor(torch.Tensor(states))
        mu_old, std_old, logstd_old = old_actor(torch.Tensor(states))
        mu_old = mu_old.detach()
        std_old = std_old.detach()
        logstd_old = logstd_old.detach()

        # kl divergence between old policy and new policy : D( pi_old || pi_new )
        # pi_old -> mu0, logstd0, std0 / pi_new -> mu, logstd, std
        # be careful of calculating KL-divergence. It is not symmetric metric
        kl = logstd_old - logstd + (std_old.pow(2) + (mu_old - mu).pow(2)) / \
            (2.0 * std.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)


class Critic(nn.Module):

    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, states):
        states = torch.Tensor(states).view([-1, self.state_dim])

        return self.network(states)


class PPO:

    def __init__(self, topo, state_dims, act_dims, global_dim,
                 state_split_func, action_split_func, lr, gamma, replay_sz, sample_sz):
        self.state_dim = sum(state_dims) + global_dim
        self.act_dim = sum(act_dims)
        self.clip_param = 0.3
        self.lr = lr
        self.gamma = gamma
        self.lam = 0.9
        self.sample_sz = sample_sz

        self.critic = Critic(self.state_dim)
        self.actor = Actor(topo, state_dims, act_dims,
                           global_dim, state_split_func, action_split_func)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = deque(maxlen=replay_sz)

    def save(self, path):
        torch.save([self.actor, self.critic], path)

    def load(self):
        self.actor, self.critic = torch.load("model.pkl")
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

    def _remember_batch(self, states, rewards):
        long_rewards = []
        sum_reward = 0
        for r in reversed(rewards):
            sum_reward = sum_reward * self.gamma + r
            long_rewards.append(sum_reward)
        samples = [[s, r] for s, r in zip(states, reversed(long_rewards))]

        self.replay_buffer.extend(samples)

    def train(self, states, actions, rewards, next_states):
        self._remember_batch(states, rewards)
        if ep > 700:
            self.clip_param = 0.2

        if len(self.replay_buffer) < 3 * self.sample_sz:
            return

        states = torch.Tensor(states).view([-1, self.state_dim])
        next_states = torch.Tensor(next_states).view([-1, self.state_dim])
        # rewards = torch.Tensor(rewards).view([-1, 1])
        actions = torch.Tensor(actions).view([-1, self.act_dim])
        next_v = self.critic(next_states).detach()
        v = self.critic(states).detach()
        # print(v.mean(), v[0])
        adv = []
        for i in range(0, len(v)):
            adv.append(0)
            tr = -v[i].item()
            for j in range(i, len(v)):
                tr += rewards[j] * self.gamma ** (j - i)
                adv[i] += (tr + next_v[j].item() * self.gamma **
                           (j - i + 1)) * self.lam ** (j - i)
            adv[i] *= (1 - self.lam)
        adv = torch.Tensor(adv).view([-1, 1])
        adv = (adv - adv.mean()) / (adv.std() + 1e-4)

        mu, std, log_std = self.actor(states)
        old_density = Actor.log_density(mu, actions, std, log_std).detach()
        for _ in range(10):
            mu, std, log_std = self.actor(states)
            new_density = Actor.log_density(mu, actions, std, log_std)

            ratio = torch.exp(new_density - old_density)
            clipped_ratio = torch.clamp(ratio,
                                        1.0 - self.clip_param,
                                        1.0 + self.clip_param)
            clipped_loss = clipped_ratio * adv

            loss = ratio * adv

            actor_loss = -torch.min(loss, clipped_loss).mean()
            actor_loss.backward()
            # nn.utils.clip_grad_norm(self.actor.parameters(), 0.5, norm_type=2)
            self.actor_optim.step()
            self.actor_optim.zero_grad()

        num_batch = max(len(states) // 256, 1)
        num_sample = len(states) // num_batch

        for _ in range(10):
            for _ in range(num_batch):
                samples = sample(self.replay_buffer, num_sample)
                states = torch.Tensor([s for s, r in samples])
                target_v = torch.Tensor([r for s, r in samples]).view([-1, 1])

                v = self.critic(states)
                critic_loss = torch.mean((v - target_v) ** 2)
                critic_loss.backward()
                # nn.utils.clip_grad_norm(self.actor.parameters(), 0.5, norm_type=2)
                self.critic_optim.step()
                self.critic_optim.zero_grad()

    def get_action(self, states):
        mu, std, log_std = self.actor(states)
        return torch.clamp(torch.normal(mu, std).detach(), -1, 1).numpy()


def split_act(act):
    return act


def split_obs(obs):
    """
    分割全局状态、动作为局部状态、动作
    """
    obss = []

    obss.append(torch.stack((obs[:, 2], obs[:, 8]), dim=1))
    obss.append(torch.stack((obs[:, 3], obs[:, 9]), dim=1))
    obss.append(torch.stack((obs[:, 4], obs[:, 10]), dim=1))

    obss.append(torch.stack(
        (obs[:, 0], obs[:, 1], obs[:, 5], obs[:, 6], obs[:, 7], obs[:, -1]), dim=1))

    return obss


if __name__ == '__main__':
    # torch.manual_seed(29)
    # random.seed(21)
    # np.random.seed(218)
    env_name = "Hopper-v2"
    env = gym.make(env_name)
    # print(env.observation_space, env.action_space)  # Box(11,) Box(3,)
    env.seed(2180)
    runs = 1
    episodes = 2400

    for run in range(runs):
        torch.manual_seed(29)
        random.seed(21)
        np.random.seed(218)
        from running_mean_std import RunningMeanStd
        rms = RunningMeanStd(env.observation_space.shape[0])
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

        obs_dims = (2, 2, 2)
        act_dims = (1, 1, 1)
        global_dim = 6

        topo = (
            (0, 1),
            (1, 2)
        )

        import pickle

        # lr = [1e-4, 1e-3]
        # replay_sz = [i for i in range(5000, 10000, 1000)]
        # sample_sz = [16, 32]
        alr = [1e-4, 1e-3]
        areplay_sz = [5000, 6000, 7000, 8000, 9000]
        asample_sz = [16, 32]
        for al in alr:
            for areplay in areplay_sz:
                for asample in asample_sz:
                    R = []
                    algo = PPO(topo, obs_dims, act_dims, global_dim,
                               split_obs, split_act, al, 0.999, areplay, asample)
                    # lr, gamma, replay_sz, sample_sz

                    dump_rewards = []
                    rewards_list = []
                    slide_list = []
                    for ep in range(episodes):

                        states = []
                        rewards = []
                        next_states = []
                        actions = []
                        total_rewards = 0
                        done = False

                        time_step = 0
                        state = env.reset()
                        scale, offset = rms.get()
                        unscaled_states = []
                        unscaled_states.append(state)
                        state = (state - offset) * scale
                        state = np.concatenate([state, [time_step]])

                        while not done:
                            action = algo.get_action(state)[0]
                            # print(action)
                            next_state, reward, done, info = env.step(action)
                            unscaled_states.append(next_state)
                            next_state = (next_state - offset) * scale
                            # if ep % 200 == 0:
                            # nv.render()e
                            states.append(state)
                            time_step += 1e-3
                            next_state = np.concatenate([next_state, [time_step]])
                            next_states.append(next_state)
                            actions.append(action)
                            rewards.append(reward)
                            state = next_state
                            total_rewards += reward
                        print(ep, total_rewards)
                        R.append(total_rewards)

                        rms.update(np.stack(unscaled_states, axis=0))
                        writer.add_scalar('reward', total_rewards, ep)
                        # print(len(states))
                        algo.train(states, actions, rewards, next_states)
                    dict = ['R']
                    data = pd.DataFrame(R)
                    pd.DataFrame(data)
                    pd.DataFrame(data).to_csv('/home/lvbaiyang/Desktop/'
                                              + str(al)
                                              + '_' + str(areplay)
                                              + '_' + str(asample) + '.csv',
                                              header=dict)


