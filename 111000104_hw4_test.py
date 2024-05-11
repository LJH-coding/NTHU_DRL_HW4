import torch
import torch.nn as nn
import numpy as np

EPS = 1e-6

class TanhGaussDistribution:
    def __init__(self, logits):
        self.logits = logits
        self.mean, self.std = torch.chunk(logits, chunks=2, dim=-1)
        self.gauss_distribution = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(self.mean, self.std),
            reinterpreted_batch_ndims=1,
        )
        self.act_high_lim = torch.tensor([1.0] * 22) #environment action_space high
        self.act_low_lim = torch.tensor([0.0] * 22) #environment action_space low

    def sample(self):
        action = self.gauss_distribution.sample()
        action_limited = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(action) + (self.act_high_lim + self.act_low_lim) / 2
        log_prob = (
                self.gauss_distribution.log_prob(action)
                - torch.log(1 + EPS - torch.pow(torch.tanh(action), 2)).sum(-1)
                - torch.log((self.act_high_lim - self.act_low_lim) / 2).sum(-1)
        )
        return action_limited, log_prob

    def rsample(self):
        action = self.gauss_distribution.rsample()
        action_limited = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(action) + (self.act_high_lim + self.act_low_lim) / 2
        log_prob = (
                self.gauss_distribution.log_prob(action)
                - torch.log(1 + EPS - torch.pow(torch.tanh(action), 2)).sum(-1)
                - torch.log((self.act_high_lim - self.act_low_lim) / 2).sum(-1)
        )
        return action_limited, log_prob

    def log_prob(self, action_limited) -> torch.Tensor:
        action = torch.atanh(
            (1 - EPS)
            * (2 * action_limited - (self.act_high_lim + self.act_low_lim))
            / (self.act_high_lim - self.act_low_lim)
        )
        log_prob = self.gauss_distribution.log_prob(action) - torch.log(
            (self.act_high_lim - self.act_low_lim)
            * (1 + EPS - torch.pow(torch.tanh(action), 2))
        ).sum(-1)
        return log_prob

    def entropy(self):
        return self.gauss_distribution.entropy()

    def mode(self):
        return (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(self.mean) + (self.act_high_lim + self.act_low_lim) / 2

    def kl_divergence(self, other: "GaussDistribution") -> torch.Tensor:
        return torch.distributions.kl.kl_divergence(self.gauss_distribution, other.gauss_distribution)

class Action_Distribution:
    def __init__(self):
        super().__init__()

    def get_act_dist(self, logits):

        act_dist = TanhGaussDistribution(logits)

        return act_dist


class Layer(nn.Module):
    def __init__(self, in_features, out_features, residual=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.ELU()
        )
        self.residual = residual

    def forward(self, x):
        y = self.layer(x)
        if self.residual:
            y += x
        return y

class PolicyNetwork(nn.Module, Action_Distribution):
    def __init__(self):
        super().__init__()
        self.min_log_std = -20
        self.max_log_std = 0.5
        self.action_distribution_cls = TanhGaussDistribution
        self.mlp = nn.Sequential(
            Layer(339, 1024, residual=False),
            Layer(1024, 1024),
            Layer(1024, 1024),
        )
        self.mean = nn.Sequential(
            nn.Linear(1024, 22)
        )
        self.log_std = nn.Sequential(
            nn.Linear(1024, 22)
        )

    def forward(self, obs):
        feature = self.mlp(obs)
        action_mean = self.mean(feature)
        action_std = torch.clamp(self.log_std(feature), self.min_log_std, self.max_log_std).exp()
        return torch.cat((action_mean, action_std), dim=-1)

def unpack_dict_obs(observation):
    res = []
    if not isinstance(observation, dict):
        if not (isinstance(observation, np.ndarray) or isinstance(observation, list)):
            res.append(observation)
        else:
            for element in observation:
                res = res + unpack_dict_obs(element)
        return res
    
    for key in observation:
        res = res + unpack_dict_obs(observation[key])

    return res

def prepross_obs(observation):
    observation = np.array(unpack_dict_obs(observation))
    observation[:242]  /= 10
    return np.array(observation)

class ApproxContainer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.policy = PolicyNetwork()
    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)

class Agent(object):
    def __init__(self):
        self.model = ApproxContainer()
        self.model.load_state_dict(torch.load('./15_opt.pth'))
        self.timesteps = 0
        self.last_action = None
        self.reset_obs = None

    def act(self, observation):
        observation = prepross_obs(observation)
        try:
            if self.reset_obs.all() == observation.all():
                self.timesteps = 0
        except:
            self.reset_obs = observation
        if observation.all() == self.reset_obs.all():
            self.timestep = 0
        if self.timestep % 4 == 0:
            self.last_action = self.predict(observation)
        self.timesteps += 1
        return self.last_action

    def predict(self, obs):
        batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
        logits = self.model.policy(batch_obs)
        mean, std = torch.chunk(logits, chunks=2, dim=-1)
        action_distribution = self.model.create_action_distributions(logits)
        action = action_distribution.mode()
        action = action.detach().numpy()[0]
        return action
    
if __name__ == '__main__':
    agent = Agent()

