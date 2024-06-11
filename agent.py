import torch
import numpy as np
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torch import nn
import copy
import optimizers
import math
from omegaconf import DictConfig
import torch_optimizer as optim
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class DQNAgent:
    def _get_dqn(self, input_dim, bias):
        net = nn.Sequential(
                nn.Conv2d(input_dim, 32, (8, 8), (4, 4), bias=bias),
                nn.ReLU(),
                nn.Conv2d(32, 64, (4, 4), (2, 2), bias=bias),
                nn.ReLU(),
                nn.Conv2d(64, 64, (3, 3), (1, 1), bias=bias),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136, 512, bias=bias),
                nn.ReLU(),
                nn.Linear(512, self.n_actions, bias=bias)
            )
        return net

    def init_weights(self, w_init, b_init):
        for m in self.online_net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                if w_init == "truncated_uniform":
                    bound = math.sqrt(1 / fan_in)
                    torch.nn.init.uniform_(m.weight, -bound, bound)
                elif w_init == "truncated_normal":
                    std = 1 / math.sqrt(fan_in)
                    torch.nn.init.normal_(m.weight, 0, std)
                elif w_init == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                elif w_init == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                elif w_init == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif w_init == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif w_init == "default":
                    pass
                else:
                    raise NotImplementedError
                if m.bias is not None:
                    if isinstance(b_init, float):
                        torch.nn.init.constant_(m.bias, b_init)
                    else:
                        bound = math.sqrt(1 / fan_in)
                        torch.nn.init.uniform_(m.bias, -bound, bound)
                    

    def __init__(self, num_actions: int, input_dim: int, agent_cfg: DictConfig, optimizer_cfg: DictConfig):
        self.curr_step = 0
        self.initialize_memory_steps = agent_cfg.burn_in
        self.n_actions = num_actions

        # create online network
        self.online_net = self._get_dqn(input_dim=input_dim, bias=agent_cfg.use_bias)
        self.init_weights(agent_cfg.weights, agent_cfg.bias)
        
        # set target network to be the same as online network and freeze layers
        self.target_net = copy.deepcopy(self.online_net)
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.online_net.to(self.device)
        self.target_net.to(self.device)

        self.batch_size = agent_cfg.batch_size
        self.gamma = agent_cfg.gamma
        self.exploration_rate = agent_cfg.exploration_rate
        #self.exploration_rate_decay = agent_cfg.exploration_rate_decay
        self.exploration_rate_decay = (agent_cfg.exploration_rate - agent_cfg.exploration_rate_min) / (1e6 // (self.batch_size // 32))
        self.exploration_rate_min = agent_cfg.exploration_rate_min
        self.replay_memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(agent_cfg.replay_memory_size,
                                                                              device=torch.device("cpu")))
        self.sync_steps = agent_cfg.sync_steps // (self.batch_size / 32) 
        #self.final_lr = optimizer_cfg.lr * math.sqrt(self.batch_size / 32) 

        #self.optimizer = optimizers.RMSprop(self.online_net.parameters(), lr=self.final_lr,
        #                                    alpha=optimizer_cfg.alpha, eps=optimizer_cfg.eps,
        #                                    weight_decay=optimizer_cfg.w_decay,
        #                                    momentum=optimizer_cfg.momentum, centered=optimizer_cfg.centered)
        self.final_lr = 0.0000625 * math.sqrt(self.batch_size / 32)

        self.optimizer = optim.Lamb(self.online_net.parameters(), lr=self.final_lr, weight_decay=0.01)
        warmup_steps = int((1e7 * 0.3) // (self.batch_size // 32))
        max_steps = int(1e7 // (self.batch_size // 32))
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer, warmup_steps, max_steps, warmup_start_lr=0.0000625 * 1e-1, eta_min=0.0000625 * 1e-2)
        
        self.train_every_n_steps = agent_cfg.train_every_n_steps
        self.clip_gradient = agent_cfg.clip_gradient

        self.loss_fn = nn.HuberLoss()

    def get_action(self, state, epsilon_greedy):
        # e-greedy policy
        if np.random.rand() < epsilon_greedy:
            action = np.random.randint(self.n_actions)
        else:
            state = (torch.tensor(state, device=self.device).float()).unsqueeze(0)
            assert state.dim() == 4

            with torch.no_grad():
                q_estimates = self.online_net(state)
                action = torch.argmax(q_estimates, dim=1).item()

        return action

    def save_experience(self, state, action, reward, next_state, is_done):
        # Clip rewards
        reward = np.clip(reward, -1.0, 1.0)
        
        state = torch.tensor(state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        next_state = torch.tensor(next_state)
        done = torch.tensor([is_done])

        self.replay_memory.add(
            TensorDict({
                "state": state,
                "next_state": next_state,
                "action": action,
                "reward": reward,
                "done": done
            }, batch_size=[]))

    def retrieve_experience(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = self.replay_memory.sample(self.batch_size).float()
        batch = batch.to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in
                                                   ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        # state = state.float()
        q_values = self.online_net(state)[
            np.arange(0, self.batch_size), action.int()
        ]
        return q_values

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        # next_state = next_state.float()
        next_q_values = self.target_net(next_state).max(1)[0]
        return (reward + (1 - done.float()) * self.gamma * next_q_values).float()
    
    def get_loss(self, td_target, td_estimate):
        loss = self.loss_fn(td_target, td_estimate)
        return loss

    def update_weights(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def sync_nets(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
    
    def update_step(self):
        # update exploration_rate
        self.exploration_rate = self.exploration_rate - self.exploration_rate_decay * self.curr_step
        #self.exploration_rate -= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # update step
        self.curr_step += 1

    def learn(self, state, action, reward, next_state, done):
        self.save_experience(state, action, reward, next_state, done)
        if self.curr_step < self.initialize_memory_steps:
            self.curr_step += 1
            return None, None

        if self.curr_step % self.sync_steps == 0:
            self.sync_nets()

        if self.curr_step % self.train_every_n_steps == 0:
            state, next_state, action, reward, done = self.retrieve_experience()
            
            with torch.autocast(device_type=self.device):
                td_est = self.td_estimate(state, action)
            
                td_tgt = self.td_target(reward, next_state, done)

                loss = self.get_loss(td_tgt, td_est) 
            
            self.update_weights(loss)
            self.update_step()
            return td_est.mean().item(), loss.item()
        else:
            self.update_step()
            return None, None
    
    def load_weights(self, path):
        self.online_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))

    def get_agent(self):
        return self.online_net.state_dict()
