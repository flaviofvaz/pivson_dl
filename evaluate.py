import torch
from agent import DQNAgent
from logger import MetricLogger
from environment import GymEnvironment
import datetime
from pathlib import Path
import os
import warnings
import numpy as np
import random
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

warnings.resetwarnings()
warnings.simplefilter("ignore", DeprecationWarning)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device}.")
    print()

    seed = 32
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    print(f"Using as seed: {seed}")
    print()
    
    # testing config
    test_eps = 0.05
    test_episodes = 30
    model_path = "./checkpoints/2023-12-01T10-28-02/best_agent.pth"

    env = GymEnvironment(cfg=cfg.environment, seed=seed)

    img_dim = 1 if cfg.environment.grayscale else 3
    agent = DQNAgent(num_actions=env.get_num_actions(), input_dim=cfg.environment.frame_stack*img_dim, 
                     agent_cfg=cfg.agent, optimizer_cfg=cfg.optimizer)
    agent.load_weights(model_path)
    
    rewards = []
    for i in range(test_episodes):
        episode_reward = run_episode(env, agent, test_eps)
        rewards.append(episode_reward)
        print(f"Episode {i+1} reward: {episode_reward}")
    
    print(f"Average reward over {test_episodes} episodes: {np.mean(rewards)}")
    print(f"Standard deviation over {test_episodes} episodes: {np.std(rewards)}")


@torch.no_grad()
def run_episode(env, agent, test_eps):
    ep_r = 0.0

    state = env.reset()

    done = False
    while not done:
        # get action
        state = process_state(state)
        action = agent.get_action(state, test_eps)

        # take action
        next_state, reward, done, _ = env.step(action)

        # update episode metrics
        ep_r += reward

        # update current state
        state = next_state

    return ep_r


def process_state(x):
    x = x[0] if isinstance(x, tuple) else x
    x = x.__array__()
    return x


if __name__ == '__main__':
    evaluate()
