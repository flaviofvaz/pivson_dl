import torch
from agent import DQNAgent
from environment import GymEnvironment
from pathlib import Path
import os
import warnings
import numpy as np
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from logger import Logger

warnings.resetwarnings()
warnings.simplefilter("ignore", DeprecationWarning)


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def train(cfg: DictConfig) -> None:
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device}.")
    print()

    seed = cfg.training.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    print(f"Using as seed: {seed}")
    print()

    save_dir = Path("./tmp")
    save_dir.mkdir(parents=True, exist_ok=True)

    env = GymEnvironment(cfg=cfg.environment, seed=seed)

    img_dim = 1 if cfg.environment.grayscale else 3
    agent = DQNAgent(num_actions=env.get_num_actions(), input_dim=cfg.environment.frame_stack*img_dim, 
                     agent_cfg=cfg.agent, optimizer_cfg=cfg.optimizer)

    # New instance of logger
    logger = Logger()

    print("Training...")

    # Play games
    e = 0
    curr_step = 0

    state = env.reset()
    state = process_state(state)

    best_r = None
    best_agent = None

    # play
    while curr_step < (cfg.training.train_steps // (agent.batch_size // 32)):
        # get action
        action = agent.get_action(state, agent.exploration_rate)

        # take action
        next_state, reward, done, loss_of_life = env.step(action, True, False, False, True)

        # process state
        next_state = process_state(next_state)

        # learn from experience
        q, loss = agent.learn(state, action, reward, next_state, (done or loss_of_life))

        # update current state
        state = next_state

        # log step
        logger.log_step(reward, loss, q)

        # update curr_step
        curr_step += 1
        
        # end game?
        if done:
            # reset game
            state = env.reset()
            state = process_state(state)
            # log episode
            logger.log_episode()
            # update episode number
            e = e + 1

        # logging progress
        if curr_step % cfg.training.report_frequency == 0:
            # log metrics to console
            mean_ep_reward, mean_ep_length, mean_ep_loss, mean_ep_q = logger.record(e,
                                                                                    agent.exploration_rate,
                                                                                    curr_step)
        # evaluate model
        if curr_step % (cfg.training.test_frequency // (agent.batch_size // 32)) == 0 and curr_step > cfg.training.burn_in_steps:
            print("Evaluating model...")
            metrics = test(env=env, agent=agent, steps=cfg.training.test_steps)

            mean = np.mean(metrics)
            std = np.std(metrics)
            print(f"Average evaluation reward: {mean}")

            # restart environment
            state = env.reset()
            state = process_state(state)

            # save best agent
            if best_r is None or mean > best_r:
                best_r = mean
                best_agent = agent.get_agent()
                
        print("Training done. Saving agent...")
        print()

        # Save agent
        filepath = os.path.join(save_dir, "best_agent.pth")
        torch.save(best_agent, filepath)
        print("Agent saved.")
        print()

        env.close()


@torch.no_grad()
def test(env: GymEnvironment, agent, steps):
    # test metrics
    rewards = []
    ep_r = 0.0

    state = env.reset()
    for _ in range(steps):
        # get action
        state = process_state(state)
        action = agent.get_action(state, 0.05)

        # take action
        next_state, reward, done, _ = env.step(action, False, False, False, False)

        # update episode metrics
        ep_r += reward

        # end game?
        if done:
            # update test metrics
            rewards.append(ep_r)
            ep_r = 0.0

            # restart game
            state = env.reset()
        else:
            # update current state
            state = next_state

    return rewards


def process_state(x):
    x = x[0] if isinstance(x, tuple) else x
    x = x.__array__()
    return x


if __name__ == '__main__':
    train()
