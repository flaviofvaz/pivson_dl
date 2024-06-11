import gym
from omegaconf import DictConfig
import numpy as np


class GymEnvironment:
    def __init__(self, cfg: DictConfig, seed: int = 0):
        if cfg.max_episode_steps == "None":
            env = gym.make(id=f"{cfg.environment_id}NoFrameskip-v4", render_mode=cfg.render_mode, full_action_space=cfg.full_action_space)
        else:
            env = gym.make(id=f"{cfg.environment_id}NoFrameskip-v4", max_episode_steps=cfg.max_episode_steps, render_mode=cfg.render_mode, 
                           full_action_space=cfg.full_action_space)
        env = gym.wrappers.AtariPreprocessing(env, noop_max=cfg.no_op_max, frame_skip=cfg.frame_skip,
                                              screen_size=cfg.screen_size, terminal_on_life_loss=False,
                                              grayscale_obs=cfg.grayscale, scale_obs=cfg.scale_obs)
        env = gym.wrappers.FrameStack(env, num_stack=cfg.frame_stack)
        self.env = env
        self.lives = None
        self.timestep = 0
        self.max_reward = cfg.max_reward
        self.discount = cfg.discount_reward

        self.seed(seed)

    def step(self, action: int, terminal_on_life_loss: bool = False, end_on_life_loss: bool = False, 
             discount: bool = False, clip_reward: bool = True):
        state, reward, done, trunc, info = self.env.step(action)
        #if self.max_reward != 'None' and clip_reward:
        #    reward = np.clip(reward, -self.max_reward, self.max_reward)
        
        if discount:
            reward *= self.discount**self.timestep

        done = done or trunc or "TimeLimit.truncated" in info
        loss_of_life = False
        if terminal_on_life_loss and not done and "lives" in info:
            if self.lives is None:
                self.lives = info["lives"]
            else:
                if info["lives"] < self.lives:
                    loss_of_life = True
                    self.lives = info["lives"]
                    if end_on_life_loss:
                        done = True
        # update timestep
        self.timestep += 1
        
        return state, reward, done, loss_of_life

    def reset(self):
        self.lives = None
        self.timestep = 0
        return self.env.reset()
    
    def close(self):
        self.env.close()
        
    def get_num_actions(self):
        return self.env.action_space.n

    def seed(self, seed):
        self.env.seed(seed)
        self.env.action_space.seed(seed)