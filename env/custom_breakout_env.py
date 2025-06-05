import gym
import numpy as np
import random
from gym import spaces

class CustomBreakoutEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(CustomBreakoutEnv, self).__init__()
        self.env = gym.make('BreakoutNoFrameskip-v4', render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # Adaptive parameters
        self.frame_count = 0
        self.paddle_speed = 1.0
        self.paddle_size = 1.0
        self.ball_speed = 1.0
        self.brick_regen_prob = 0.0

        self.env.metadata['render_fps'] = 60

    def reset(self, **kwargs):
        self.frame_count = 0
        self.paddle_speed = 1.0
        self.paddle_size = 1.0
        self.ball_speed = 1.0
        self.brick_regen_prob = 0.0

        obs, info = self.env.reset(**kwargs)  # Gym 0.26+ returns (obs, info)
        return obs, info

    def step(self, action):
        self.frame_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # === Dynamic difficulty logic ===
        difficulty_multiplier = 1.0

        # Paddle size change every 500 frames
        if self.frame_count % 500 == 0:
            self.paddle_size = random.choice([0.5, 1.0, 1.5])
            if self.paddle_size < 1.0:
                difficulty_multiplier *= 1.2
            elif self.paddle_size > 1.0:
                difficulty_multiplier *= 0.8

        # Increase ball speed randomly
        if random.random() < 0.01:
            self.ball_speed *= 1.1
            difficulty_multiplier *= 1.1

        # Simulate brick regeneration (if implemented)
        if random.random() < self.brick_regen_prob:
            self._regenerate_bricks()

        # Adjust reward based on difficulty
        adjusted_reward = reward * difficulty_multiplier

        return obs, adjusted_reward, terminated, truncated, info

    def _regenerate_bricks(self):
        # Placeholder — implement custom logic here
        pass

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
