import gym
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import cv2
from models.dqn_model import DQN
from utils.replay_buffer import ReplayBuffer
from utils.curriculum import CurriculumScheduler
from env.custom_breakout_env import CustomBreakoutEnv

# Preprocess frame: grayscale, resize, normalize
def preprocess(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84))
    return obs / 255.0

def train():
    env = CustomBreakoutEnv()
    input_shape = (4, 84, 84)
    num_actions = env.action_space.n

    policy_net = DQN(input_shape, num_actions)
    target_net = DQN(input_shape, num_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(10000)
    scheduler = CurriculumScheduler(total_steps=1000000)

    batch_size = 32
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.1
    target_update = 1000

    # Initialize frame stack
    frame_stack = deque(maxlen=4)
    obs, _ = env.reset()  # unpack obs and ignore info
    obs = preprocess(obs)

    for _ in range(4):
        frame_stack.append(obs)
    state = np.stack(frame_stack, axis=0)

    for step in range(1, 1000001):
        difficulty = scheduler.get_difficulty(step)
        # Placeholder for dynamic difficulty logic (you can apply it here)

        # ε-greedy action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(np.expand_dims(state, axis=0)).float()
                q_values = policy_net(state_tensor)
                action = q_values.argmax(dim=1).item()

        next_obs, reward, done, _ = env.step(action)
        next_obs = preprocess(next_obs)
        frame_stack.append(next_obs)
        next_state = np.stack(frame_stack, axis=0)

        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

        if done:
            obs, _ = env.reset()
            obs = preprocess(obs)
            for _ in range(4):
                frame_stack.append(obs)
            state = np.stack(frame_stack, axis=0)

        # Training
        if len(replay_buffer) > batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            states_tensor = torch.from_numpy(np.array(states)).float()
            actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states_tensor = torch.from_numpy(np.array(next_states)).float()
            dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

            current_q_values = policy_net(states_tensor).gather(1, actions_tensor)
            next_q_values = target_net(next_states_tensor).max(1)[0].detach().unsqueeze(1)
            expected_q_values = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)

            loss = F.mse_loss(current_q_values, expected_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target network
        if step % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Save model periodically
        if step % 10000 == 0:
            torch.save(policy_net.state_dict(), f'dqn_model_step_{step}.pth')
            print(f"Step {step}: Model checkpoint saved.")

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    env.close()
    torch.save(policy_net.state_dict(), 'dqn_model_final.pth')
    print("Training complete. Final model saved.")

if __name__ == "__main__":
    train()
