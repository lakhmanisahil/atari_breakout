import torch
import numpy as np
import cv2
from collections import deque
from models.dqn_model import DQN
from env.custom_breakout_env import CustomBreakoutEnv

def preprocess(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84))
    obs = obs / 255.0
    return obs

def evaluate(model_path="dqn_model_step_110000.pth", episodes=5):
    env = CustomBreakoutEnv(render_mode="human")  # Ensure proper rendering
    num_actions = env.action_space.n
    input_shape = (4, 84, 84)

    policy_net = DQN(input_shape, num_actions)
    policy_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    policy_net.eval()

    for ep in range(episodes):
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        obs = preprocess(obs)
        frame_stack = deque([obs] * 4, maxlen=4)

        done = False
        total_reward = 0

        while not done:
            env.render()

            state = np.array(frame_stack)
            state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32)

            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = q_values.argmax(dim=1).item()

            step_result = env.step(action)
            next_obs = step_result[0] if isinstance(step_result, tuple) else step_result
            reward = step_result[1]
            done = step_result[2]
            # info = step_result[3]  # Not used but available if needed

            next_obs = preprocess(next_obs)
            frame_stack.append(next_obs)

            total_reward += reward

        print(f"Episode {ep + 1} — Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    evaluate()
