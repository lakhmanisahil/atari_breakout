# Adaptive RL Agent for Atari Breakout with Dynamic Difficulty


## Setup Instructions

1. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
* first clone this repo or download the files

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run train.py for training:**
   ```bash
   python train.py
   ```

3. **Run evaluation:**
   ```bash
   python evaluate.py
   ```

## Project Structure
- `models/dqn_model.py`: CNN model for DQN.
- `env/custom_breakout_env.py`: Environment wrapper with dynamic difficulty.
- `train.py`: Training script for DQN models.
- `evaluate.py`: Evaluation script for trained DQN models.
- `requirements.txt`: Dependencies.

## Notes
- Make sure to download or train a model and place it as `dqn_model_step_1000000.pth` or similar.
- You can modify the environment parameters and difficulty in `custom_breakout_env.py`.
