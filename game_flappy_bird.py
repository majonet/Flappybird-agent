import os
import random
import itertools
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
import flappy_bird_gymnasium


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ============================================================
# CONFIGURATION
# ============================================================

EPISODES = 2000

LEARNING_RATE = 0.001
GAMMA = 0.99

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

REPLAY_MEMORY_SIZE = 50_000
BATCH_SIZE = 64

TARGET_UPDATE_FREQUENCY = 1000

LEARNING_START = 1000

MAX_STEPS_PER_EPISODE = 10_000

SAVE_DIR = "runs"

BEST_MODEL_FILE = os.path.join(
    SAVE_DIR,
    "flappybird_best.pt"
)

LAST_MODEL_FILE = os.path.join(
    SAVE_DIR,
    "flappybird_last.pt"
)

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# DQN NETWORK
# ============================================================

class DQN(nn.Module):

    def __init__(self, num_states, num_actions):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(num_states, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# REPLAY MEMORY
# ============================================================

class ReplayMemory:

    def __init__(self, capacity):

        self.memory = deque(
            maxlen=capacity
        )

    def append(
        self,
        state,
        action,
        next_state,
        reward,
        terminated
    ):

        self.memory.append(
            (
                state,
                action,
                next_state,
                reward,
                terminated
            )
        )

    def sample(self, batch_size):

        return random.sample(
            self.memory,
            batch_size
        )

    def __len__(self):

        return len(self.memory)


# ============================================================
# AGENT
# ============================================================

class Agent:

    def __init__(self):

        self.gamma = GAMMA

        self.epsilon = EPSILON_START

        self.memory = ReplayMemory(
            REPLAY_MEMORY_SIZE
        )

        self.optimizer = None

        self.total_steps = 0

        self.best_reward = float("-inf")


    # ========================================================
    # SELECT ACTION
    # ========================================================

    def select_action(
        self,
        state,
        policy_dqn,
        env
    ):

        # Exploration
        if random.random() < self.epsilon:

            action = env.action_space.sample()

            return torch.tensor(
                action,
                dtype=torch.long,
                device=device
            )

        # Exploitation
        with torch.no_grad():

            q_values = policy_dqn(
                state.unsqueeze(0)
            )

            action = q_values.argmax(
                dim=1
            ).item()

        return torch.tensor(
            action,
            dtype=torch.long,
            device=device
        )


    # ========================================================
    # OPTIMIZE
    # ========================================================

    def optimize(
        self,
        policy_dqn,
        target_dqn
    ):

        if len(self.memory) < BATCH_SIZE:
            return None

        batch = self.memory.sample(
            BATCH_SIZE
        )

        states = torch.stack(
            [x[0] for x in batch]
        )

        actions = torch.stack(
            [x[1] for x in batch]
        )

        next_states = torch.stack(
            [x[2] for x in batch]
        )

        rewards = torch.stack(
            [x[3] for x in batch]
        )

        terminated = torch.tensor(
            [x[4] for x in batch],
            dtype=torch.float32,
            device=device
        )

        # ---------------------------------------------
        # Current Q
        # ---------------------------------------------

        current_q = policy_dqn(
            states
        ).gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # ---------------------------------------------
        # Target Q
        # ---------------------------------------------

        with torch.no_grad():

            next_q = target_dqn(
                next_states
            ).max(
                dim=1
            ).values

            target_q = rewards + (
                1 - terminated
            ) * self.gamma * next_q

        # ---------------------------------------------
        # Loss
        # ---------------------------------------------

        loss = nn.SmoothL1Loss()(
            current_q,
            target_q
        )

        # ---------------------------------------------
        # Backpropagation
        # ---------------------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        # Prevent very large gradients
        torch.nn.utils.clip_grad_norm_(
            policy_dqn.parameters(),
            max_norm=10.0
        )

        self.optimizer.step()

        return loss.item()


    # ========================================================
    # SAVE MODEL
    # ========================================================

    def save_model(
        self,
        policy_dqn,
        filename
    ):

        torch.save(
            {
                "model_state_dict":
                    policy_dqn.state_dict(),

                "epsilon":
                    self.epsilon,

                "best_reward":
                    self.best_reward,

                "total_steps":
                    self.total_steps
            },
            filename
        )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(
        self,
        policy_dqn,
        filename
    ):

        if not os.path.exists(filename):

            print(
                "No checkpoint found."
            )

            return False

        checkpoint = torch.load(
            filename,
            map_location=device
        )

        # Support both checkpoint format
        # and normal state_dict format

        if "model_state_dict" in checkpoint:

            policy_dqn.load_state_dict(
                checkpoint["model_state_dict"]
            )

            self.epsilon = checkpoint.get(
                "epsilon",
                EPSILON_START
            )

            self.best_reward = checkpoint.get(
                "best_reward",
                float("-inf")
            )

            self.total_steps = checkpoint.get(
                "total_steps",
                0
            )

        else:

            policy_dqn.load_state_dict(
                checkpoint
            )

        print(
            f"Loaded model: {filename}"
        )

        return True


    # ========================================================
    # TRAIN
    # ========================================================

    def train(
        self,
        episodes=EPISODES,
        resume=False
    ):

        # ----------------------------------------------------
        # Environment
        # ----------------------------------------------------

        env = gym.make(
            "FlappyBird-v0",
            render_mode=None
        )

        num_states = env.observation_space.shape[0]

        num_actions = env.action_space.n

        print(
            "Number of states:",
            num_states
        )

        print(
            "Number of actions:",
            num_actions
        )

        # ----------------------------------------------------
        # Networks
        # ----------------------------------------------------

        policy_dqn = DQN(
            num_states,
            num_actions
        ).to(device)

        target_dqn = DQN(
            num_states,
            num_actions
        ).to(device)

        target_dqn.load_state_dict(
            policy_dqn.state_dict()
        )

        target_dqn.eval()

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        self.optimizer = optim.Adam(
            policy_dqn.parameters(),
            lr=LEARNING_RATE
        )

        # ----------------------------------------------------
        # Resume
        # ----------------------------------------------------

        if resume:

            self.load_model(
                policy_dqn,
                LAST_MODEL_FILE
            )

            # Target must also receive
            # loaded policy weights

            target_dqn.load_state_dict(
                policy_dqn.state_dict()
            )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        for episode in range(
            episodes
        ):

            state, info = env.reset()

            state = torch.tensor(
                state,
                dtype=torch.float32,
                device=device
            )

            episode_reward = 0.0

            episode_steps = 0

            terminated = False

            truncated = False

            losses = []

            # ------------------------------------------------
            # Episode
            # ------------------------------------------------

            while (
                not terminated
                and not truncated
                and episode_steps < MAX_STEPS_PER_EPISODE
            ):

                # --------------------------------------------
                # Select action
                # --------------------------------------------

                action = self.select_action(
                    state,
                    policy_dqn,
                    env
                )

                # --------------------------------------------
                # Environment step
                # --------------------------------------------

                next_state, reward, terminated, truncated, info = env.step(
                    action.item()
                )

                next_state = torch.tensor(
                    next_state,
                    dtype=torch.float32,
                    device=device
                )

                reward_tensor = torch.tensor(
                    float(reward),
                    dtype=torch.float32,
                    device=device
                )

                # --------------------------------------------
                # Store experience
                # --------------------------------------------

                self.memory.append(
                    state,
                    action,
                    next_state,
                    reward_tensor,
                    terminated or truncated
                )

                # --------------------------------------------
                # Update state
                # --------------------------------------------

                state = next_state

                episode_reward += float(
                    reward
                )

                episode_steps += 1

                self.total_steps += 1

                # --------------------------------------------
                # Train DQN
                # --------------------------------------------

                if (
                    self.total_steps >= LEARNING_START
                    and len(self.memory) >= BATCH_SIZE
                ):

                    loss = self.optimize(
                        policy_dqn,
                        target_dqn
                    )

                    if loss is not None:
                        losses.append(loss)

                # --------------------------------------------
                # Update target network
                # --------------------------------------------

                if (
                    self.total_steps
                    % TARGET_UPDATE_FREQUENCY
                    == 0
                ):

                    target_dqn.load_state_dict(
                        policy_dqn.state_dict()
                    )

            # ------------------------------------------------
            # Epsilon decay
            # ------------------------------------------------

            self.epsilon = max(
                EPSILON_MIN,
                self.epsilon * EPSILON_DECAY
            )

            # ------------------------------------------------
            # Save last model
            # ------------------------------------------------

            self.save_model(
                policy_dqn,
                LAST_MODEL_FILE
            )

            # ------------------------------------------------
            # Save best model
            # ------------------------------------------------

            if episode_reward > self.best_reward:

                self.best_reward = episode_reward

                self.save_model(
                    policy_dqn,
                    BEST_MODEL_FILE
                )

                best_text = " <-- BEST"

            else:

                best_text = ""

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            if len(losses) > 0:

                avg_loss = np.mean(
                    losses
                )

            else:

                avg_loss = 0.0

            print(
                f"Episode: {episode + 1:5d} | "
                f"Reward: {episode_reward:8.2f} | "
                f"Steps: {episode_steps:5d} | "
                f"Epsilon: {self.epsilon:.4f} | "
                f"Loss: {avg_loss:.5f}"
                f"{best_text}"
            )

        env.close()

        print("\nTraining finished.")

        print(
            "Best reward:",
            self.best_reward
        )

        print(
            "Best model:",
            BEST_MODEL_FILE
        )


    # ========================================================
    # EVALUATE
    # ========================================================

    def evaluate(
        self,
        episodes=20
    ):

        env = gym.make(
            "FlappyBird-v0",
            render_mode="human"
        )

        num_states = (
            env.observation_space.shape[0]
        )

        num_actions = (
            env.action_space.n
        )

        policy_dqn = DQN(
            num_states,
            num_actions
        ).to(device)

        # --------------------------------------------
        # Load BEST model
        # --------------------------------------------

        if not self.load_model(
            policy_dqn,
            BEST_MODEL_FILE
        ):

            print(
                "Best model does not exist."
            )

            env.close()

            return

        policy_dqn.eval()

        rewards = []

        # --------------------------------------------
        # Evaluation episodes
        # --------------------------------------------

        for episode in range(
            episodes
        ):

            state, info = env.reset()

            state = torch.tensor(
                state,
                dtype=torch.float32,
                device=device
            )

            episode_reward = 0.0

            terminated = False

            truncated = False

            steps = 0

            while (
                not terminated
                and not truncated
                and steps < MAX_STEPS_PER_EPISODE
            ):

                with torch.no_grad():

                    q_values = policy_dqn(
                        state.unsqueeze(0)
                    )

                    action = q_values.argmax(
                        dim=1
                    ).item()

                next_state, reward, terminated, truncated, info = env.step(
                    action
                )

                state = torch.tensor(
                    next_state,
                    dtype=torch.float32,
                    device=device
                )

                episode_reward += float(
                    reward
                )

                steps += 1

            rewards.append(
                episode_reward
            )

            print(
                f"Evaluation Episode "
                f"{episode + 1}/{episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Steps: {steps}"
            )

        env.close()

        print(
            "\nAverage evaluation reward:",
            np.mean(rewards)
        )

        print(
            "Maximum evaluation reward:",
            np.max(rewards)
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    agent = Agent()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    agent.train(
        episodes=EPISODES,
        resume=False
    )

    # --------------------------------------------------------
    # AFTER TRAINING:
    # Uncomment this to evaluate the BEST model.
    # --------------------------------------------------------

    # agent.evaluate(episodes=20)
