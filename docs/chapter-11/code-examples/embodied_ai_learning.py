#!/usr/bin/env python3

"""
Embodied AI Learning System for Humanoid Robots

This script demonstrates an embodied AI learning system for humanoid robots
that learns from physical interaction with the environment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class HumanoidEmbodiedAI(nn.Module):
    """
    Embodied AI system for humanoid robots that learns from physical interaction
    """
    def __init__(self, state_dim=56, action_dim=28, hidden_dim=256):
        super(HumanoidEmbodiedAI, self).__init__()

        # Perception network (processes sensory input)
        self.perception_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # Motor control network (generates actions)
        self.motor_net = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Cognitive network (higher-level decision making)
        self.cognitive_net = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 10)  # 10 high-level goals
        )

        # Memory system for learning from experience
        self.memory_size = 10000
        self.memory = deque(maxlen=self.memory_size)

        # Action selection network
        self.action_selector = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 10, hidden_dim),  # +10 for goals
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state, goal=None):
        """Forward pass through the embodied AI system"""
        # Process sensory state through perception network
        perception_features = self.perception_net(state)

        # Generate motor commands
        motor_commands = self.motor_net(perception_features)

        # Generate high-level goals (if not provided)
        if goal is None:
            goals = self.cognitive_net(perception_features)
        else:
            goals = goal

        # Combine perception and goals for final action
        combined_input = torch.cat([perception_features, goals], dim=-1)
        final_action = self.action_selector(combined_input)

        return final_action, goals, perception_features

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory for learning"""
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample_batch(self, batch_size=32):
        """Sample a batch of experiences from memory"""
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)
        states = torch.stack([e[0] for e in batch])
        actions = torch.stack([e[1] for e in batch])
        rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32)
        next_states = torch.stack([e[3] for e in batch])
        dones = torch.tensor([e[4] for e in batch], dtype=torch.float32)

        return states, actions, rewards, next_states, dones


class HumanoidLearningSystem:
    """
    Learning system for humanoid robots with embodied AI
    """
    def __init__(self, state_dim=56, action_dim=28):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize embodied AI model
        self.embodied_ai = HumanoidEmbodiedAI(state_dim, action_dim)
        self.optimizer = optim.Adam(self.embodied_ai.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()

        # Learning parameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Experience replay
        self.batch_size = 32
        self.update_target_freq = 100
        self.steps = 0

    def select_action(self, state, goal=None, add_noise=True):
        """Select action using the embodied AI system"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action, goals, _ = self.embodied_ai(state_tensor, goal)

        # Add exploration noise
        if add_noise and random.random() < self.epsilon:
            noise = torch.randn_like(action) * 0.1
            action += noise

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action.squeeze(0).numpy(), goals.squeeze(0).numpy()

    def train_step(self):
        """Perform a training step"""
        if len(self.embodied_ai.memory) < self.batch_size:
            return 0.0

        # Sample batch from memory
        batch = self.embodied_ai.sample_batch(self.batch_size)
        if batch is None:
            return 0.0

        states, actions, rewards, next_states, dones = batch

        # Compute target Q-values
        with torch.no_grad():
            next_actions, _, _ = self.embodied_ai(next_states)
            target_q_values = rewards + (1 - dones) * self.gamma * next_actions.sum(dim=1)

        # Compute current Q-values
        current_actions, _, _ = self.embodied_ai(states)
        current_q_values = current_actions.sum(dim=1)

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1

        return loss.item()

    def learn_from_interaction(self, num_episodes=1000):
        """Learn from physical interaction with the environment"""
        total_rewards = []

        for episode in range(num_episodes):
            # Initialize episode (in real implementation, this would reset the robot/environment)
            state = np.random.randn(self.state_dim)  # Simulated initial state
            total_reward = 0
            done = False
            step = 0

            while not done and step < 100:  # Max 100 steps per episode
                # Select action
                action, goal = self.select_action(state)

                # Simulate environment step (in real implementation, this would be physical interaction)
                next_state = state + 0.1 * action[:self.state_dim] + np.random.randn(self.state_dim) * 0.01
                reward = self.compute_reward(state, action, next_state)  # Compute reward
                done = self.check_termination(next_state, step)  # Check if episode should end

                # Store experience
                self.embodied_ai.remember(
                    torch.FloatTensor(state),
                    torch.FloatTensor(action),
                    reward,
                    torch.FloatTensor(next_state),
                    done
                )

                # Perform training step
                loss = self.train_step()

                # Update state and accumulate reward
                state = next_state
                total_reward += reward
                step += 1

            total_rewards.append(total_reward)

            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")

        return total_rewards

    def compute_reward(self, state, action, next_state):
        """Compute reward for the embodied AI system"""
        # In real implementation, this would be based on task success
        # For this example, we'll use a simple reward function
        reward = 0.0

        # Encourage stability (small joint velocities)
        reward -= 0.1 * np.sum(action**2)

        # Encourage balance (keep center of mass stable)
        com_stability = np.abs(next_state[0:2]).mean()  # First 2 dimensions for CoM x,y
        reward -= 0.5 * com_stability

        # Encourage forward progress (if applicable)
        reward += 0.1 * next_state[0]  # Encourage positive x movement

        return reward

    def check_termination(self, state, step):
        """Check if the episode should terminate"""
        # In real implementation, this would check for falls, etc.
        # For this example, terminate if CoM is too unstable
        com_deviation = np.abs(state[0:2]).max()  # Check CoM x,y position
        return com_deviation > 1.0 or step > 100  # Terminate if CoM too far or max steps reached


# Example usage
def main():
    print("Initializing Humanoid Embodied AI Learning System...")

    # Initialize learning system
    learning_system = HumanoidLearningSystem(state_dim=56, action_dim=28)

    print("Starting learning from interaction...")
    rewards = learning_system.learn_from_interaction(num_episodes=500)

    print(f"Learning completed. Final average reward: {np.mean(rewards[-50:]):.2f}")

    # Test the learned policy
    print("\nTesting learned policy...")
    test_state = np.random.randn(56)
    action, goal = learning_system.select_action(test_state, add_noise=False)
    print(f"Test action: {action[:5]}...")  # Show first 5 action components
    print(f"Learned goal: {goal[:5]}...")   # Show first 5 goal components


if __name__ == "__main__":
    main()