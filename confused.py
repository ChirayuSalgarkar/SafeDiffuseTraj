import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class Trajectory:
    states: torch.Tensor      # Shape: (batch, horizon, state_dim)
    actions: torch.Tensor     # Shape: (batch, horizon, action_dim)
    rewards: torch.Tensor     # Shape: (batch, horizon)
    dones: torch.Tensor       # Shape: (batch, horizon)

class PhysicsGameWorld(nn.Module):
    """World model for the physics game using diffusion."""
    def __init__(
        self,
        state_dim: int = 8,   # position(2) + velocity(2) + apple(2) + closest_obstacles(2)
        action_dim: int = 2,
        horizon: int = 10,
        n_timesteps: int = 100,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_timesteps = n_timesteps
        
        # Diffusion model components
        hidden_dim = 256
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Initialize diffusion schedule
        self.betas = torch.linspace(1e-4, 2e-2, n_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def diffusion_step(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Single step of the diffusion process."""
        timestep_embed = t.view(-1, 1)
        input_x = torch.cat([x, actions, timestep_embed], dim=-1)
        return self.net(input_x)
        
    def sample(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample future trajectories given initial state and actions."""
        batch_size = actions.shape[0]
        device = actions.device
        
        # Initialize trajectory with noise
        x = torch.randn(batch_size, self.horizon, self.state_dim, device=device)
        x[:, 0] = initial_state  # Set initial state
        
        # Iterative denoising
        for t in reversed(range(self.n_timesteps)):
            timesteps = torch.full((batch_size,), t, device=device)
            with torch.no_grad():
                pred = self.diffusion_step(x, actions, timesteps)
                alpha = self.alphas[t]
                alpha_cumprod = self.alphas_cumprod[t]
                beta = self.betas[t]
                
                # Update x using the diffusion equation
                noise = torch.randn_like(x) * temperature if t > 0 else 0
                x = (1 / torch.sqrt(alpha)) * (
                    x - (beta / (torch.sqrt(1 - alpha_cumprod))) * pred
                ) + torch.sqrt(beta) * noise
                
        return x

class PhysicsGamePwD:
    """Physics Game environment with Planning with Diffusion support."""
    def __init__(self):
        self.world_model = PhysicsGameWorld()
        self.replay_buffer = []
        
    def collect_experience(self, env, policy, n_episodes: int) -> List[Trajectory]:
        """Collect experience for training the world model."""
        trajectories = []
        
        for _ in range(n_episodes):
            states, actions, rewards, dones = [], [], [], []
            obs, _ = env.reset()
            done = False
            
            while not done:
                action = policy(obs)  # Your policy here
                next_obs, reward, terminated, truncated, _ = env.step(action)
                
                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                dones.append(terminated or truncated)
                
                obs = next_obs
                done = terminated or truncated
            
            trajectory = Trajectory(
                states=torch.tensor(states),
                actions=torch.tensor(actions),
                rewards=torch.tensor(rewards),
                dones=torch.tensor(dones)
            )
            trajectories.append(trajectory)
            
        return trajectories

    def train_world_model(self, trajectories: List[Trajectory], n_epochs: int = 100):
        """Train the diffusion world model on collected trajectories."""
        optimizer = torch.optim.Adam(self.world_model.parameters(), lr=1e-4)
        
        for _ in range(n_epochs):
            for traj in trajectories:
                # Sample timestep
                t = torch.randint(0, self.world_model.n_timesteps, (1,))
                
                # Forward pass
                pred = self.world_model.diffusion_step(
                    traj.states, traj.actions, t
                )
                
                # Compute loss
                noise = torch.randn_like(traj.states)
                alpha_cumprod = self.world_model.alphas_cumprod[t]
                noisy_states = torch.sqrt(alpha_cumprod) * traj.states + \
                             torch.sqrt(1 - alpha_cumprod) * noise
                
                loss = nn.MSELoss()(pred, noise)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def plan(
        self,
        initial_state: torch.Tensor,
        n_samples: int = 100,
        horizon: int = 10,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Plan actions using diffusion sampling."""
        # Sample random action sequences
        actions = torch.randn(n_samples, horizon, self.world_model.action_dim)
        
        # Sample trajectories using the world model
        trajectories = self.world_model.sample(
            initial_state, actions, temperature
        )
        
        # Score trajectories (example: sum of predicted rewards)
        scores = self._score_trajectories(trajectories)
        
        # Return best action sequence
        best_idx = torch.argmax(scores)
        return actions[best_idx]

    def _score_trajectories(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Score predicted trajectories based on reward/success metrics."""
        # Example scoring function - can be modified based on task
        apple_collection = self._predict_apple_collections(trajectories)
        obstacle_avoidance = self._predict_obstacle_avoidance(trajectories)
        return apple_collection + obstacle_avoidance
