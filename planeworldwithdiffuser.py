import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List

class DiffusionModel:
    def __init__(self, n_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.n_steps = n_steps
        self.beta = torch.linspace(beta_start, beta_end, n_steps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion process"""
        noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bar[t]
        mean = torch.sqrt(alpha_bar_t)[:, None] * x_0
        var = 1 - alpha_bar_t[:, None]
        return mean + torch.sqrt(var) * noise, noise

class TrajectoryEncoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # position (2) + velocity (2)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class StatePredictor(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for time embedding
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # predict position (2) + velocity (2)
        )
        
    def forward(self, x, t):
        t_embed = t.unsqueeze(-1) / 1000.  # Simple time embedding
        x_t = torch.cat([x, t_embed], dim=-1)
        return self.net(x_t)

class DiffusionGameModel:
    def __init__(self, game: PhysicsGame):
        self.game = game
        self.diffusion = DiffusionModel()
        self.encoder = TrajectoryEncoder()
        self.predictor = StatePredictor()
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.predictor.parameters()), 
            lr=1e-4
        )

    def collect_trajectory(self, actions: List[np.ndarray], max_steps=50) -> torch.Tensor:
        state = self.game.reset()
        states = []
        
        for action in actions[:max_steps]:
            state, _, done = self.game.step(action)
            if done:
                break
            states.append(np.concatenate([state.position, state.velocity]))
            
        return torch.tensor(states, dtype=torch.float32)

    def train_step(self, trajectory: torch.Tensor) -> float:
        encoded = self.encoder(trajectory)
        
        # Sample timestep
        t = torch.randint(0, self.diffusion.n_steps, (encoded.shape[0],))
        
        # Add noise
        noised_encoded, noise = self.diffusion.q_sample(encoded, t)
        
        # Predict and compute loss
        pred_noise = self.predictor(noised_encoded, t)
        loss = nn.MSELoss()(pred_noise, noise)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def generate_trajectory(self, n_steps=50) -> torch.Tensor:
        with torch.no_grad():
            # Start from random noise
            x = torch.randn(n_steps, 128)
            
            # Reverse diffusion process
            for t in reversed(range(self.diffusion.n_steps)):
                t_batch = torch.full((n_steps,), t)
                pred = self.predictor(x, t_batch)
                
                # Update using predicted noise
                alpha = self.diffusion.alpha[t]
                alpha_bar = self.diffusion.alpha_bar[t]
                beta = self.diffusion.beta[t]
                
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = 0
                
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred)
                if t > 0:
                    x = x + torch.sqrt(beta) * noise
            
            # Decode trajectory
            return self.encoder(x)

def train_diffusion(game: PhysicsGame, n_episodes=1000):
    model = DiffusionGameModel(game)
    
    for episode in range(n_episodes):
        # Generate random actions for training
        actions = [np.random.randn(2) for _ in range(50)]
        trajectory = model.collect_trajectory(actions)
        
        loss = model.train_step(trajectory)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Loss: {loss:.6f}")
    
    return model
