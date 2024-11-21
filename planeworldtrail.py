import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import time

@dataclass
class GameState:
    position: np.ndarray  # p ∈ ℝ²
    velocity: np.ndarray  # v ∈ ℝ²
    apple_location: np.ndarray  # l ∈ ℝ²
    score: int
    steps_remaining: int
    is_terminal: bool

class PhysicsGame:
    def __init__(
        self,
        initial_position: np.ndarray,
        initial_velocity: np.ndarray,
        obstacles: List[Tuple[np.ndarray, float]],  # List of (center, radius) for circular obstacles
        bounds: Tuple[float, float, float, float],  # (xmin, xmax, ymin, ymax)
        max_steps: int = 1000
    ):
        self.initial_position = initial_position
        self.initial_velocity = initial_velocity
        self.obstacles = obstacles
        self.bounds = bounds
        self.max_steps = max_steps
        self.path_history = []
        self.reset()
    
    def generate_apple(self) -> np.ndarray:
        """Generate a new apple location that doesn't intersect with obstacles."""
        while True:
            apple = np.array([
                np.random.uniform(self.bounds[0], self.bounds[1]),
                np.random.uniform(self.bounds[2], self.bounds[3])
            ])
            if not self.intersects_obstacles(apple, apple):
                return apple
    
    def reset(self) -> GameState:
        """Reset the game to initial state and return it."""
        self.path_history = []
        self.state = GameState(
            position=self.initial_position.copy(),
            velocity=self.initial_velocity.copy(),
            apple_location=self.generate_apple(),
            score=0,
            steps_remaining=self.max_steps,
            is_terminal=False
        )
        self.path_history.append(self.initial_position.copy())
        return self.state
    
    def line_intersects_circle(
        self,
        start: np.ndarray,
        end: np.ndarray,
        center: np.ndarray,
        radius: float
    ) -> bool:
        """Check if line segment intersects with circle using vector algebra."""
        line_vec = end - start
        center_to_start = start - center
        
        a = np.dot(line_vec, line_vec)
        b = 2 * np.dot(center_to_start, line_vec)
        c = np.dot(center_to_start, center_to_start) - radius * radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return False
            
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        
        return (0 <= t1 <= 1) or (0 <= t2 <= 1)
    
    def intersects_obstacles(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Check if line segment intersects with any obstacle."""
        for center, radius in self.obstacles:
            if self.line_intersects_circle(start, end, center, radius):
                return True
        return False
    
    def crosses_apple(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Check if line segment crosses near the apple."""
        apple_radius = 0.5  # Defines how close we need to get to "collect" the apple
        return self.line_intersects_circle(start, end, self.state.apple_location, apple_radius)
    
    def step(self, action: np.ndarray) -> Tuple[GameState, float, bool]:
        """Take a step in the environment."""
        if self.state.is_terminal:
            return self.state, 0.0, True
            
        # Scale down the action force
        action = action * 0.1
        
        # Add drag to velocity
        drag_factor = 0.95
        self.state.velocity *= drag_factor
        
        # Calculate next position and velocity
        next_velocity = self.state.velocity + action
        
        # Optional: Cap maximum velocity
        max_speed = 0.5
        speed = np.linalg.norm(next_velocity)
        if speed > max_speed:
            next_velocity = (next_velocity / speed) * max_speed
            
        next_position = self.state.position + next_velocity
        
        # Check for boundary collisions
        if (next_position[0] < self.bounds[0] or 
            next_position[0] > self.bounds[1] or 
            next_position[1] < self.bounds[2] or 
            next_position[1] > self.bounds[3]):
            self.state.is_terminal = True
            print("\nGame Over! Hit the boundary!")
            return self.state, -1.0, True
        
        # Check for collisions with obstacles
        if self.intersects_obstacles(self.state.position, next_position):
            self.state.is_terminal = True
            print("\nGame Over! Hit an obstacle!")
            return self.state, -1.0, True
            
        # Check if we collected an apple
        reward = 0.0
        if self.crosses_apple(self.state.position, next_position):
            reward = 1.0
            next_apple = self.generate_apple()
            self.path_history = []  # Clear trail when apple is collected
        else:
            next_apple = self.state.apple_location
            
        # Store the new position in path history
        self.path_history.append(next_position.copy())
        # Keep only last 50 positions
        self.path_history = self.path_history[-50:]
            
        # Update state
        self.state = GameState(
            position=next_position,
            velocity=next_velocity,
            apple_location=next_apple,
            score=self.state.score + (1 if reward > 0 else 0),
            steps_remaining=self.state.steps_remaining - 1,
            is_terminal=False
        )
        
        # Check if we're out of steps
        done = self.state.steps_remaining <= 0
        if done:
            self.state.is_terminal = True
            print("\nGame Over! Out of steps!")
            
        return self.state, reward, done
    
    def render(self) -> None:
        """Render the current state of the game using matplotlib."""
        plt.clf()
        
        # Plot trail
        if len(self.path_history) > 1:
            path = np.array(self.path_history)
            plt.plot(path[:, 0], path[:, 1], 'b-', alpha=0.3, linewidth=2)
        
        # Plot obstacles
        for center, radius in self.obstacles:
            circle = plt.Circle(center, radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
            
        # Plot apple
        apple = plt.Circle(self.state.apple_location, 0.5, color='red', alpha=0.7)
        plt.gca().add_patch(apple)
        
        # Plot agent
        plt.scatter(self.state.position[0], self.state.position[1], color='blue', s=100)
        
        # Plot velocity vector
        plt.arrow(
            self.state.position[0],
            self.state.position[1],
            self.state.velocity[0],
            self.state.velocity[1],
            color='blue',
            alpha=0.5,
            width=0.1,
            head_width=0.3
        )
        
        # Set bounds
        plt.xlim(self.bounds[0], self.bounds[1])
        plt.ylim(self.bounds[2], self.bounds[3])
        plt.gca().set_aspect('equal')
        
        # Add score and steps remaining
        plt.title(f'Score: {self.state.score} | Steps: {self.state.steps_remaining}')
        
        plt.draw()
        plt.pause(0.1)

def main():
    # Initialize matplotlib
    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    
    # Create game instance
    game = PhysicsGame(
        initial_position=np.array([0.0, 0.0]),
        initial_velocity=np.array([0.0, 0.0]),
        obstacles=[
            (np.array([5.0, 5.0]), 1.0),
            (np.array([-3.0, 4.0]), 1.5),
            (np.array([4.0, -3.0]), 1.2),
            (np.array([-5.0, -5.0]), 1.3),
        ],
        bounds=(-10, 10, -10, 10)
    )
    
    # Reset game to start
    state = game.reset()
    
    # Initialize action
    current_action = np.zeros(2)
    
    def on_key_press(event):
        nonlocal current_action
        if event.key == 'up':
            current_action = np.array([0.0, 1.0])
        elif event.key == 'down':
            current_action = np.array([0.0, -1.0])
        elif event.key == 'left':
            current_action = np.array([-1.0, 0.0])
        elif event.key == 'right':
            current_action = np.array([1.0, 0.0])
    
    # Connect the key press event to the figure
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    try:
        while not state.is_terminal:
            action = current_action
            state, reward, done = game.step(action)
            current_action = np.zeros(2)
            game.render()
            plt.pause(0.05)
            
            if reward > 0:
                print(f"Apple collected! Score: {state.score}")
            
            if state.is_terminal:
                print(f"\nGame Over! Final Score: {state.score}")
            
    except KeyboardInterrupt:
        print("\nGame stopped by user")
    finally:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()
