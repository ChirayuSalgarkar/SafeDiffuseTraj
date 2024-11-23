import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import time
from abc import ABC, abstractmethod
import shapely
from functools import singledispatch

@dataclass
class GameState:
    position: np.ndarray  # p ∈ ℝ²
    velocity: np.ndarray  # v ∈ ℝ²
    apple: shapely.Geometry
    score: int
    steps_remaining: int
    is_terminal: bool

xmin, xmax, ymin, ymax = 0, 1, 2, 3
apple_radius = .5

@singledispatch
def plot_shape(shape, ax, *args, **kwargs):
    raise NotImplementedError(f"No way to plot {type(shape)}")

@plot_shape.register
def _(collection: shapely.GeometryCollection, ax, *args, **kwargs):
    for shape in collection.geoms:
        plot_shape(shape, ax, *args, **kwargs)

@plot_shape.register
def _(polygon: shapely.Polygon, ax, *args, **kwargs):
    coords = np.array(polygon.exterior.xy)
    ax.add_patch(patch.Polygon(coords.T, *args, **kwargs))

@plot_shape.register
def _(line: shapely.LinearRing | shapely.LineString, ax, *args, **kwargs):
    print(line.coords.xy)
    x,y = line.coords.xy
    ax.plot(x, y, *args, **kwargs)

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
        self.obstacles = shapely.union_all([shapely.Point(center).buffer(distance) for
                           center,distance in obstacles] +
                          [shapely.LinearRing(( (bounds[xmin], bounds[ymin]),
                                     (bounds[xmin], bounds[ymax]),
                                     (bounds[xmax], bounds[ymax]),
                                     (bounds[xmax], bounds[ymin]) ))])
        self.bounds = bounds
        self.max_steps = max_steps
        self.reset()

    def generate_apple(self) -> np.ndarray:
        """Generate a new apple location that doesn't intersect with obstacles."""
        while True:
            apple_center = np.array([
                np.random.uniform(self.bounds[xmin], self.bounds[xmax]),
                np.random.uniform(self.bounds[ymin], self.bounds[ymax])
            ])
            apple = shapely.Point(apple_center).buffer(apple_radius)
            if not self.obstacles.intersects(apple):
                return apple

    def reset(self) -> GameState:
        """Reset the game to initial state and return it."""
        self.path_history = []
        self.state = GameState(
            position=self.initial_position.copy(),
            velocity=self.initial_velocity.copy(),
            apple=self.generate_apple(),
            score=0,
            steps_remaining=self.max_steps,
            is_terminal=False
        )
        self.path_history.append(self.initial_position.copy())
        return self.state

    def step(self, action: np.ndarray) -> Tuple[GameState, float, bool]:
        """Take a step in the environment."""
        if self.state.is_terminal:
            return self.state, 0.0, True

        # next_position, next_velocity = self.f((self.state.position, self.state.velocity), action)
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

        vec_line = shapely.LineString((self.state.position, next_position))

        # Check for collisions with obstacles
        if self.obstacles.intersects(vec_line):
            self.state.is_terminal = True
            print("\nGame Over! Hit an obstacle!")
            return self.state, -1.0, True
            
        # Check if we collected an apple
        reward = 0.0
        if self.state.apple.intersects(vec_line):
            reward = 1.0
            next_apple = self.generate_apple()
            self.path_history = []  # Clear trail when apple is collected
        else:
            next_apple = self.state.apple

        # Store the new position in path history
        self.path_history.append(next_position.copy())
        # Keep only last 50 positions
        self.path_history = self.path_history[-50:]

        # Update state
        self.state = GameState(
            position=next_position,
            velocity=next_velocity,
            apple=next_apple,
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

    def render(self, ax) -> None:
        """Render the current state of the game using matplotlib."""
        ax.cla()

        # Plot trail
        if len(self.path_history) > 1:
            path = np.array(self.path_history)
            ax.plot(path[:, 0], path[:, 1], 'b-', alpha=0.3, linewidth=2)

        plot_shape(self.obstacles, ax, color='grey')

        plot_shape(self.state.apple, ax, color='red')

        # Plot agent
        ax.scatter(self.state.position[0], self.state.position[1], color='blue', s=100)

        # Plot velocity vector
        ax.arrow(
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
        ax.set_xlim(self.bounds[xmin], self.bounds[xmax])
        ax.set_ylim(self.bounds[ymin], self.bounds[ymax])
        ax.set_aspect('equal')

        # Add score and steps remaining
        ax.set_title(f'Score: {self.state.score} | Steps: {self.state.steps_remaining}')

        ax.redraw_in_frame()

def main():
    # Initialize matplotlib
    plt.ion()
    fig, ax = plt.subplots()

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
            game.render(ax)
            plt.pause(0.05)

            if reward > 0:
                print(f"Apple collected! Score: {state.score}")

            if state.is_terminal:
                print(f"\nGame Over! Final Score: {state.score}")
    except KeyboardInterrupt:
        print("\nGame stopped by user")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
