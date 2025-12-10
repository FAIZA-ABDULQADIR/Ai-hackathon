#!/usr/bin/env python3

"""
Isaac Sim Path Planning

This script demonstrates path planning algorithms in Isaac Sim,
including A* algorithm implementation and path visualization.
"""

import numpy as np
import heapq
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
import matplotlib.pyplot as plt
from typing import List, Tuple
import time


class GridMap:
    """Simple grid map for path planning"""
    def __init__(self, width: int, height: int, resolution: float = 0.5):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid = np.zeros((height, width), dtype=np.uint8)  # 0 = free, 1 = obstacle

    def set_obstacle(self, x: int, y: int):
        """Set a cell as an obstacle"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 1

    def is_free(self, x: int, y: int) -> bool:
        """Check if a cell is free"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y, x] == 0
        return False

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cells"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if self.is_free(nx, ny):
                # Calculate cost based on movement (diagonal vs orthogonal)
                cost = 1.414 if abs(dx) == 1 and abs(dy) == 1 else 1.0
                neighbors.append((nx, ny, cost))
        return neighbors

    def world_to_grid(self, x_world: float, y_world: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x_world + (self.width * self.resolution / 2)) / self.resolution)
        grid_y = int((y_world + (self.height * self.resolution / 2)) / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, x_grid: int, y_grid: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        world_x = x_grid * self.resolution - (self.width * self.resolution / 2)
        world_y = y_grid * self.resolution - (self.height * self.resolution / 2)
        return world_x, world_y


class AStarPlanner:
    """A* path planning algorithm implementation"""
    def __init__(self, grid_map: GridMap):
        self.grid_map = grid_map

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Calculate heuristic distance (Euclidean distance)"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Plan a path using A* algorithm"""
        # Priority queue: (f_score, g_score, position)
        open_set = [(0, 0, start[0], start[1])]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current_f, current_g, current_x, current_y = heapq.heappop(open_set)
            current = (current_x, current_y)

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Reverse to get start-to-goal path

            for neighbor_x, neighbor_y, move_cost in self.grid_map.get_neighbors(current_x, current_y):
                neighbor = (neighbor_x, neighbor_y)
                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor_x, neighbor_y))

        return []  # No path found


class IsaacSimPathPlanner:
    """
    Path planning integration with Isaac Sim
    """
    def __init__(self):
        # Initialize Isaac Sim world
        self.world = World(stage_units_in_meters=1.0)
        self.grid_map = None
        self.planner = None

    def setup_environment(self):
        """Setup environment for path planning"""
        print("Setting up Isaac Sim environment for path planning...")

        # Get assets root path
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("Could not find Isaac Sim assets. Using mock environment.")
            # Create a simple grid map without Isaac Sim assets
            self.grid_map = GridMap(20, 20, resolution=0.5)
        else:
            # Add a simple room environment
            add_reference_to_stage(
                usd_path=f"{assets_root_path}/Isaac/Environments/Simple_Room.usd",
                prim_path="/World/Room"
            )

            # Create a grid map based on the environment
            self.grid_map = GridMap(20, 20, resolution=0.5)

        # Add some obstacles to the grid map (simulating furniture, walls, etc.)
        # Horizontal wall
        for i in range(5, 15):
            self.grid_map.set_obstacle(i, 10)
        # Vertical wall
        for i in range(5, 10):
            self.grid_map.set_obstacle(15, i)
        # Square obstacle
        for i in range(3):
            for j in range(3):
                self.grid_map.set_obstacle(7+i, 7+j)

        # Initialize path planner
        self.planner = AStarPlanner(self.grid_map)

        print("Environment setup completed.")

    def plan_and_execute_path(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Plan and execute a path from start to goal in world coordinates"""
        print(f"\nPlanning path from {start_pos} to {goal_pos}")

        # Convert world coordinates to grid coordinates
        start_grid = self.grid_map.world_to_grid(start_pos[0], start_pos[1])
        goal_grid = self.grid_map.world_to_grid(goal_pos[0], goal_pos[1])

        print(f"Grid coordinates - Start: {start_grid}, Goal: {goal_grid}")

        # Plan the path
        start_time = time.time()
        path = self.planner.plan_path(start_grid, goal_grid)
        planning_time = time.time() - start_time

        if path:
            print(f"Path found with {len(path)} waypoints in {planning_time:.4f}s")

            # Convert path to world coordinates
            world_path = []
            for x, y in path:
                world_x, world_y = self.grid_map.grid_to_world(x, y)
                world_path.append((world_x, world_y))

            # Print path statistics
            total_distance = 0
            for i in range(1, len(world_path)):
                dx = world_path[i][0] - world_path[i-1][0]
                dy = world_path[i][1] - world_path[i-1][1]
                total_distance += np.sqrt(dx**2 + dy**2)

            print(f"Total path distance: {total_distance:.2f}m")
            print(f"Average step size: {total_distance/len(world_path):.3f}m")

            return world_path
        else:
            print("No path found from start to goal")
            return []

    def visualize_path(self, path: List[Tuple[float, float]], start: Tuple[float, float], goal: Tuple[float, float]):
        """Visualize the planned path"""
        if not path:
            print("No path to visualize")
            return

        # Create a visualization grid
        vis_grid = self.grid_map.grid.copy().astype(float)

        # Convert path to grid coordinates and mark on grid
        for world_x, world_y in path:
            grid_x, grid_y = self.grid_map.world_to_grid(world_x, world_y)
            if 0 <= grid_x < self.grid_map.width and 0 <= grid_y < self.grid_map.height:
                vis_grid[grid_y, grid_x] = 0.5  # Path in gray

        # Mark start and goal in grid coordinates
        start_grid = self.grid_map.world_to_grid(start[0], start[1])
        goal_grid = self.grid_map.world_to_grid(goal[0], goal[1])

        if 0 <= start_grid[0] < self.grid_map.width and 0 <= start_grid[1] < self.grid_map.height:
            vis_grid[start_grid[1], start_grid[0]] = 0.2  # Start in light gray
        if 0 <= goal_grid[0] < self.grid_map.width and 0 <= goal_grid[1] < self.grid_map.height:
            vis_grid[goal_grid[1], goal_grid[0]] = 0.8  # Goal in dark gray

        # Create the plot
        plt.figure(figsize=(12, 10))
        plt.imshow(vis_grid, cmap='gray', origin='upper',
                  extent=[-self.grid_map.width*self.grid_map.resolution/2,
                         self.grid_map.width*self.grid_map.resolution/2,
                         -self.grid_map.height*self.grid_map.resolution/2,
                         self.grid_map.height*self.grid_map.resolution/2])

        # Plot the path as a line
        if path:
            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]
            plt.plot(path_x, path_y, 'b-', linewidth=2, label='Planned Path')

        # Mark start and goal positions
        plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
        plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')

        plt.title('Isaac Sim Path Planning Visualization')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.colorbar(label='Grid Value (0=free, 1=obstacle, 0.5=path)')
        plt.show()

    def run_path_planning_demo(self):
        """Run a complete path planning demonstration"""
        print("Starting Isaac Sim Path Planning Demo...")

        # Initialize the world
        self.world.reset()

        # Setup environment
        self.setup_environment()

        # Define multiple path planning scenarios
        scenarios = [
            {"start": (-4.0, -4.0), "goal": (4.0, 4.0), "name": "Corner to corner"},
            {"start": (0.0, -4.0), "goal": (0.0, 4.0), "name": "Bottom to top"},
            {"start": (-3.0, 0.0), "goal": (3.0, 0.0), "name": "Left to right"}
        ]

        all_paths = []

        for i, scenario in enumerate(scenarios):
            print(f"\n--- Scenario {i+1}: {scenario['name']} ---")
            path = self.plan_and_execute_path(scenario["start"], scenario["goal"])
            if path:
                all_paths.append((path, scenario["start"], scenario["goal"]))
                print(f"Scenario {i+1} completed successfully")

        # Visualize the last path
        if all_paths:
            last_path, last_start, last_goal = all_paths[-1]
            self.visualize_path(last_path, last_start, last_goal)

        print(f"\nPath planning demo completed. Processed {len(scenarios)} scenarios.")

    def cleanup(self):
        """Clean up resources"""
        self.world.clear()


def main():
    """Main function to run the path planning demo"""
    print("Initializing Isaac Sim Path Planning Demo...")

    # Create path planner instance
    path_planner = IsaacSimPathPlanner()

    try:
        # Run the complete path planning demo
        path_planner.run_path_planning_demo()

    except KeyboardInterrupt:
        print("\nPath planning demo interrupted by user.")
    except Exception as e:
        print(f"\nError during path planning: {e}")
    finally:
        # Clean up
        path_planner.cleanup()
        print("Path planning demo cleanup completed.")


if __name__ == "__main__":
    main()