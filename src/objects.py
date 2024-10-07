from abc import ABC, abstractmethod
import random
import numpy as np

class Particle:
    def __init__(self, position: np.ndarray, velocity: np.ndarray, color: str = 'black', 
                 radius: float = 1.0, density: float = 1.0) -> None:
        """
        Initialize a particle with position, velocity, color, radius, and density.
        
        :param position: A numpy array of shape (2,) representing (x, y) position.
        :param velocity: A numpy array of shape (2,) representing (vx, vy) velocity.
        :param color: String representing the color of the particle.
        :param radius: Float representing the radius of the particle.
        :param density: Float representing the density of the particle.
        """
        self.position = np.array(position)  # (x, y)
        self.velocity = np.array(velocity)  # (vx, vy)
        self.color = color
        self.radius = radius
        self.density = density
        self.mass = self.calculate_mass()
        self.path_history = []  # Stores positions over time

        # Add the initial position to the history
        self.path_history.append(self.position.copy())

    def calculate_mass(self) -> float:
        """
        Calculate the mass of the particle based on its density and radius.
        In 2D, mass is proportional to area (π * r²).
        
        :return: The calculated mass of the particle.
        """
        return np.pi * (self.radius ** 2) * self.density  # In 2D, mass is proportional to area

    def move(self, dt: float) -> None:
        """
        Update the position of the particle based on its velocity and time step.
        
        :param dt: Time step for the movement.
        """
        self.position += self.velocity * dt
        # Record the new position after each move
        self.path_history.append(self.position.copy())


class Scene(ABC):
    def __init__(self, width: float, height: float, cell_size: float) -> None:
        """
        Initialize the abstract scene, containing particles and a spatial grid for efficient collision detection.
        
        :param width: Width of the scene.
        :param height: Height of the scene.
        :param cell_size: Size of each cell in the spatial grid for particle positioning.
        """
        self.particles = []  # List to hold particles in the scene
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        # Calculate the number of cells along width and height
        self.grid_width = int(np.ceil(width / cell_size))
        self.grid_height = int(np.ceil(height / cell_size))
        
        # Initialize the grid as a 2D list of empty lists, each cell containing particle lists
        self.grid = [[[] for _ in range(self.grid_height)] for _ in range(self.grid_width)]
        self.epsilon = 1e-9  # Tolerance to avoid floating-point precision issues


    @abstractmethod
    def add_particle(self, particle: Particle) -> None:
        """
        Abstract method to add a particle to the scene. This method must be implemented in derived classes.
        
        :param particle: Particle to be added to the scene.
        """
        pass

    @abstractmethod
    def check_wall_collision(self, particle: Particle, dt: float) -> None:
        """
        Abstract method to check and resolve wall collisions for a particle. Implemented in derived classes.
        
        :param particle: The particle to check for wall collision.
        :param dt: The time step of the simulation for continuous collision detection.
        """
        pass

    @staticmethod
    def particles_overlap(p1: Particle, p2: Particle) -> bool:
        """
        Check if two particles overlap based on their positions and radii.
        
        :param p1: First particle.
        :param p2: Second particle.
        :return: True if the particles overlap, False otherwise.
        """
        distance = np.linalg.norm(p1.position - p2.position)
        return distance < (p1.radius + p2.radius)

    def get_cell_indices(self, position: np.ndarray) -> (int, int):
        """
        Get the indices of the grid cell corresponding to the particle's position.
        
        :param position: Position of the particle.
        :return: Tuple of grid indices (i, j).
        """
        x, y = position
        i = int(x / self.cell_size)
        j = int(y / self.cell_size)
        # Ensure indices are within bounds
        return max(0, min(i, self.grid_width - 1)), max(0, min(j, self.grid_height - 1))

    def add_to_grid(self, particle: Particle) -> None:
        """
        Add a particle to the spatial grid at the appropriate cell.
        
        :param particle: Particle to be added to the grid.
        """
        i, j = self.get_cell_indices(particle.position)
        self.grid[i][j].append(particle)

    def update_grid(self) -> None:
        """
        Clear and update the grid with the current positions of all particles.
        """
        # Clear the grid
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                self.grid[i][j].clear()

        # Re-add all particles to the grid
        for particle in self.particles:
            self.add_to_grid(particle)

    def get_neighboring_cells(self, i: int, j: int):
        """
        Get the neighboring cells (and the current cell) around grid position (i, j).
        These neighboring cells will be checked for potential collisions.
        
        :param i: X index of the grid cell.
        :param j: Y index of the grid cell.
        :yield: A list of particles in each neighboring cell.
        """
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.grid_width and 0 <= nj < self.grid_height:
                    yield self.grid[ni][nj]

    def elastic_collision(self, p1: Particle, p2: Particle, dist: float, delta_pos: np.ndarray) -> None:
        """
        Resolve an elastic collision between two particles, updating their velocities and positions.
        
        :param p1: First particle.
        :param p2: Second particle.
        :param dist: Distance between the particles.
        :param delta_pos: Vector from p2 to p1.
        """
        delta_vel = p1.velocity - p2.velocity
        norm_delta_pos = delta_pos / dist
        velocity_along_axis = np.dot(delta_vel, norm_delta_pos)

        # If the particles are moving apart, skip the collision resolution
        if velocity_along_axis > 0:
            return

        # Compute impulse magnitude for elastic collision
        impulse_magnitude = (2 * velocity_along_axis) / (p1.mass + p2.mass)

        # Update velocities
        p1.velocity -= impulse_magnitude * p2.mass * norm_delta_pos
        p2.velocity += impulse_magnitude * p1.mass * norm_delta_pos

        # Correct particle positions to resolve any overlap
        overlap = p1.radius + p2.radius - dist + self.epsilon
        correction = norm_delta_pos * (overlap / 2)
        p1.position += correction
        p2.position -= correction

    def resolve_collisions(self) -> None:
        """
        Check and resolve collisions between particles using grid-based spatial optimization.
        """
        # Update the grid based on the new particle positions
        self.update_grid()

        # Iterate through the grid cells
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                cell_particles = self.grid[i][j]

                # Check for collisions within the same cell
                for k in range(len(cell_particles)):
                    for l in range(k + 1, len(cell_particles)):
                        p1, p2 = cell_particles[k], cell_particles[l]
                        self.check_and_resolve_collision(p1, p2)

                # Check for collisions with neighboring cells
                for neighbor_cell in self.get_neighboring_cells(i, j):
                    for p1 in cell_particles:
                        for p2 in neighbor_cell:
                            if p1 is not p2:
                                self.check_and_resolve_collision(p1, p2)

    def check_and_resolve_collision(self, p1: Particle, p2: Particle) -> None:
        """
        Check if two particles are colliding, and if so, resolve the collision.
        
        :param p1: First particle.
        :param p2: Second particle.
        """
        delta_pos = p1.position - p2.position
        dist = np.linalg.norm(delta_pos)

        # If particles overlap, resolve the collision
        if dist < (p1.radius + p2.radius - self.epsilon):
            self.elastic_collision(p1, p2, dist, delta_pos)

    def update_positions(self, dt: float) -> None:
        """
        Update the positions of all particles and check for wall collisions.
        
        :param dt: Time step for updating particle positions.
        """
        for particle in self.particles:
            # Check and resolve wall collisions before moving particles
            self.check_wall_collision(particle, dt)
            # Move the particles
            particle.move(dt)

    def simulate(self, n_steps: int, dt: float) -> None:
        """
        Simulate the particle system for a given number of steps with time step dt.
        
        :param n_steps: Number of simulation steps.
        :param dt: Time step for each simulation step.
        """
        for _ in range(n_steps):
            # Update particle positions
            self.update_positions(dt)
            # Resolve any collisions
            self.resolve_collisions()


class Rectangle(Scene):
    def __init__(self, width: float, height: float) -> None:
        """
        Initialize a rectangular scene with specified width and height.
        
        :param width: Width of the rectangle.
        :param height: Height of the rectangle.
        """
        super().__init__(width=width, height=height, cell_size=width / 20)
        self.width = width
        self.height = height

    def add_particle(self, particle: Particle) -> None:
        """
        Add a particle to the rectangle, ensuring it is within bounds.
        
        :param particle: The particle to be added.
        :raises ValueError: If the particle is out of bounds.
        """
        if (particle.radius <= particle.position[0] <= self.width - particle.radius and 
            particle.radius <= particle.position[1] <= self.height - particle.radius):
            self.particles.append(particle)
        else:
            raise ValueError("Particle out of bounds")
        
    def _is_valid_particle_(self, particle: Particle) -> bool:
        """
        Check if a particle can be placed in the rectangle without overlap or out of bounds.
        
        :param particle: The particle to check.
        :return: True if the particle is valid, False otherwise.
        """
        if not (particle.radius <= particle.position[0] <= self.width - particle.radius and
                particle.radius <= particle.position[1] <= self.height - particle.radius):
            return False
        for other_particle in self.particles:
            if self.particles_overlap(particle, other_particle):
                return False
        return True
    
    def generate_particles(self, num_particles: int, radius_range: (float, float), density_range: (float, float)) -> None:
        """
        Generate a given number of particles with random positions, velocities, radius, and density.
        
        :param num_particles: Number of particles to generate.
        :param radius_range: Tuple of (min_radius, max_radius) for particle radii.
        :param density_range: Tuple of (min_density, max_density) for particle densities.
        :raises RuntimeError: If unable to place all particles without overlap after many attempts.
        """
        max_attempts = 1000
        attempts = 0
        while len(self.particles) < num_particles and attempts < max_attempts:
            position = np.array([random.uniform(0, self.width), random.uniform(0, self.height)])
            velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)]) * 100
            radius = random.uniform(*radius_range)
            density = random.uniform(*density_range)
            new_particle = Particle(position, velocity, color=tuple(np.random.uniform(0, 1, 3)), radius=radius, density=density)
            if self._is_valid_particle_(new_particle):
                self.particles.append(new_particle)
            attempts += 1
        if attempts >= max_attempts:
            raise RuntimeError("Unable to place all particles without overlap after many attempts.")

    def check_wall_collision(self, particle: Particle, dt: float) -> None:
        """
        Check and resolve wall collisions for a particle in the rectangular scene.
        
        :param particle: The particle to check for wall collision.
        :param dt: Time step for continuous collision detection.
        """
        next_position = particle.position + particle.velocity * dt
        
        # Check for left or right wall collisions
        if next_position[0] - particle.radius < 0:
            # Reflect velocity for left wall
            particle.velocity[0] = -particle.velocity[0]
            particle.position[0] = 2 * particle.radius - particle.position[0]
        elif next_position[0] + particle.radius > self.width:
            # Reflect velocity for right wall
            particle.velocity[0] = -particle.velocity[0]
            particle.position[0] = 2 * (self.width - particle.radius) - particle.position[0]

        # Check for top or bottom wall collisions
        if next_position[1] - particle.radius < 0:
            # Reflect velocity for bottom wall
            particle.velocity[1] = -particle.velocity[1]
            particle.position[1] = 2 * particle.radius - particle.position[1]
        elif next_position[1] + particle.radius > self.height:
            # Reflect velocity for top wall
            particle.velocity[1] = -particle.velocity[1]
            particle.position[1] = 2 * (self.height - particle.radius) - particle.position[1]


class Circle(Scene):
    def __init__(self, radius: float) -> None:
        """
        Initialize a circular scene with a given radius.
        
        :param radius: Radius of the circular boundary.
        """
        super().__init__(width=radius * 2, height=radius * 2, cell_size=radius / 20)
        self.radius = radius

    def add_particle(self, particle: Particle) -> None:
        """
        Add a particle to the circular scene.
        
        :param particle: The particle to be added.
        """
        self.particles.append(particle)

    def check_wall_collision(self, particle: Particle, dt: float) -> None:
        """
        Check and resolve wall collisions with the circular boundary.
        
        :param particle: The particle to check for boundary collision.
        :param dt: Time step for continuous collision detection.
        """
        distance_to_center = np.linalg.norm(particle.position)

        # Check if the particle is colliding with the circular boundary
        if distance_to_center + particle.radius >= self.radius:
            # Calculate the normal direction from the center of the circle
            normal = particle.position / distance_to_center

            # Reflect the particle's velocity along the normal direction
            particle.velocity -= 2 * np.dot(particle.velocity, normal) * normal

            # Correct the particle's position to avoid penetrating the boundary
            penetration_depth = (distance_to_center + particle.radius) - self.radius
            particle.position -= normal * penetration_depth

    def is_valid_particle(self, particle: Particle) -> bool:
        """
        Check if a particle can be placed in the circle without overlapping or exceeding the boundary.
        
        :param particle: The particle to check.
        :return: True if the particle is valid, False otherwise.
        """
        if np.linalg.norm(particle.position) + particle.radius > self.radius:
            return False
        for other_particle in self.particles:
            if self.particles_overlap(particle, other_particle):
                return False
        return True

    def generate_particles(self, num_particles: int, radius_range: (float, float), 
                           density_range: (float, float)) -> None:
        """
        Generate a given number of particles in the circular boundary.
        
        :param num_particles: Number of particles to generate.
        :param radius_range: Tuple of (min_radius, max_radius) for particle radii.
        :param density_range: Tuple of (min_density, max_density) for particle densities.
        :raises RuntimeError: If unable to place all particles without overlap after many attempts.
        """
        max_attempts = 1000
        attempts = 0
        while len(self.particles) < num_particles and attempts < max_attempts:
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(0, self.radius)
            position = np.array([distance * np.cos(angle), distance * np.sin(angle)])
            velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)]) * 100
            radius = random.uniform(*radius_range)
            density = random.uniform(*density_range)
            new_particle = Particle(position, velocity, color=tuple(np.random.uniform(0, 1, 3)), radius=radius, density=density)
            if self.is_valid_particle(new_particle):
                self.particles.append(new_particle)
            attempts += 1
        if attempts >= max_attempts:
            raise RuntimeError("Unable to place all particles without overlap after many attempts.")
        


class DoubleRectangle(Scene):
    def __init__(self, left_dims: (float, float), right_dims: (float, float), pipe_dims: (float, float)):
        """
        Initialize a DoubleRectangle with two side-by-side rectangular containers connected by a pipe.
        
        :param left_dims: Tuple (width, height) for the left container.
        :param right_dims: Tuple (width, height) for the right container.
        :param pipe_dims: Tuple (width, height) for the pipe connecting the two containers.
        """
        left_width, left_height = left_dims
        right_width, right_height = right_dims
        pipe_width, pipe_height = pipe_dims

        total_width = left_width + right_width + pipe_width
        max_height = max(left_height, right_height)

        super().__init__(total_width, max_height, cell_size=min(left_width, right_width) / 20)

        self.left_width = left_width
        self.left_height = left_height
        self.right_width = right_width
        self.right_height = right_height
        self.pipe_width = pipe_width
        self.pipe_height = pipe_height
        self.epsilon = 1e-9  # Floating-point tolerance

    def is_valid_particle(self, particle: Particle) -> bool:
        """
        Check if a particle can be placed in the rectangle without overlapping other particles.
        
        :param particle: The particle to check.
        :return: True if the particle is valid, False otherwise.
        """
       
        for other_particle in self.particles:
            if self.particles_overlap(particle, other_particle):
                return False
        return True
    
    def add_particle(self, particle: Particle, location: str):
        """
        Add a particle to either the left or right container, or allow it to pass through the pipe region.
        
        :param particle: The particle to add.
        :param location: 'left', 'right', or 'split' to indicate where the particles are generated.
        """
        if location == 'left':
            if self._is_in_left_container_(particle):
                self.particles.append(particle)
            else:
                raise ValueError("Particle out of bounds in left container")

        elif location == 'right':
            if self._is_in_right_container_(particle):
                self.particles.append(particle)
            else:
                raise ValueError("Particle out of bounds in right container")

        elif location == 'split':
            if random.random() < 0.5:
                if self._is_in_left_container_(particle):
                    self.particles.append(particle)
                else:
                    if self._is_in_right_container_(particle):
                        self.particles.append(particle)
        else:
            raise ValueError("Invalid location. Choose 'left', 'right', or 'split'.")

    def _is_in_left_container_(self, particle: Particle) -> bool:
        """
        Check if the particle is within the bounds of the left container.
        """
        x, y = particle.position
        return (particle.radius <= x <= self.left_width - particle.radius and 
                particle.radius <= y <= self.left_height - particle.radius)

    def _is_in_right_container_(self, particle: Particle) -> bool:
        """
        Check if the particle is within the bounds of the right container.
        """
        x, y = particle.position
        return (self.left_width + self.pipe_width + particle.radius <= x <= self.left_width + self.pipe_width + self.right_width - particle.radius and 
                particle.radius <= y <= self.right_height - particle.radius)

    def _is_in_pipe_(self, particle: Particle) -> bool:
        """
        Check if the particle is within the bounds of the pipe.
        """
        x, y = particle.position
        return (self.left_width <= x <= self.left_width + self.pipe_width and 
                self.left_height / 2 - self.pipe_height / 2 <= y <= self.left_height / 2 + self.pipe_height / 2)

    def check_wall_collision(self, particle: Particle, dt: float):
        """
        Check and handle wall collisions in either the left or right container, or the pipe region.
        """
        next_position = particle.position + particle.velocity * dt
        x, y = next_position

        # Left container wall collisions
        if x - particle.radius < 0:  # Left wall
            particle.velocity[0] = -particle.velocity[0]
            next_position[0] = particle.radius
        elif self.left_width - particle.radius < x < self.left_width + particle.radius:  # Near right wall of left container
            if y < self.left_height / 2 - self.pipe_height / 2 or y > self.left_height / 2 + self.pipe_height / 2:
                particle.velocity[0] = -particle.velocity[0]
                next_position[0] = self.left_width - particle.radius

        # Right container wall collisions
        elif self.left_width + self.pipe_width - particle.radius < x < self.left_width + self.pipe_width + particle.radius:  # Near left wall of right container
            if y < self.right_height / 2 - self.pipe_height / 2 or y > self.right_height / 2 + self.pipe_height / 2:
                particle.velocity[0] = -particle.velocity[0]
                next_position[0] = self.left_width + self.pipe_width + particle.radius
        elif x + particle.radius > self.left_width + self.pipe_width + self.right_width:  # Right wall
            particle.velocity[0] = -particle.velocity[0]
            next_position[0] = self.left_width + self.pipe_width + self.right_width - particle.radius

        # Vertical collisions
        if x <= self.left_width:  # Left container
            if y - particle.radius < 0 or y + particle.radius > self.left_height:
                particle.velocity[1] = -particle.velocity[1]
                next_position[1] = particle.radius if y < self.left_height / 2 else self.left_height - particle.radius
        elif x >= self.left_width + self.pipe_width:  # Right container
            if y - particle.radius < 0 or y + particle.radius > self.right_height:
                particle.velocity[1] = -particle.velocity[1]
                next_position[1] = particle.radius if y < self.right_height / 2 else self.right_height - particle.radius
        else:  # Pipe region
            if y - particle.radius < self.left_height / 2 - self.pipe_height / 2 or y + particle.radius > self.left_height / 2 + self.pipe_height / 2:
                particle.velocity[1] = -particle.velocity[1]
                next_position[1] = (self.left_height / 2 - self.pipe_height / 2 + particle.radius 
                                    if y < self.left_height / 2 
                                    else self.left_height / 2 + self.pipe_height / 2 - particle.radius)

        # Update particle position after handling collisions
        particle.position = next_position


    def generate_particles(self, num_particles: int, radius_range: (float, float), density_range: (float, float), location: str):
        """
        Generate particles in either the left, right container, or split between them.
        :param num_particles: Number of particles to generate.
        :param radius_range: Range of particle radii.
        :param density_range: Range of particle densities.
        :param location: 'left', 'right', or 'split' to indicate where the particles are generated.
        """
        max_attempts = 1000
        attempts = 0
        while len(self.particles) < num_particles and attempts < max_attempts:
            if location == 'split':
                location_choice = 'left' if random.random() < 0.5 else 'right'
            else:
                location_choice = location

            position = np.array([
                random.uniform(0, self.left_width) if location_choice == 'left'
                else random.uniform(self.left_width + self.pipe_width, self.left_width + self.pipe_width + self.right_width),
                random.uniform(0, self.left_height if location_choice == 'left' else self.right_height)
            ])
            velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)]) * 100
            radius = random.uniform(*radius_range)
            density = random.uniform(*density_range)

            # Assign color based on location and 'split' option
            if location == 'split':
                color = (0, 0, 1) if location_choice == 'left' else (0, 1, 0)  # Blue for left, Green for right
            else:
                color = tuple(np.random.uniform(0, 1, 3))  # Random color for non-split options

            new_particle = Particle(position, velocity, color=color, radius=radius, density=density)

            if location_choice == 'left' and self._is_in_left_container_(new_particle):
                if self.is_valid_particle(new_particle):
                    self.particles.append(new_particle)
            elif location_choice == 'right' and self._is_in_right_container_(new_particle):
                if self.is_valid_particle(new_particle):
                    self.particles.append(new_particle)

            attempts += 1

        if attempts >= max_attempts:
            raise RuntimeError("Unable to place all particles without overlap after many attempts.")

