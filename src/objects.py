from abc import ABC, abstractmethod
import random
import numpy as np

#TODO: Update the grid structure for the scene so that i it efficiently covers differnt shapes (and not just rectangles)
#TODO: Fix DoubleRectangle class - when the two boxes are different sizes, the transition between the pipe behaves weird
#TODO: Optimize edge cell marking process
#TODO: Fix optimized edge collision checking for cicle class (with no gravity)
#TODO: Add generate_grid_cells() and mark_edge_cells() to DoubleRectangle and Triangle


class Particle:
    def __init__(self, position: np.ndarray, velocity: np.ndarray, color: str = 'black', 
                 radius: float = 1.0, density: float = 1.0, restitution: float = 1.0) -> None:
        """
        Initialize a particle with position, velocity, color, radius, and density.
        
        :param position: A numpy array of shape (2,) representing (x, y) position.
        :param velocity: A numpy array of shape (2,) representing (vx, vy) velocity.
        :param color: String representing the color of the particle.
        :param radius: Float representing the radius of the particle.
        :param density: Float representing the density of the particle.
        """
        self.position = np.array(position)    # (x, y)
        self.velocity = np.array(velocity)    # (vx, vy)
        self.color = color
        self.radius = radius
        self.density = density
        self.restitution = restitution
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
    def __init__(self, width: float, height: float, vertical_gravity: float) -> None:
        """
        Initialize the abstract scene, containing particles and a spatial grid for efficient collision detection.
        
        :param width: Width of the scene.
        :param height: Height of the scene.
        :param cell_size: Size of each cell in the spatial grid for particle positioning.
        """
        self.particles = []  # List to hold particles in the scene
        self.width = width
        self.height = height
        self.vertical_gravity = vertical_gravity
        
        # All of these is set when generate_particles(...) is called 
        self.cell_size = None
        self.grid_width = None 
        self.grid_height = None 
        self.grid = None
        

    @abstractmethod
    def mark_edge_cells(self) -> None:
            """
            Abstract method to mark the edge cells in the grid based on the scene geometry.
            Implemented in derived classes like Rectangle, Circle, or Triangle.
            """
            pass
    
    @abstractmethod
    def generate_grid_cells(self) -> None:
            """
            Abstract method to generate the cells in the grid based on the scene geometry.
            Implemented in derived classes like Rectangle, Circle, or Triangle.
            """

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

    def get_cell_indices(self, positions: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Get the grid indices for an array of particle positions.
        
        :param positions: An array of particle positions of shape (N, 2).
        :return: A tuple of arrays (i_indices, j_indices) representing the grid cells.
        """
        # Compute the indices for each particle
        i_indices = np.clip((positions[:, 0] // self.cell_size).astype(int), 0, self.grid_width - 1)
        j_indices = np.clip((positions[:, 1] // self.cell_size).astype(int), 0, self.grid_height - 1)
        return i_indices, j_indices

    def update_grid(self) -> None:
        """
        Clear and update the grid with the current positions of all particles.
        """
        # Clear the grid
        self.grid = [[[[],False, False] for _ in range(self.grid_height)] for _ in range(self.grid_width)]

        # Mark the edges
        self.mark_edge_cells()

        # Get all particle positions as a numpy array
        positions = np.array([particle.position for particle in self.particles])

        # Compute the grid indices for all particles
        i_indices, j_indices = self.get_cell_indices(positions)

        # Efficiently add particles to the grid
        for particle, i, j in zip(self.particles, i_indices, j_indices):
            self.grid[i][j][0].append(particle)


    def collision(self, p1: Particle, p2: Particle, dist: float, delta_pos: np.ndarray, restitution: float) -> None:
        """
        Resolve a collision between two particles, updating their velocities and positions
        according to elastic collision principles.
        
        :param p1: First particle.
        :param p2: Second particle.
        :param dist: Distance between the particles.
        :param delta_pos: Vector from p2 to p1.
        :param restitution: Coefficient of restitution, determining elasticity.
        """
        # Normalize the delta position vector to get the collision normal
        norm_delta_pos = delta_pos / dist

        # Compute the relative velocity between the particles
        delta_vel = p1.velocity - p2.velocity

        # Compute the velocity component along the collision normal
        velocity_along_normal = np.dot(delta_vel, norm_delta_pos)

        # If the particles are moving apart (velocity along normal is positive), skip collision resolution
        if velocity_along_normal > 0:
            return

        # Calculate the impulse magnitude for a perfectly elastic collision
        # (taking into account the coefficient of restitution)
        impulse_magnitude = (-(1 + restitution) * velocity_along_normal) / (1 / p1.mass + 1 / p2.mass)

        # Calculate the impulse vector
        impulse = impulse_magnitude * norm_delta_pos

        # Update velocities based on the impulse
        p1.velocity += impulse / p1.mass
        p2.velocity -= impulse / p2.mass

        # Position correction to prevent overlap 
        overlap = (p1.radius + p2.radius) - dist 
        correction = norm_delta_pos * (overlap / 2)
        p1.position += correction
        p2.position -= correction



    def resolve_collisions(self) -> None:
        """
        Check and resolve collisions between particles using grid-based spatial optimization,
        with additional sorting along an axis for further optimization.
        """

        # Update the grid based on the new particle positions
        self.update_grid()

        # Iterate through the grid cells
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                
                # Only check if cell has particles
                if len(self.grid[i][j][0]) > 0:

                    # If not sorted
                    if not self.grid[i][j][1]: 
                        # Sort particles in the current cell by x-axis
                        self.grid[i][j][0].sort(key=lambda p: p.position[0])
                        # Set sorted indicator
                        self.grid[i][j][1] = True

                    # Check for collisions within the same cell
                    for k in range(len(self.grid[i][j][0])):
                        for l in range(k + 1, len(self.grid[i][j][0])):
                            # Early exit if the distance along the x-axis exceeds the sum of radii
                            if self.grid[i][j][0][l].position[0] - self.grid[i][j][0][k].position[0] > (self.grid[i][j][0][k].radius + self.grid[i][j][0][l].radius):
                                break

                            self.check_and_resolve_collision(self.grid[i][j][0][l], self.grid[i][j][0][k])

                    # Check for collisions with neighboring cells
                    neighbours = [(-1, -1), (-1, 0), (-1, 1),   # Column behind
                                   (0, -1),  (0, 1),            # Left and Right
                                   (1, -1),  (1, 0),  (1, 1)]   # Column in front

                    for (di, dj) in neighbours:
                        ni, nj = i + di, j + dj
                        # Check if the neighboring cell is within bounds
                        if 0 <= ni < self.grid_width and 0 <= nj < self.grid_height:
                            # Sort the neighboring cell if not already sorted
                            if not self.grid[ni][nj][1]:
                                self.grid[ni][nj][0].sort(key=lambda p: p.position[0])
                                self.grid[ni][nj][1] = True

                            # If neighbour is behind (go large -> small in neighbour cell)
                            if di < 0 or dj < 0:
                                for k in range(len(self.grid[i][j][0])):
                                    for l in range(len(self.grid[ni][nj][0])-1,-1,-1):
                                        # Compute sum of radia once 
                                        R = self.grid[i][j][0][k].radius + self.grid[ni][nj][0][l].radius

                                        # Early exit if the x-axis distance exceeds the sum of radii
                                        if self.grid[ni][nj][0][l].position[0] - self.grid[i][j][0][k].position[0] > (R):
                                            break

                                        # Skip comparison if it's the same particle
                                        if np.all(self.grid[i][j][0][k].position == self.grid[ni][nj][0][l].position):
                                            continue

                                        # Use a more precise distance check
                                        if np.linalg.norm(self.grid[i][j][0][k].position - self.grid[ni][nj][0][l].position) <= (R):
                                            self.check_and_resolve_collision(self.grid[i][j][0][k], self.grid[ni][nj][0][l])


                            # If neighbour is infront (go large -> small in current cell)
                            elif di > 0 or dj > 0:
                                for k in range(len(self.grid[i][j][0])-1,-1,-1):
                                    for l in range(len(self.grid[ni][nj][0])):
                                        # Compute sum of radia once 
                                        R = self.grid[i][j][0][k].radius + self.grid[ni][nj][0][l].radius

                                        # Early exit if the x-axis distance exceeds the sum of radii
                                        if self.grid[ni][nj][0][l].position[0] - self.grid[i][j][0][k].position[0] > (R):
                                            break

                                        # Skip comparison if it's the same particle
                                        if np.all(self.grid[i][j][0][k].position == self.grid[ni][nj][0][l].position):
                                            continue

                                        # Use a more precise distance check
                                        if np.linalg.norm(self.grid[i][j][0][k].position - self.grid[ni][nj][0][l].position) <= (R):
                                            self.check_and_resolve_collision(self.grid[i][j][0][k], self.grid[ni][nj][0][l])

                            # If neighbour above or below infront
                            else:
                                for k in range(len(self.grid[i][j][0])):
                                    for l in range(len(self.grid[ni][nj][0])):
                                        # Compute sum of radia once 
                                        R = self.grid[i][j][0][k].radius + self.grid[ni][nj][0][l].radius

                                        # Early exit if the x-axis distance exceeds the sum of radii
                                        if self.grid[ni][nj][0][l].position[0] - self.grid[i][j][0][k].position[0] > (R):
                                            break

                                        # Skip comparison if it's the same particle
                                        if np.all(self.grid[i][j][0][k].position == self.grid[ni][nj][0][l].position):
                                            continue

                                        # Use a more precise distance check
                                        if np.linalg.norm(self.grid[i][j][0][k].position - self.grid[ni][nj][0][l].position) <= (R):
                                            self.check_and_resolve_collision(self.grid[i][j][0][k], self.grid[ni][nj][0][l])


        ##### Slow but robust native working method - default to this for breaking changes #####
        """for i in range(len(self.particles)):
            for j in range(i+1,len(self.particles)):
                self.check_and_resolve_collision(self.particles[i], self.particles[j])"""



    def check_and_resolve_collision(self, p1: Particle, p2: Particle) -> None:
        """
        Check if two particles are colliding, and if so, resolve the collision.
        
        :param p1: First particle.
        :param p2: Second particle.
        """
        delta_pos = p1.position - p2.position
        dist = np.linalg.norm(delta_pos)

        # If particles overlap, resolve the collision
        if dist < (p1.radius + p2.radius):
            restitution = (p1.restitution + p2.restitution) / 2
            self.collision(p1, p2, dist, delta_pos, restitution=restitution)


    def update_positions(self, dt: float) -> None:
        """
        Update the positions of all particles and check for wall collisions.
        Apply gravity to each particle.
        """

        # Iterate through the grid cells
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                # If is an edge cell -> also check for wall collision
                if self.grid[i][j][2]:
                    for particle in self.grid[i][j][0]:
                        # Apply gravity if it's set
                        if self.vertical_gravity != 0.0:
                            particle.velocity[1] -= self.vertical_gravity * dt

                        # Check and resolve wall collisions before moving particles
                        self.check_wall_collision(particle, dt)

                        # Move the particle
                        particle.move(dt)
                else:
                    for particle in self.grid[i][j][0]:
                        # Apply gravity if it's set
                        if self.vertical_gravity != 0.0:
                            particle.velocity[1] -= self.vertical_gravity * dt

                        # Move the particle
                        particle.move(dt)

        """##### Slow but robust native working method - default to this for breaking changes #####
        for particle in self.particles:
            # Apply gravity if it's set
            if self.vertical_gravity != 0.0:
                particle.velocity[1] -= self.vertical_gravity * dt

            # Check and resolve wall collisions before moving particles
            self.check_wall_collision(particle, dt)

            # Move the particle
            particle.move(dt)"""


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
    def __init__(self, width: float, height: float, vertical_gravity: float = 0.0, restitution: float = 1.0) -> None:
        """
        Initialize a rectangular scene with specified width and height.
        
        :param width: Width of the rectangle.
        :param height: Height of the rectangle.
        """
        super().__init__(width=width, height=height, vertical_gravity=vertical_gravity)
        self.width = width
        self.height = height
        self.restitution = restitution


    def generate_grid_cells(self):
        return [[[[],False, False] for _ in range(self.grid_height)] for _ in range(self.grid_width)]

    def mark_edge_cells(self) -> None:
        """
        Mark the edge cells in the grid for a rectangular scene.
        """
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                # Cells along the rectangle's perimeter are edge cells
                if i == 0 or i == self.grid_width - 1 or j == 0 or j == self.grid_height - 1:
                    self.grid[i][j][2] = True  # Mark this cell as an edge cell

    def check_wall_collision(self, particle: Particle, dt: float) -> None:
        """
        Check and resolve wall collisions for a particle in the rectangular scene.
        :param particle: The particle to check for wall collision.
        :param dt: Time step for continuous collision detection.
        """
        next_position = particle.position + particle.velocity * dt

        # Check for left wall collision
        if next_position[0] - particle.radius < 0:
            penetration = particle.radius - next_position[0]
            particle.position[0] = particle.radius + penetration * particle.restitution
            particle.velocity[0] = -particle.velocity[0] * particle.restitution

        # Check for right wall collision
        elif next_position[0] + particle.radius > self.width:
            penetration = next_position[0] + particle.radius - self.width
            particle.position[0] = self.width - particle.radius - penetration * particle.restitution
            particle.velocity[0] = -particle.velocity[0] * particle.restitution

        # Check for bottom wall collision
        if next_position[1] - particle.radius < 0:
            penetration = particle.radius - next_position[1]
            particle.position[1] = particle.radius + penetration * particle.restitution
            particle.velocity[1] = -particle.velocity[1] * particle.restitution

        # Check for top wall collision
        elif next_position[1] + particle.radius > self.height:
            penetration = next_position[1] + particle.radius - self.height
            particle.position[1] = self.height - particle.radius - penetration * particle.restitution
            particle.velocity[1] = -particle.velocity[1] * particle.restitution

        # Update position based on remaining time step
        remaining_dt = dt * (1 - particle.restitution)
        particle.position += particle.velocity * remaining_dt


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
    
    def generate_particles(self, num_particles: int, radius_range: (float, float), density_range: (float, float), speed_scaling: float = 1, max_attempts: int = 1000) -> None:
        """
        Generate a given number of particles with random positions, velocities, radius, and density.
        
        :param num_particles: Number of particles to generate.
        :param radius_range: Tuple of (min_radius, max_radius) for particle radii.
        :param density_range: Tuple of (min_density, max_density) for particle densities.
        :raises RuntimeError: If unable to place all particles without overlap after many attempts.
        """
        
        attempts = 0
        while len(self.particles) < num_particles and attempts < max_attempts:
            position = np.array([random.uniform(0, self.width), random.uniform(0, self.height)])
            velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])*speed_scaling
            radius = random.uniform(*radius_range)
            density = random.uniform(*density_range)
            new_particle = Particle(position, velocity, 
                                    color=tuple(np.random.uniform(0, 1, 3)), 
                                    radius=radius, 
                                    density=density,
                                    restitution=self.restitution)
            if self._is_valid_particle_(new_particle):
                self.particles.append(new_particle)
            attempts += 1
        if attempts >= max_attempts:
            raise RuntimeError("Unable to place all particles without overlap after many attempts.")

        #### Has to fit at least approx 4 of the largest particles in single cell ####
        max_particle_diameter = max([2*p.radius for p in self.particles])
        self.cell_size = 2.5 * max_particle_diameter
        self.grid_width = int(np.ceil(self.width / self.cell_size))
        self.grid_height = int(np.ceil(self.height / self.cell_size))
        # Initialize the grid
        self.grid = self.generate_grid_cells()
        # Mark edge cells after initializing the grid
        self.mark_edge_cells()

    


class Circle(Scene):
    def __init__(self, radius: float, vertical_gravity: float = 0.0, restitution: float = 1.0) -> None:
        """
        Initialize a circular scene with a given radius.
        
        :param radius: Radius of the circular boundary.
        """
        super().__init__(width=radius * 2, height=radius * 2, vertical_gravity=vertical_gravity)
        self.radius = radius
        self.restitution = restitution


    def generate_grid_cells(self):
        """
        Generate a 2D grid of cells for a circular scene. 
        Each cell is initialized with an empty particle list, a sorted flag, and an edge flag.
        
        :return: 2D list of grid cells with format [[[particles], is_sorted, is_edge], ...]
        """
        return [[[[], False, False] for _ in range(self.grid_height)] for _ in range(self.grid_width)]

    def mark_edge_cells(self) -> None:
        """
        Mark the edge cells in the grid for a circular scene.
        The edge cells are those whose center lies approximately at the circle's boundary,
        extending outwards by a thickness of 2 cells.
        """
        # Get the center of the grid (in terms of grid indices)
        center_x = self.grid_width / 2
        center_y = self.grid_height / 2

        # Define thickness in terms of cell count
        edge_thickness_cells = 2

        # Iterate through all the grid cells
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                # Calculate the center of the current grid cell (i, j) in grid coordinates
                cell_center_x = (i + 0.5) - center_x  # Center relative to the circle's center
                cell_center_y = (j + 0.5) - center_y  # Center relative to the circle's center

                # Calculate the distance from the center of the circle
                distance_from_center = np.sqrt(cell_center_x**2 + cell_center_y**2)

                # Scale the distance based on the radius of the circle
                scaled_distance = distance_from_center * self.cell_size

                # Mark as an edge cell if the distance is within the range of the circle's radius +/- (edge_thickness_cells * cell_size)
                if self.radius - (edge_thickness_cells * self.cell_size) <= scaled_distance <= self.radius + (edge_thickness_cells * self.cell_size):
                    self.grid[i][j][2] = True  # Mark this cell as an edge cell


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

            # Reflect the particle's velocity along the normal direction and apply restitution
            particle.velocity -= 2 * particle.restitution * np.dot(particle.velocity, normal) * normal

            # Correct the particle's position to avoid penetrating the boundary
            penetration_depth = (distance_to_center + particle.radius) - self.radius
            particle.position -= normal * penetration_depth

    
    def add_particle(self, particle: Particle) -> None:
        """
        Add a particle to the circular scene.
        
        :param particle: The particle to be added.
        """
        self.particles.append(particle)

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
                           density_range: (float, float), speed_scaling: float = 1, max_attempts: int = 1000) -> None:
        """
        Generate a given number of particles in the circular boundary.
        
        :param num_particles: Number of particles to generate.
        :param radius_range: Tuple of (min_radius, max_radius) for particle radii.
        :param density_range: Tuple of (min_density, max_density) for particle densities.
        :raises RuntimeError: If unable to place all particles without overlap after many attempts.
        """
        attempts = 0
        while len(self.particles) < num_particles and attempts < max_attempts:
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(0, self.radius)
            position = np.array([distance * np.cos(angle), distance * np.sin(angle)])
            velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)]) * speed_scaling
            radius = random.uniform(*radius_range)
            density = random.uniform(*density_range)
            new_particle = Particle(position, velocity, color=tuple(np.random.uniform(0, 1, 3)), radius=radius, density=density,
                                    restitution=self.restitution)
            if self.is_valid_particle(new_particle):
                self.particles.append(new_particle)
            attempts += 1
        if attempts >= max_attempts:
            raise RuntimeError("Unable to place all particles without overlap after many attempts.")
        
        #### Has to fit at least approx 4 of the largest particles in single cell ####
        max_particle_diameter = max([2*p.radius for p in self.particles])
        self.cell_size = 2 * max_particle_diameter
        self.grid_width = int(np.ceil(self.width / self.cell_size))
        self.grid_height = int(np.ceil(self.height / self.cell_size))
        # Initialize the grid
        self.grid = self.generate_grid_cells()
        # Mark edge cells after initializing the grid
        self.mark_edge_cells()
        


class DoubleRectangle(Scene):
    def __init__(self, left_dims: (float, float), right_dims: (float, float), pipe_dims: (float, float), vertical_gravity: float = 0.0, restitution: float = 1.0):
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

        super().__init__(total_width, max_height, vertical_gravity=vertical_gravity)

        self.left_width = left_width
        self.left_height = left_height
        self.right_width = right_width
        self.right_height = right_height
        self.pipe_width = pipe_width
        self.pipe_height = pipe_height
        self.restitution = restitution

    def check_wall_collision(self, particle: Particle, dt: float):
        """
        Check and handle wall collisions in either the left or right container, or the pipe region.
        """
        next_position = particle.position + particle.velocity * dt
        x, y = next_position

        # Left container wall collisions
        if x - particle.radius < 0:  # Left wall
            particle.velocity[0] = -particle.restitution * particle.velocity[0]
            next_position[0] = particle.radius
        elif self.left_width - particle.radius < x < self.left_width + particle.radius:  # Near right wall of left container
            if y < self.left_height / 2 - self.pipe_height / 2 or y > self.left_height / 2 + self.pipe_height / 2:
                particle.velocity[0] = -particle.restitution * particle.velocity[0]
                next_position[0] = self.left_width - particle.radius

        # Right container wall collisions
        elif self.left_width + self.pipe_width - particle.radius < x < self.left_width + self.pipe_width + particle.radius:  # Near left wall of right container
            if y < self.right_height / 2 - self.pipe_height / 2 or y > self.right_height / 2 + self.pipe_height / 2:
                particle.velocity[0] = -particle.restitution * particle.velocity[0]
                next_position[0] = self.left_width + self.pipe_width + particle.radius
        elif x + particle.radius > self.left_width + self.pipe_width + self.right_width:  # Right wall
            particle.velocity[0] = -particle.restitution * particle.velocity[0]
            next_position[0] = self.left_width + self.pipe_width + self.right_width - particle.radius

        # Vertical collisions
        if x <= self.left_width:  # Left container
            if y - particle.radius < 0 or y + particle.radius > self.left_height:
                particle.velocity[1] = -particle.restitution * particle.velocity[1]
                next_position[1] = particle.radius if y < self.left_height / 2 else self.left_height - particle.radius
        elif x >= self.left_width + self.pipe_width:  # Right container
            if y - particle.radius < 0 or y + particle.radius > self.right_height:
                particle.velocity[1] = -particle.restitution * particle.velocity[1]
                next_position[1] = particle.radius if y < self.right_height / 2 else self.right_height - particle.radius
        else:  # Pipe region
            if y - particle.radius < self.left_height / 2 - self.pipe_height / 2 or y + particle.radius > self.left_height / 2 + self.pipe_height / 2:
                particle.velocity[1] = -particle.restitution * particle.velocity[1]
                next_position[1] = (self.left_height / 2 - self.pipe_height / 2 + particle.radius 
                                    if y < self.left_height / 2 
                                    else self.left_height / 2 + self.pipe_height / 2 - particle.radius)

        # Update particle position after handling collisions
        particle.position = next_position


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



    def generate_particles(self, num_particles: int, radius_range: (float, float), density_range: (float, float), location: str, speed_scaling: float = 1, max_attempts: int = 1000):
        """
        Generate particles in either the left, right container, or split between them.
        :param num_particles: Number of particles to generate.
        :param radius_range: Range of particle radii.
        :param density_range: Range of particle densities.
        :param location: 'left', 'right', or 'split' to indicate where the particles are generated.
        """
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
            velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)]) * speed_scaling
            radius = random.uniform(*radius_range)
            density = random.uniform(*density_range)

            # Assign color based on location and 'split' option
            if location == 'split':
                color = (0, 0, 1) if location_choice == 'left' else (0, 1, 0)  # Blue for left, Green for right
            else:
                color = tuple(np.random.uniform(0, 1, 3))  # Random color for non-split options

            new_particle = Particle(position, velocity, color=color, radius=radius, density=density,
                                    restitution=self.restitution)

            if location_choice == 'left' and self._is_in_left_container_(new_particle):
                if self.is_valid_particle(new_particle):
                    self.particles.append(new_particle)
            elif location_choice == 'right' and self._is_in_right_container_(new_particle):
                if self.is_valid_particle(new_particle):
                    self.particles.append(new_particle)

            attempts += 1

        if attempts >= max_attempts:
            raise RuntimeError("Unable to place all particles without overlap after many attempts.")

        #### Has to fit at least approx 4 of the largest particles in single cell ####
        max_particle_diameter = max([2*p.radius for p in self.particles])
        self.cell_size = 2.5 * max_particle_diameter
        self.grid_width = int(np.ceil(self.width / self.cell_size))
        self.grid_height = int(np.ceil(self.height / self.cell_size))
        # Initialize the grid
        self.grid = self.generate_grid_cells()
        # Mark edge cells after initializing the grid
        self.mark_edge_cells()


class Triangle(Scene):
    def __init__(self, base: float, height: float, angle: float = 0, vertical_gravity: float = 0.0, restitution: float = 1.0) -> None:
        """
        Initialize a triangular scene with specified base, height, and angle.
        
        :param base: Base length of the triangle.
        :param height: Height of the triangle.
        :param angle: Angle between the base and the line to the top vertex (in radians), default is 0.
        """
        super().__init__(width=base, height=height, vertical_gravity=vertical_gravity)
        self.base = base
        self.height = height
        self.angle = angle
        self.vertices = self._calculate_vertices()
        self.restitution = restitution

    def check_wall_collision(self, particle: Particle, dt: float) -> None:
        """
        Check and resolve wall collisions for a particle in the triangular scene.
        
        :param particle: The particle to check for wall collision.
        :param dt: Time step for continuous collision detection.
        """
        next_position = particle.position + particle.velocity * dt

        v1, v2, v3 = self.vertices
        edges = [(v1, v2), (v2, v3), (v3, v1)]
        
        for edge in edges:
            edge_vector = edge[1] - edge[0]
            normal = np.array([-edge_vector[1], edge_vector[0]])
            normal /= np.linalg.norm(normal)
            
            # Check if the particle is colliding with this edge
            distance = self._point_to_line_distance(next_position, edge[0], edge[1])
            if distance < particle.radius:
                # Reflect the velocity
                particle.velocity -= 2 * particle.restitution * np.dot(particle.velocity, normal) * normal
                
                # Move the particle away from the edge
                overlap = particle.radius - distance
                particle.position += overlap * normal

    def _calculate_vertices(self) -> np.ndarray:
        """
        Calculate the vertices of the triangle based on base, height, and angle.
        
        :return: numpy array of shape (3, 2) containing the x, y coordinates of the vertices.
        """
        half_base = self.base / 2
        top_x = half_base - self.height * np.tan(self.angle)
        vertices = np.array([
            [0, 0],
            [self.base, 0],
            [top_x, self.height]
        ])
        
        return vertices

    def _is_point_in_triangle(self, point: np.ndarray, radius: float = 0) -> bool:
        """
        Check if a point (with optional radius) is inside the triangle.
        
        :param point: The point to check.
        :param radius: The radius of the particle (for boundary checking).
        :return: True if the point is inside the triangle, False otherwise.
        """
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        v1, v2, v3 = self.vertices
        d1 = sign(point, v1, v2)
        d2 = sign(point, v2, v3)
        d3 = sign(point, v3, v1)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        # Check if the point is inside the triangle
        is_inside = not (has_neg and has_pos)
        
        # If radius is provided, check if the circle is completely inside the triangle
        if radius > 0 and is_inside:
            for edge in [(v1, v2), (v2, v3), (v3, v1)]:
                if self._point_to_line_distance(point, edge[0], edge[1]) < radius:
                    return False
        
        return is_inside


    def _point_to_line_distance(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """
        Calculate the distance from a point to a line segment.
        
        :param point: The point to calculate the distance from.
        :param line_start: The start point of the line segment.
        :param line_end: The end point of the line segment.
        :return: The distance from the point to the line segment.
        """
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        point_vec_scaled = point_vec / line_len
        t = np.dot(line_unitvec, point_vec_scaled)
        t = max(0.0, min(1.0, t))
        nearest = line_start + line_unitvec * t * line_len
        dist = np.linalg.norm(point - nearest)
        return dist
    
    def _is_valid_particle_(self, particle: Particle) -> bool:
        """
        Check if a particle can be placed in the triangle without overlap or out of bounds.
        
        :param particle: The particle to check.
        :return: True if the particle is valid, False otherwise.
        """
        if not self._is_point_in_triangle(particle.position, particle.radius):
            return False
        for other_particle in self.particles:
            if self.particles_overlap(particle, other_particle):
                return False
        return True
    

    def _generate_random_position(self) -> np.ndarray:
        """
        Generate a random position within the triangle.
        
        :return: numpy array of shape (2,) containing the x, y coordinates of the random position.
        """
        r1 = np.random.random()
        r2 = np.random.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        v1, v2, v3 = self.vertices
        return (1 - r1 - r2) * v1 + r1 * v2 + r2 * v3
    
    def add_particle(self, particle: Particle) -> None:
        """
        Add a particle to the triangle, ensuring it is within bounds.
        
        :param particle: The particle to be added.
        :raises ValueError: If the particle is out of bounds.
        """
        if self._is_valid_particle_(particle):
            self.particles.append(particle)
        else:
            raise ValueError("Particle out of bounds")

    def generate_particles(self, num_particles: int, radius_range: (float, float), density_range: (float, float), speed_scaling: float = 1, max_attempts: int = 1000) -> None:
        """
        Generate a given number of particles with random positions, velocities, radius, and density within the triangle.
        
        :param num_particles: Number of particles to generate.
        :param radius_range: Tuple of (min_radius, max_radius) for particle radii.
        :param density_range: Tuple of (min_density, max_density) for particle densities.
        :raises RuntimeError: If unable to place all particles without overlap after many attempts.
        """
        attempts = 0
        while len(self.particles) < num_particles and attempts < max_attempts:
            position = self._generate_random_position()
            velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)]) * speed_scaling
            radius = random.uniform(*radius_range)
            density = random.uniform(*density_range)
            new_particle = Particle(position, velocity, color=tuple(np.random.uniform(0, 1, 3)), radius=radius, density=density,
                                    restitution=self.restitution)
            if self._is_valid_particle_(new_particle):
                self.particles.append(new_particle)
            attempts += 1
        if attempts >= max_attempts:
            raise RuntimeError("Unable to place all particles without overlap after many attempts.")

        #### Has to fit at least approx 4 of the largest particles in single cell ####
        max_particle_diameter = max([2*p.radius for p in self.particles])
        self.cell_size = 2.5 * max_particle_diameter
        self.grid_width = int(np.ceil(self.width / self.cell_size))
        self.grid_height = int(np.ceil(self.height / self.cell_size))
        # Initialize the grid
        self.grid = self.generate_grid_cells()
        # Mark edge cells after initializing the grid
        self.mark_edge_cells()
