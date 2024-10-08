from abc import ABC, abstractmethod
import random
import numpy as np

#TODO: Update the grid structure for the scene so that i it efficiently covers differnt shapes (and not just rectangles)

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
    def __init__(self, width: float, height: float, cell_size: float, vertical_gravity: float) -> None:
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

        self.vertical_gravity = vertical_gravity



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
        self.grid = [[[] for _ in range(self.grid_height)] for _ in range(self.grid_width)]

        # Get all particle positions as a numpy array
        positions = np.array([particle.position for particle in self.particles])

        # Compute the grid indices for all particles
        i_indices, j_indices = self.get_cell_indices(positions)

        # Efficiently add particles to the grid
        for particle, i, j in zip(self.particles, i_indices, j_indices):
            self.grid[i][j].append(particle)


    def collision(self, p1: Particle, p2: Particle, dist: float, delta_pos: np.ndarray, restitution: float) -> None:
        """
        Resolve a collision between two particles, updating their velocities and positions.
        
        :param p1: First particle.
        :param p2: Second particle.
        :param dist: Distance between the particles.
        :param delta_pos: Vector from p2 to p1.
        """
        delta_vel = p1.velocity - p2.velocity
        norm_delta_pos = delta_pos / dist
        velocity_along_axis = np.dot(delta_vel, norm_delta_pos)

        # If the particles are moving apart, skip the collision resolution (Early exit)
        if velocity_along_axis > 0:
            return

        # Compute impulse magnitude for elastic collision
        impulse_magnitude = (2 * velocity_along_axis) / (p1.mass + p2.mass)

        # Update velocities and apply the restitution coefficient (elastic: resitution = 1, inelastic: 0 < restitution < 1)
        p1.velocity -= restitution * impulse_magnitude * p2.mass * norm_delta_pos
        p2.velocity += restitution * impulse_magnitude * p1.mass * norm_delta_pos

        # Correct particle positions to resolve any overlap
        overlap = p1.radius + p2.radius - dist + self.epsilon
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
                cell_particles = self.grid[i][j]

                # Sort particles in the current cell by x-axis (or y-axis)
                cell_particles.sort(key=lambda p: p.position[0])  # Sort by x-axis, can switch to y-axis if desired

                # Check for collisions within the same cell
                for k in range(len(cell_particles)):
                    p1 = cell_particles[k]
                    for l in range(k + 1, len(cell_particles)):
                        p2 = cell_particles[l]

                        # Early exit if the distance along the x-axis exceeds the sum of radii
                        if p2.position[0] - p1.position[0] > (p1.radius + p2.radius):
                            break

                        # Check and resolve the collision if necessary
                        self.check_and_resolve_collision(p1, p2)

                # Check for collisions with neighboring cells in the "forward" direction
                for di in [0, 1]:
                    for dj in [0, 1] if di == 0 else [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid_width and 0 <= nj < self.grid_height:
                            neighbor_cell = self.grid[ni][nj]
                            for p1 in cell_particles:
                                for p2 in neighbor_cell:
                                    if p1 is not p2:
                                        # Only check neighboring cell particles that could collide along the x-axis
                                        if abs(p2.position[0] - p1.position[0]) <= (p1.radius + p2.radius):
                                            self.check_and_resolve_collision(p1, p2)


    def check_and_resolve_collision(self, p1: Particle, p2: Particle) -> None:
        """
        Check if two particles are colliding, and if so, resolve the collision.
        
        :param p1: First particle.
        :param p2: Second particle.
        """
        delta_pos = p1.position - p2.position
        dist = np.linalg.norm(delta_pos)
        restitution = (p1.restitution + p2.restitution) / 2


        # If particles overlap, resolve the collision
        if dist < (p1.radius + p2.radius - self.epsilon):
            self.collision(p1, p2, dist, delta_pos, restitution=restitution)


    def update_positions(self, dt: float) -> None:
        """
        Update the positions of all particles and check for wall collisions.
        Apply gravity to each particle.
        """
        for particle in self.particles:
            # Apply gravity if it's set
            if self.vertical_gravity != 0.0:
                #print("v berfore: ", particle.velocity)
                particle.velocity[1] -= self.vertical_gravity * dt
                #print("v after: ", particle.velocity)
                #print("="*50)

            # Check and resolve wall collisions before moving particles
            self.check_wall_collision(particle, dt)

            # Move the particle
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
    def __init__(self, width: float, height: float, vertical_gravity: float = 0.0, restitution: float = 1.0) -> None:
        """
        Initialize a rectangular scene with specified width and height.
        
        :param width: Width of the rectangle.
        :param height: Height of the rectangle.
        """
        super().__init__(width=width, height=height, cell_size=width / 20, vertical_gravity=vertical_gravity)
        self.width = width
        self.height = height
        self.restitution = restitution

    def check_wall_collision(self, particle: Particle, dt: float) -> None:
        """
        Check and resolve wall collisions for a particle in the rectangular scene.
        
        :param particle: The particle to check for wall collision.
        :param dt: Time step for continuous collision detection.
        """
        next_position = particle.position + particle.velocity * dt

        # Check for left or right wall collisions
        if next_position[0] - particle.radius < 0:
            # Reflect velocity for left wall with restitution
            particle.velocity[0] = -particle.velocity[0] * particle.restitution
            particle.position[0] = 2 * particle.radius - particle.position[0]
        elif next_position[0] + particle.radius > self.width:
            # Reflect velocity for right wall with restitution
            particle.velocity[0] = -particle.velocity[0] * particle.restitution
            particle.position[0] = 2 * (self.width - particle.radius) - particle.position[0]

        # Check for top or bottom wall collisions
        if next_position[1] - particle.radius < 0:
            # Reflect velocity for bottom wall with restitution
            particle.velocity[1] = -particle.velocity[1] * particle.restitution
            particle.position[1] = 2 * particle.radius - particle.position[1]
        elif next_position[1] + particle.radius > self.height:
            # Reflect velocity for top wall with restitution
            particle.velocity[1] = -particle.velocity[1] * particle.restitution
            particle.position[1] = 2 * (self.height - particle.radius) - particle.position[1]


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
            velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])*10 
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

    


class Circle(Scene):
    def __init__(self, radius: float, vertical_gravity: float = 0.0, restitution: float = 1.0) -> None:
        """
        Initialize a circular scene with a given radius.
        
        :param radius: Radius of the circular boundary.
        """
        super().__init__(width=radius * 2, height=radius * 2, cell_size=radius / 20, vertical_gravity=vertical_gravity)
        self.radius = radius
        self.restitution = restitution

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
            new_particle = Particle(position, velocity, color=tuple(np.random.uniform(0, 1, 3)), radius=radius, density=density,
                                    restitution=self.restitution)
            if self.is_valid_particle(new_particle):
                self.particles.append(new_particle)
            attempts += 1
        if attempts >= max_attempts:
            raise RuntimeError("Unable to place all particles without overlap after many attempts.")
        


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

        super().__init__(total_width, max_height, cell_size=min(left_width, right_width) / 20, vertical_gravity=vertical_gravity)

        self.left_width = left_width
        self.left_height = left_height
        self.right_width = right_width
        self.right_height = right_height
        self.pipe_width = pipe_width
        self.pipe_height = pipe_height
        self.epsilon = 1e-9  # Floating-point tolerance
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


class Triangle(Scene):
    def __init__(self, base: float, height: float, angle: float = 0, vertical_gravity: float = 0.0, restitution: float = 1.0) -> None:
        """
        Initialize a triangular scene with specified base, height, and angle.
        
        :param base: Base length of the triangle.
        :param height: Height of the triangle.
        :param angle: Angle between the base and the line to the top vertex (in radians), default is 0.
        """
        super().__init__(width=base, height=height, cell_size=base / 20, vertical_gravity=vertical_gravity)
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

    def generate_particles(self, num_particles: int, radius_range: (float, float), density_range: (float, float)) -> None:
        """
        Generate a given number of particles with random positions, velocities, radius, and density within the triangle.
        
        :param num_particles: Number of particles to generate.
        :param radius_range: Tuple of (min_radius, max_radius) for particle radii.
        :param density_range: Tuple of (min_density, max_density) for particle densities.
        :raises RuntimeError: If unable to place all particles without overlap after many attempts.
        """
        max_attempts = 1000
        attempts = 0
        while len(self.particles) < num_particles and attempts < max_attempts:
            position = self._generate_random_position()
            velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)]) * 100
            radius = random.uniform(*radius_range)
            density = random.uniform(*density_range)
            new_particle = Particle(position, velocity, color=tuple(np.random.uniform(0, 1, 3)), radius=radius, density=density,
                                    restitution=self.restitution)
            if self._is_valid_particle_(new_particle):
                self.particles.append(new_particle)
            attempts += 1
        if attempts >= max_attempts:
            raise RuntimeError("Unable to place all particles without overlap after many attempts.")
