import numpy as np 
from numba import njit

@njit
def circle_check_wall_collision_numba(particle_pos, particle_vel, particle_rad, radius, particle_rest, dt: float) -> None:
        # Calculate distance from the particle to the center
        distance_to_center = np.linalg.norm(particle_pos)
        
        # Check if the particle is colliding with the wall
        if distance_to_center + particle_rad >= radius:
            # Calculate penetration depth
            penetration_depth = distance_to_center + particle_rad - radius
            
            # Normal vector (normalized direction from the center to the particle)
            normal = particle_pos / distance_to_center
            
            # Correct the particle's position to resolve the collision
            particle_pos -= normal * penetration_depth
            
            # Reflect the particle's velocity along the normal direction
            velocity_dot_normal = np.dot(particle_vel, normal)
            particle_vel -= 2 * particle_rest * velocity_dot_normal * normal
        return particle_pos, particle_vel 


@njit
def collision_numba(p1_vel, p1_pos, p1_mass, p1_rad,
                    p2_vel, p2_pos, p2_mass, p2_rad,
                    dist, delta_pos, restitution: float):
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
        delta_vel = p1_vel - p2_vel

        # Compute the velocity component along the collision normal
        velocity_along_normal = np.dot(delta_vel, norm_delta_pos)

        # If the particles are moving apart (velocity along normal is positive), skip collision resolution
        if velocity_along_normal > 0:
            return p1_pos, p1_vel, p2_pos, p2_vel

        # Calculate the impulse magnitude for a perfectly elastic collision
        # (taking into account the coefficient of restitution)
        impulse_magnitude = (-(1 + restitution) * velocity_along_normal) / (1 / p1_mass + 1 / p2_mass)

        # Calculate the impulse vector
        impulse = impulse_magnitude * norm_delta_pos

        # Update velocities based on the impulse
        p1_vel += impulse / p1_mass
        p2_vel -= impulse / p2_mass

        # Position correction to prevent overlap 
        overlap = (p1_rad + p2_rad) - dist 
        correction = norm_delta_pos * (overlap / 2)
        p1_pos += correction
        p2_pos -= correction
        return p1_pos, p1_vel, p2_pos, p2_vel
