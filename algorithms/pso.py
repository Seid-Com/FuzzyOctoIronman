import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class PSO:
    """
    Particle Swarm Optimization for DBSCAN parameter tuning
    Based on the methodology described in the research paper
    """
    
    def __init__(self, n_particles=20, n_iterations=50, w=0.5, c1=2.0, c2=2.0):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        
        # Parameter bounds for DBSCAN
        self.eps_bounds = (0.1, 2.0)
        self.min_pts_bounds = (3, 20)
        
    def initialize_particles(self):
        """Initialize particle positions and velocities"""
        positions = np.random.uniform(
            low=[self.eps_bounds[0], self.min_pts_bounds[0]],
            high=[self.eps_bounds[1], self.min_pts_bounds[1]],
            size=(self.n_particles, 2)
        )
        
        velocities = np.random.uniform(
            low=-0.1, high=0.1, 
            size=(self.n_particles, 2)
        )
        
        return positions, velocities
    
    def evaluate_fitness(self, data, eps, min_pts):
        """
        Evaluate fitness using Silhouette score
        Returns negative silhouette score for minimization
        """
        try:
            # Ensure min_pts is integer
            min_pts = max(2, int(min_pts))
            
            # Run DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_pts)
            labels = dbscan.fit_predict(data)
            
            # Check if we have valid clusters
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
            
            if n_clusters < 2:
                return -1.0  # Poor fitness for no clusters or single cluster
            
            # Calculate silhouette score (only for non-noise points)
            mask = labels != -1
            if np.sum(mask) < 2:
                return -1.0
                
            score = silhouette_score(data[mask], labels[mask])
            return score
            
        except Exception:
            return -1.0  # Return poor fitness on error
    
    def update_velocity(self, velocity, position, personal_best, global_best, r1, r2):
        """Update particle velocity according to PSO equations"""
        cognitive_component = self.c1 * r1 * (personal_best - position)
        social_component = self.c2 * r2 * (global_best - position)
        
        new_velocity = (self.w * velocity + 
                       cognitive_component + 
                       social_component)
        
        # Velocity clamping
        max_velocity = np.array([
            (self.eps_bounds[1] - self.eps_bounds[0]) * 0.1,
            (self.min_pts_bounds[1] - self.min_pts_bounds[0]) * 0.1
        ])
        
        new_velocity = np.clip(new_velocity, -max_velocity, max_velocity)
        return new_velocity
    
    def update_position(self, position, velocity):
        """Update particle position with boundary constraints"""
        new_position = position + velocity
        
        # Apply boundary constraints
        new_position[0] = np.clip(new_position[0], 
                                 self.eps_bounds[0], 
                                 self.eps_bounds[1])
        new_position[1] = np.clip(new_position[1], 
                                 self.min_pts_bounds[0], 
                                 self.min_pts_bounds[1])
        
        return new_position
    
    def optimize(self, data):
        """
        Main PSO optimization loop
        Returns best parameters found
        """
        # Initialize particles
        positions, velocities = self.initialize_particles()
        
        # Initialize personal and global bests
        personal_best_positions = positions.copy()
        personal_best_fitness = np.full(self.n_particles, -np.inf)
        
        global_best_position = None
        global_best_fitness = -np.inf
        
        # PSO main loop
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                # Evaluate current particle
                eps, min_pts = positions[i]
                fitness = self.evaluate_fitness(data, eps, min_pts)
                
                # Update personal best
                if fitness > personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = positions[i].copy()
                
                # Update global best
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = positions[i].copy()
            
            # Update velocities and positions
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2)
                
                velocities[i] = self.update_velocity(
                    velocities[i], positions[i],
                    personal_best_positions[i], global_best_position,
                    r1, r2
                )
                
                positions[i] = self.update_position(positions[i], velocities[i])
        
        # Return best parameters
        if global_best_position is not None:
            return {
                'eps': global_best_position[0],
                'min_pts': global_best_position[1],
                'fitness': global_best_fitness
            }
        else:
            # Fallback to reasonable defaults if optimization failed
            return {
                'eps': 0.5,
                'min_pts': 5,
                'fitness': -1.0
            }
