import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class FuzzyDBSCAN:
    """
    Fuzzy DBSCAN implementation based on the research methodology
    Integrates fuzzy membership functions with density-based clustering
    """
    
    def __init__(self, eps=0.5, min_pts=5, alpha=1.0, beta=2.0):
        self.eps = eps
        self.min_pts = min_pts
        self.alpha = alpha  # Controls steepness of membership function
        self.beta = beta    # Controls spread of exponential function
        
    def _find_neighbors(self, data, point_idx):
        """Find neighbors within eps distance"""
        distances = cdist([data[point_idx]], data)[0]
        return np.where(distances <= self.eps)[0]
    
    def _is_core_point(self, data, point_idx):
        """Check if point is a core point"""
        neighbors = self._find_neighbors(data, point_idx)
        return len(neighbors) >= self.min_pts
    
    def _compute_fuzzy_core_membership(self, data, point_idx):
        """
        Compute fuzzy core membership based on equation (4) from the paper
        μ_core(xi) = 1 / (1 + exp(-α(C - Cthreshold)))
        """
        neighbors = self._find_neighbors(data, point_idx)
        C = len(neighbors)  # Number of neighbors
        C_threshold = self.min_pts
        
        # Fuzzy core membership function
        membership = 1.0 / (1.0 + np.exp(-self.alpha * (C - C_threshold)))
        return membership, neighbors
    
    def _compute_cluster_membership(self, data, point_idx, cluster_centers, cluster_id):
        """
        Compute fuzzy membership to cluster based on equation (5)
        μ_ij = exp(-β * d_ij) / Σ(exp(-β * d_ik))
        """
        if len(cluster_centers) == 0:
            return 0.0
        
        point = data[point_idx]
        distances = []
        
        for center in cluster_centers:
            dist = np.linalg.norm(point - center)
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Compute exponential weights
        exp_weights = np.exp(-self.beta * distances)
        
        # Normalize to get membership probabilities
        total_weight = np.sum(exp_weights)
        if total_weight == 0:
            return 0.0
        
        # Return membership to the specific cluster
        if cluster_id < len(exp_weights):
            return exp_weights[cluster_id] / total_weight
        else:
            return 0.0
    
    def _expand_cluster(self, data, point_idx, neighbors, cluster_id, labels, visited, cluster_centers):
        """Expand cluster using density connectivity"""
        labels[point_idx] = cluster_id
        
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                neighbor_neighbors = self._find_neighbors(data, neighbor_idx)
                
                if len(neighbor_neighbors) >= self.min_pts:
                    # Add new neighbors to expand further
                    for nn in neighbor_neighbors:
                        if nn not in neighbors:
                            neighbors = np.append(neighbors, nn)
            
            if labels[neighbor_idx] == -1:  # Not yet assigned to any cluster
                labels[neighbor_idx] = cluster_id
                # Update cluster center
                cluster_points = np.where(np.array(labels) == cluster_id)[0]
                if len(cluster_points) > 0:
                    cluster_centers[cluster_id] = np.mean(data[cluster_points], axis=0)
            
            i += 1
        
        return labels, cluster_centers
    
    def fit_predict(self, data):
        """
        Main fuzzy DBSCAN algorithm
        Returns labels and fuzzy membership matrices
        """
        n_points = len(data)
        labels = np.full(n_points, -1)  # Initialize as noise
        visited = np.full(n_points, False)
        cluster_id = 0
        cluster_centers = {}
        
        # First pass: Identify core points and form initial clusters
        for i in range(n_points):
            if visited[i]:
                continue
                
            visited[i] = True
            core_membership, neighbors = self._compute_fuzzy_core_membership(data, i)
            
            # If it's a core point (high core membership)
            if core_membership > 0.5 and len(neighbors) >= self.min_pts:
                # Initialize cluster center
                cluster_centers[cluster_id] = data[i].copy()
                
                # Expand cluster
                labels, cluster_centers = self._expand_cluster(
                    data, i, neighbors, cluster_id, labels, visited, cluster_centers
                )
                cluster_id += 1
        
        # Second pass: Compute fuzzy memberships for all points
        fuzzy_memberships = []
        
        for i in range(n_points):
            point_memberships = []
            
            # Compute membership to each cluster
            for cid in range(cluster_id):
                if cid in cluster_centers:
                    membership = self._compute_cluster_membership(
                        data, i, [cluster_centers[cid]], 0
                    )
                else:
                    membership = 0.0
                point_memberships.append(membership)
            
            # Normalize memberships
            total_membership = sum(point_memberships)
            if total_membership > 0:
                point_memberships = [m / total_membership for m in point_memberships]
            
            fuzzy_memberships.append(point_memberships)
        
        # Assign final labels based on highest membership
        for i in range(n_points):
            if len(fuzzy_memberships[i]) > 0:
                max_membership_cluster = np.argmax(fuzzy_memberships[i])
                max_membership_value = fuzzy_memberships[i][max_membership_cluster]
                
                # Only assign if membership is significant
                if max_membership_value > 0.1:
                    labels[i] = max_membership_cluster
        
        return labels, fuzzy_memberships
