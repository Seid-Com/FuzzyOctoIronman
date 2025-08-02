import pytest
import pandas as pd
import numpy as np
from algorithms.pso import PSO
from algorithms.fuzzy_dbscan import FuzzyDBSCAN
from utils.data_processing import preprocess_data, z_score_normalize
from utils.metrics import calculate_performance_metrics

@pytest.fixture
def sample_data():
    """Create sample urban infrastructure data for testing"""
    return pd.DataFrame({
        'facility_type': ['education', 'health', 'entertainment', 'education', 'health'],
        'year_established': [2005, 2012, 2008, 2015, 2003],
        'latitude': [9.0307, 9.0397, 9.0257, 9.0407, 9.0287],
        'longitude': [38.7407, 38.7507, 38.7357, 38.7557, 38.7387]
    })

def test_data_preprocessing(sample_data):
    """Test data preprocessing functionality"""
    processed = preprocess_data(sample_data)
    
    assert 'facility_type_encoded' in processed.columns
    assert len(processed) == 5
    assert processed['facility_type_encoded'].dtype in ['int32', 'int64']

def test_z_score_normalization():
    """Test Z-score normalization"""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    normalized = z_score_normalize(pd.DataFrame(data))
    
    # Check that normalized data has mean ~0 and std ~1
    assert abs(np.mean(normalized)) < 1e-10
    assert abs(np.std(normalized) - 1.0) < 1e-10

def test_pso_optimization():
    """Test PSO parameter optimization"""
    # Generate simple test data
    np.random.seed(42)
    data = np.random.rand(20, 3)
    
    pso = PSO(n_particles=5, n_iterations=10)
    result = pso.optimize(data)
    
    assert 'eps' in result
    assert 'min_pts' in result
    assert 'fitness' in result
    assert result['eps'] > 0
    assert result['min_pts'] > 0

def test_fuzzy_dbscan():
    """Test Fuzzy DBSCAN clustering"""
    # Generate clustered test data
    np.random.seed(42)
    cluster1 = np.random.normal([0, 0], 0.1, (10, 2))
    cluster2 = np.random.normal([2, 2], 0.1, (10, 2))
    data = np.vstack([cluster1, cluster2])
    
    fuzzy_dbscan = FuzzyDBSCAN(eps=0.3, min_pts=3)
    labels, memberships = fuzzy_dbscan.fit_predict(data)
    
    assert len(labels) == 20
    assert len(memberships) == 20
    assert len(set(labels)) >= 2  # Should find at least 2 clusters

def test_performance_metrics():
    """Test performance metrics calculation"""
    # Create simple test data with known clusters
    data = np.array([[0, 0], [0, 1], [1, 0], [5, 5], [5, 6], [6, 5]])
    labels = np.array([0, 0, 0, 1, 1, 1])
    
    score = calculate_performance_metrics(data, labels)
    
    assert score >= -1.0 and score <= 1.0
    assert isinstance(score, float)

def test_invalid_data_handling():
    """Test handling of invalid data"""
    invalid_data = pd.DataFrame({
        'facility_type': ['education'],
        'year_established': [2005],
        'latitude': [91],  # Invalid latitude
        'longitude': [38.7407]
    })
    
    processed = preprocess_data(invalid_data)
    assert len(processed) == 0  # Should filter out invalid coordinates

if __name__ == "__main__":
    pytest.main([__file__])