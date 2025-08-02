import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def create_cluster_visualization(data, labels, title):
    """
    Create 2D scatter plot for cluster visualization
    """
    # Create visualization dataframe
    viz_data = data.copy()
    viz_data['cluster'] = labels
    viz_data['cluster'] = viz_data['cluster'].astype(str)
    
    # Replace -1 (noise) with 'Noise'
    viz_data['cluster'] = viz_data['cluster'].replace('-1', 'Noise')
    
    # Create scatter plot
    fig = px.scatter(
        viz_data,
        x='longitude',
        y='latitude',
        color='cluster',
        hover_data=['facility_type', 'year_established'],
        title=title,
        labels={
            'longitude': 'Longitude',
            'latitude': 'Latitude',
            'cluster': 'Cluster'
        }
    )
    
    # Update layout
    fig.update_layout(
        showlegend=True,
        height=500,
        xaxis_title="Longitude",
        yaxis_title="Latitude"
    )
    
    return fig

def create_comparison_plot(fuzzy_scores, standard_scores, datasets):
    """
    Create comparison bar plot for different methods
    """
    comparison_data = []
    
    for i, dataset in enumerate(datasets):
        comparison_data.extend([
            {'Dataset': dataset, 'Method': 'Fuzzy-PSO DBSCAN', 'Silhouette Score': fuzzy_scores[i]},
            {'Dataset': dataset, 'Method': 'Standard DBSCAN', 'Silhouette Score': standard_scores[i]}
        ])
    
    df = pd.DataFrame(comparison_data)
    
    fig = px.bar(
        df,
        x='Dataset',
        y='Silhouette Score',
        color='Method',
        barmode='group',
        title='Performance Comparison: Fuzzy-PSO DBSCAN vs Standard DBSCAN'
    )
    
    fig.update_layout(
        yaxis_title="Silhouette Score",
        xaxis_title="Dataset",
        showlegend=True
    )
    
    return fig

def create_membership_heatmap(memberships, max_points=50):
    """
    Create heatmap visualization for fuzzy memberships
    """
    if len(memberships) == 0:
        return None
    
    # Limit to first max_points for better visualization
    memberships_sample = memberships[:max_points]
    
    # Convert to matrix format
    max_clusters = max(len(m) for m in memberships_sample)
    membership_matrix = np.zeros((len(memberships_sample), max_clusters))
    
    for i, point_memberships in enumerate(memberships_sample):
        for j, membership in enumerate(point_memberships):
            if j < max_clusters:
                membership_matrix[i, j] = membership
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=membership_matrix.T,
        x=list(range(len(memberships_sample))),
        y=[f'Cluster {i}' for i in range(max_clusters)],
        colorscale='Viridis',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Fuzzy Membership Heatmap",
        xaxis_title="Data Points",
        yaxis_title="Clusters",
        height=400
    )
    
    return fig

def create_parameter_evolution_plot(pso_history):
    """
    Visualize PSO parameter evolution over iterations
    """
    if not pso_history:
        return None
    
    iterations = list(range(len(pso_history)))
    eps_values = [params['eps'] for params in pso_history]
    min_pts_values = [params['min_pts'] for params in pso_history]
    fitness_values = [params['fitness'] for params in pso_history]
    
    # Create subplot
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=iterations,
        y=eps_values,
        mode='lines+markers',
        name='Eps Parameter',
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=fitness_values,
        mode='lines+markers',
        name='Fitness (Silhouette Score)',
        yaxis='y2'
    ))
    
    # Update layout with secondary y-axis
    fig.update_layout(
        title="PSO Parameter Evolution",
        xaxis_title="Iteration",
        yaxis=dict(title="Eps Parameter", side="left"),
        yaxis2=dict(title="Fitness Score", side="right", overlaying="y"),
        showlegend=True
    )
    
    return fig

def create_facility_distribution_plot(data):
    """
    Create visualization of facility type distribution
    """
    facility_counts = data['facility_type'].value_counts()
    
    fig = px.pie(
        values=facility_counts.values,
        names=facility_counts.index,
        title="Distribution of Facility Types"
    )
    
    return fig

def create_temporal_analysis_plot(data):
    """
    Create temporal analysis visualization
    """
    # Count facilities by year
    yearly_counts = data.groupby(['year_established', 'facility_type']).size().reset_index(name='count')
    
    fig = px.line(
        yearly_counts,
        x='year_established',
        y='count',
        color='facility_type',
        title="Facility Development Over Time"
    )
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Facilities",
        showlegend=True
    )
    
    return fig
