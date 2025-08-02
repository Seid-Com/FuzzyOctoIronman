import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import io
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from algorithms.pso import PSO
from algorithms.fuzzy_dbscan import FuzzyDBSCAN
from utils.data_processing import preprocess_data, z_score_normalize
from utils.visualization import create_cluster_visualization, create_comparison_plot
from utils.metrics import calculate_performance_metrics, statistical_validation

# Configure page
st.set_page_config(
    page_title="Adaptive Fuzzy-PSO DBSCAN",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🏙️ Adaptive Fuzzy-PSO DBSCAN for Smart City Data Analysis")
    st.markdown("""
    An enhanced density-based clustering approach implementing the research by Seid Mehammed Abdu and Md Nasre Alam.
    This prototype demonstrates fuzzy logic integration with PSO-optimized DBSCAN for urban infrastructure analysis.
    """)

    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV Dataset",
        type=['csv'],
        help="Upload a CSV file with columns: facility_type, year_established, latitude, longitude"
    )
    
    if uploaded_file is not None:
        # Load and display data
        try:
            data = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Dataset loaded: {len(data)} records")
            
            # Validate required columns
            required_columns = ['facility_type', 'year_established', 'latitude', 'longitude']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                st.stop()
            
            # Display data overview
            st.subheader("📊 Dataset Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(data))
            with col2:
                st.metric("Facility Types", data['facility_type'].nunique())
            with col3:
                st.metric("Time Range", f"{int(data['year_established'].min())}-{int(data['year_established'].max())}")
            
            # Show data sample
            with st.expander("View Data Sample", expanded=False):
                st.dataframe(data.head(10))
            
            # PSO Parameters
            st.sidebar.subheader("PSO Parameters")
            n_particles = st.sidebar.slider("Number of Particles", 10, 50, 20)
            n_iterations = st.sidebar.slider("Iterations", 20, 100, 50)
            w = st.sidebar.slider("Inertia Weight", 0.1, 1.0, 0.5)
            c1 = st.sidebar.slider("Cognitive Coefficient", 0.1, 3.0, 2.0)
            c2 = st.sidebar.slider("Social Coefficient", 0.1, 3.0, 2.0)
            
            # Fuzzy Parameters
            st.sidebar.subheader("Fuzzy Parameters")
            alpha = st.sidebar.slider("Alpha (steepness)", 0.1, 5.0, 1.0)
            beta = st.sidebar.slider("Beta (spread)", 0.1, 5.0, 2.0)
            
            # Time-based analysis option
            temporal_analysis = st.sidebar.checkbox("Enable Temporal Analysis (Pre/Post 2010)")
            
            if st.sidebar.button("🚀 Run Clustering Analysis", type="primary"):
                run_clustering_analysis(data, n_particles, n_iterations, w, c1, c2, alpha, beta, temporal_analysis)
                
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    else:
        # Show example data format
        st.info("👆 Please upload a CSV dataset to begin analysis")
        
        st.subheader("📋 Expected Data Format")
        example_data = pd.DataFrame({
            'facility_type': ['education', 'health', 'entertainment', 'education'],
            'year_established': [2005, 2012, 2008, 2015],
            'latitude': [9.0307, 9.0397, 9.0257, 9.0407],
            'longitude': [38.7407, 38.7507, 38.7357, 38.7557]
        })
        st.dataframe(example_data)
        
        st.markdown("""
        **Required Columns:**
        - `facility_type`: Categorical (education, health, entertainment)
        - `year_established`: Numerical year
        - `latitude`: Geographic coordinate
        - `longitude`: Geographic coordinate
        """)

def run_clustering_analysis(data, n_particles, n_iterations, w, c1, c2, alpha, beta, temporal_analysis):
    """Execute the complete clustering analysis pipeline"""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Data preprocessing
        status_text.text("Step 1/6: Preprocessing data...")
        progress_bar.progress(10)
        
        processed_data = preprocess_data(data)
        normalized_features = z_score_normalize(processed_data[['year_established', 'latitude', 'longitude']])
        
        # Step 2: Initialize PSO
        status_text.text("Step 2/6: Initializing PSO optimization...")
        progress_bar.progress(20)
        
        pso = PSO(
            n_particles=n_particles,
            n_iterations=n_iterations,
            w=w, c1=c1, c2=c2
        )
        
        if temporal_analysis:
            # Split data by 2010
            pre_2010 = processed_data[processed_data['year_established'] < 2010]
            post_2010 = processed_data[processed_data['year_established'] >= 2010]
            
            datasets = {
                'Pre-2010': (pre_2010, z_score_normalize(pre_2010[['year_established', 'latitude', 'longitude']])),
                'Post-2010': (post_2010, z_score_normalize(post_2010[['year_established', 'latitude', 'longitude']]))
            }
        else:
            datasets = {'Full Dataset': (processed_data, normalized_features)}
        
        results = {}
        
        for dataset_name, (orig_data, norm_data) in datasets.items():
            if len(orig_data) < 10:  # Skip if too few points
                continue
                
            st.subheader(f"🔬 Analysis Results: {dataset_name}")
            
            # Step 3: PSO Parameter Optimization
            status_text.text(f"Step 3/6: Optimizing parameters for {dataset_name}...")
            progress_bar.progress(30)
            
            best_params = pso.optimize(norm_data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Optimized Eps", f"{best_params['eps']:.4f}")
            with col2:
                st.metric("Optimized MinPts", int(best_params['min_pts']))
            
            # Step 4: Fuzzy DBSCAN Clustering
            status_text.text(f"Step 4/6: Running Fuzzy DBSCAN for {dataset_name}...")
            progress_bar.progress(50)
            
            fuzzy_dbscan = FuzzyDBSCAN(
                eps=best_params['eps'],
                min_pts=int(best_params['min_pts']),
                alpha=alpha,
                beta=beta
            )
            
            fuzzy_labels, fuzzy_memberships = fuzzy_dbscan.fit_predict(norm_data)
            
            # Step 5: Standard DBSCAN for comparison
            status_text.text(f"Step 5/6: Running standard DBSCAN for comparison...")
            progress_bar.progress(70)
            
            dbscan = DBSCAN(eps=best_params['eps'], min_samples=int(best_params['min_pts']))
            standard_labels = dbscan.fit_predict(norm_data)
            
            # Calculate metrics
            fuzzy_silhouette = calculate_performance_metrics(norm_data, fuzzy_labels)
            standard_silhouette = calculate_performance_metrics(norm_data, standard_labels)
            
            results[dataset_name] = {
                'fuzzy_silhouette': fuzzy_silhouette,
                'standard_silhouette': standard_silhouette,
                'fuzzy_labels': fuzzy_labels,
                'standard_labels': standard_labels,
                'fuzzy_memberships': fuzzy_memberships,
                'data': orig_data,
                'normalized_data': norm_data
            }
            
            # Display performance metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fuzzy-PSO DBSCAN Silhouette", f"{fuzzy_silhouette:.4f}")
            with col2:
                st.metric("Standard DBSCAN Silhouette", f"{standard_silhouette:.4f}")
            with col3:
                improvement = ((fuzzy_silhouette - standard_silhouette) / standard_silhouette) * 100
                st.metric("Improvement", f"{improvement:.1f}%", delta=f"{fuzzy_silhouette - standard_silhouette:.4f}")
            
            # Step 6: Visualizations
            status_text.text(f"Step 6/6: Creating visualizations for {dataset_name}...")
            progress_bar.progress(90)
            
            create_visualizations(orig_data, fuzzy_labels, standard_labels, fuzzy_memberships, dataset_name)
        
        # Statistical validation across all results
        if len(results) > 1:
            perform_statistical_validation(results)
        
        # Export functionality
        create_export_section(results)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis completed successfully!")
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.exception(e)

def create_visualizations(data, fuzzy_labels, standard_labels, fuzzy_memberships, dataset_name):
    """Create comprehensive visualizations"""
    
    # Cluster comparison visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Standard DBSCAN - {dataset_name}")
        fig_standard = create_cluster_visualization(data, standard_labels, "Standard DBSCAN")
        st.plotly_chart(fig_standard, use_container_width=True)
    
    with col2:
        st.subheader(f"Fuzzy-PSO DBSCAN - {dataset_name}")
        fig_fuzzy = create_cluster_visualization(data, fuzzy_labels, "Fuzzy-PSO DBSCAN")
        st.plotly_chart(fig_fuzzy, use_container_width=True)
    
    # Interactive map visualization
    st.subheader(f"🗺️ GIS Visualization - {dataset_name}")
    
    # Create Folium map
    center_lat = data['latitude'].mean()
    center_lon = data['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Color palette for clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
    
    # Add fuzzy clustering points
    unique_labels = np.unique(fuzzy_labels)
    for i, (idx, row) in enumerate(data.iterrows()):
        label = fuzzy_labels[i]
        if label == -1:  # Noise points
            color = 'black'
            popup_text = f"Noise Point<br>Type: {row['facility_type']}<br>Year: {row['year_established']}"
        else:
            color = colors[label % len(colors)]
            membership = fuzzy_memberships[i][label] if label < len(fuzzy_memberships[i]) else 0
            popup_text = f"Cluster {label}<br>Type: {row['facility_type']}<br>Year: {row['year_established']}<br>Membership: {membership:.3f}"
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            popup=popup_text,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    # Display map
    map_data = st_folium(m, width=700, height=500)
    
    # Fuzzy membership visualization
    st.subheader(f"🔍 Fuzzy Membership Analysis - {dataset_name}")
    
    if len(fuzzy_memberships) > 0 and len(fuzzy_memberships[0]) > 1:
        # Create membership heatmap
        membership_data = []
        for i, memberships in enumerate(fuzzy_memberships):
            for cluster_id, membership in enumerate(memberships):
                membership_data.append({
                    'Point': i,
                    'Cluster': f'Cluster {cluster_id}',
                    'Membership': membership
                })
        
        if membership_data:
            membership_df = pd.DataFrame(membership_data)
            
            # Sample first 50 points for better visualization
            if len(membership_df['Point'].unique()) > 50:
                sample_points = membership_df['Point'].unique()[:50]
                membership_df = membership_df[membership_df['Point'].isin(sample_points)]
            
            fig_heatmap = px.density_heatmap(
                membership_df, 
                x='Point', 
                y='Cluster', 
                z='Membership',
                title=f"Fuzzy Membership Heatmap - {dataset_name}",
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

def perform_statistical_validation(results):
    """Perform statistical validation across results"""
    
    st.subheader("📈 Statistical Validation")
    
    fuzzy_scores = [result['fuzzy_silhouette'] for result in results.values()]
    standard_scores = [result['standard_silhouette'] for result in results.values()]
    
    if len(fuzzy_scores) >= 2:
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(fuzzy_scores, standard_scores)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("T-statistic", f"{t_stat:.4f}")
        with col2:
            st.metric("P-value", f"{p_value:.6f}")
        with col3:
            significance = "Significant" if p_value < 0.05 else "Not Significant"
            st.metric("Result", significance)
        
        # Results interpretation
        if p_value < 0.05:
            st.success("✅ The improvement is statistically significant (p < 0.05)")
        else:
            st.warning("⚠️ The improvement is not statistically significant (p ≥ 0.05)")
        
        # Performance comparison chart
        comparison_df = pd.DataFrame({
            'Dataset': list(results.keys()) * 2,
            'Method': ['Fuzzy-PSO DBSCAN'] * len(results) + ['Standard DBSCAN'] * len(results),
            'Silhouette Score': fuzzy_scores + standard_scores
        })
        
        fig_comparison = px.bar(
            comparison_df,
            x='Dataset',
            y='Silhouette Score',
            color='Method',
            barmode='group',
            title='Performance Comparison Across Datasets'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

def create_export_section(results):
    """Create export functionality for results"""
    
    st.subheader("📥 Export Results")
    
    # Prepare summary data
    summary_data = []
    for dataset_name, result in results.items():
        summary_data.append({
            'Dataset': dataset_name,
            'Fuzzy_PSO_DBSCAN_Silhouette': result['fuzzy_silhouette'],
            'Standard_DBSCAN_Silhouette': result['standard_silhouette'],
            'Improvement': result['fuzzy_silhouette'] - result['standard_silhouette'],
            'Improvement_Percentage': ((result['fuzzy_silhouette'] - result['standard_silhouette']) / result['standard_silhouette']) * 100
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export
        csv_buffer = io.StringIO()
        summary_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📊 Download Results CSV",
            data=csv_buffer.getvalue(),
            file_name="fuzzy_pso_dbscan_results.csv",
            mime="text/csv"
        )
    
    with col2:
        # Detailed results
        if st.button("📋 Show Detailed Results"):
            st.dataframe(summary_df)

if __name__ == "__main__":
    main()
