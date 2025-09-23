import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from pyvis.network import Network
import tempfile
import os
from datetime import datetime, timedelta
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

# Import custom utilities
from utils.preprocessing import DataPreprocessor
from utils.pattern_mining import PatternMiner
from utils.visualization import Visualizer
from utils.insights import InsightGenerator

# Configure Streamlit page
st.set_page_config(
    page_title="Digital Library Analytics",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache configuration
@st.cache_data
def load_and_process_data(library_file_content, metadata_file_content):
    """Cache the data loading and processing step"""
    import io
    
    # Convert uploaded files to DataFrames
    library_df = pd.read_csv(io.StringIO(library_file_content))
    metadata_df = pd.read_csv(io.StringIO(metadata_file_content))
    
    # Process data
    preprocessor = DataPreprocessor()
    merged_data = preprocessor.merge_data(library_df, metadata_df)
    
    return merged_data

@st.cache_data
def generate_association_rules_cached(_data, min_support, min_confidence, min_lift):
    """Cache association rules generation"""
    pattern_miner = PatternMiner()
    rules_df = pattern_miner.generate_association_rules(
        _data, min_support, min_confidence, min_lift
    )
    return rules_df

@st.cache_data
def generate_insights_cached(_data):
    """Cache insights generation"""
    insight_generator = InsightGenerator()
    insights = insight_generator.generate_insights(_data)
    return insights

@st.cache_data
def create_visualizations_cached(_data):
    """Cache visualization creation"""
    visualizer = Visualizer()
    
    viz_data = {
        'top_books': visualizer.plot_top_borrowed_books(_data),
        'trends': visualizer.plot_borrowing_trends(_data),
        'ratings': visualizer.plot_rating_distribution(_data),
        'devices': visualizer.plot_device_usage(_data),
        'session_duration': visualizer.plot_session_duration_by_device(_data),
        'device_ratings': visualizer.plot_rating_by_device(_data)
    }
    
    return viz_data

# Page config
st.set_page_config(
    page_title="Digital Library Analytics",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #f8f9fa;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">📚 Digital Library Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'merged_data' not in st.session_state:
        st.session_state.merged_data = None
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = DataPreprocessor()
    
    # Sidebar
    st.sidebar.markdown("## 📂 Data Upload")
    
    # File upload
    library_file = st.sidebar.file_uploader(
        "Upload Digital Library Dataset",
        type=['csv'],
        help="Upload your digital_library_dataset.csv file"
    )
    
    metadata_file = st.sidebar.file_uploader(
        "Upload Metadata",
        type=['csv'],
        help="Upload your metadata.csv file"
    )
    
    if library_file and metadata_file:
        # Convert uploaded files to string content for caching
        library_content = str(library_file.read(), "utf-8")
        metadata_content = str(metadata_file.read(), "utf-8")
        
        if st.sidebar.button("🔄 Load & Process Data"):
            with st.spinner("Processing data..."):
                try:
                    # Use cached data loading
                    merged_data = load_and_process_data(library_content, metadata_content)
                    st.session_state.merged_data = merged_data
                    st.session_state.data_loaded = True
                    
                    # Display basic info
                    st.sidebar.info(f"📊 Library: {len(merged_data)} records, {merged_data['book_id'].nunique()} unique books")
                    
                    # Check merge success
                    books_with_metadata = merged_data['title'].notna().sum()
                    total_records = len(merged_data)
                    merge_success_rate = (books_with_metadata / total_records) * 100
                    
                    st.sidebar.success("✅ Data loaded successfully!")
                    st.sidebar.success(f"📈 Merge success: {merge_success_rate:.1f}% of records have book metadata")
                    
                    if merge_success_rate < 90:
                        missing_books = merged_data[merged_data['title'].isna()]['book_id'].nunique()
                        st.sidebar.warning(f"⚠️ {missing_books} book IDs from library dataset not found in metadata")
                    
                except Exception as e:
                    st.sidebar.error(f"❌ Error loading data: {str(e)}")
                    st.sidebar.info("💡 Tip: Make sure both files have 'book_id' column for merging")
    
    if st.session_state.data_loaded:
        # Sidebar filters
        st.sidebar.markdown("## 📊 Analysis Parameters")
        
        min_support = st.sidebar.slider(
            "Minimum Support",
            min_value=0.01,
            max_value=0.5,
            value=0.05,
            step=0.01,
            help="Minimum support for frequent itemsets"
        )
        
        min_confidence = st.sidebar.slider(
            "Minimum Confidence", 
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence for association rules"
        )
        
        min_lift = st.sidebar.slider(
            "Minimum Lift",
            min_value=1.0,
            max_value=5.0,
            value=1.2,
            step=0.1,
            help="Minimum lift for association rules"
        )
        
        # Search functionality
        st.sidebar.markdown("## 🔍 Search")
        search_term = st.sidebar.text_input(
            "Search books by title",
            placeholder="Enter book title..."
        )
        
        # Main content
        display_dashboard(
            st.session_state.merged_data,
            min_support,
            min_confidence, 
            min_lift,
            search_term
        )
        
        # Export functionality
        st.sidebar.markdown("## 📤 Export Results")
        if st.sidebar.button("Download Analysis Results"):
            export_results(st.session_state.merged_data)
    
    else:
        # Welcome message
        st.markdown("""
        ### Welcome to Digital Library Analytics! 📖
        
        This application helps you analyze borrowing patterns and generate book recommendations from your digital library data.
        
        **To get started:**
        1. Upload your `digital_library_dataset.csv` file
        2. Upload your `metadata.csv` file
        3. Click "Load & Process Data"
        
        **Features:**
        - 📊 **Dashboard**: Visualize borrowing trends and patterns
        - 🔗 **Association Rules**: Discover book recommendation relationships
        - 💡 **Insights**: Get automatic insights from your data
        - 📱 **Device Analysis**: Understand user behavior across devices
        """)
        
        # Sample data structure
        with st.expander("📋 Expected Data Structure"):
            st.markdown("""
            **digital_library_dataset.csv should contain:**
            - `user_id`: Unique borrower identifier
            - `book_id`: Book identifier
            - `borrow_timestamp`: Date of borrowing  
            - `return_timestamp`: Date of return
            - `rating`: User rating (1-5)
            - `device_type`: Borrowing device (desktop, tablet, mobile)
            - `session_duration`: Time spent (seconds)
            - `action_type`: borrow/preview/etc
            - `recommendation_score`: Recommendation indicator
            
            **metadata.csv should contain:**
            - `book_id`: Matches with main dataset
            - `title`: Human-readable book title
            - `author`: Author's name
            - `year`: Year of publication
            """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Developed with Python & Streamlit | For Educational Purposes</p>
        <p>📚 Digital Library Pattern Analysis & Recommendation System</p>
    </div>
    """, unsafe_allow_html=True)

def display_dashboard(data, min_support, min_confidence, min_lift, search_term):
    """Display the main dashboard with all analytics"""
    
    # Filter data if search term is provided
    display_data = data.copy()
    if search_term:
        display_data = data[data['title'].str.contains(search_term, case=False, na=False)]
        st.info(f"🔍 Showing results for: '{search_term}' ({len(display_data)} records)")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_borrows = len(display_data[display_data['action_type'] == 'borrow'])
        st.markdown(f"""
        <div class="metric-container">
            <h3>{total_borrows:,}</h3>
            <p>Total Borrows</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        unique_users = display_data['user_id'].nunique()
        st.markdown(f"""
        <div class="metric-container">
            <h3>{unique_users:,}</h3>
            <p>Active Users</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        unique_books = display_data['book_id'].nunique()
        st.markdown(f"""
        <div class="metric-container">
            <h3>{unique_books:,}</h3>
            <p>Unique Books</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_rating = display_data['rating'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <h3>{avg_rating:.1f}⭐</h3>
            <p>Avg Rating</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔗 Association Rules", "💡 Insights", "📱 Device Analysis"])
    
    with tab1:
        display_dashboard_charts(display_data)
    
    with tab2:
        display_association_rules(display_data, min_support, min_confidence, min_lift)
    
    with tab3:
        display_insights(display_data)
    
    with tab4:
        display_device_analysis(display_data)

def display_dashboard_charts(data):
    """Display dashboard charts using cached visualizations"""
    
    # Use cached visualizations
    viz_data = create_visualizations_cached(data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Top Borrowed Books")
        st.plotly_chart(viz_data['top_books'], use_container_width=True)
    
    with col2:
        st.subheader("📈 Borrowing Trends Over Time")
        st.plotly_chart(viz_data['trends'], use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("⭐ Rating Distribution")
        st.plotly_chart(viz_data['ratings'], use_container_width=True)
    
    with col4:
        st.subheader("📱 Device Usage")
        st.plotly_chart(viz_data['devices'], use_container_width=True)

def display_association_rules(data, min_support, min_confidence, min_lift):
    """Display association rules analysis with caching"""
    st.subheader("🔗 Book Association Rules")
    
    with st.spinner("Mining association rules..."):
        try:
            # Use cached association rules generation
            rules_df = generate_association_rules_cached(
                data, min_support, min_confidence, min_lift
            )
            
            if len(rules_df) == 0:
                st.warning("No association rules found with the current parameters. Try lowering the thresholds.")
                return
            
            # Display rules table
            st.subheader("📋 Association Rules Table")
            
            # Format the rules for better display
            display_rules = rules_df.copy()
            display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
            
            # Round numerical columns
            numerical_cols = ['support', 'confidence', 'lift', 'conviction']
            for col in numerical_cols:
                if col in display_rules.columns:
                    display_rules[col] = display_rules[col].round(3)
            
            st.dataframe(
                display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']],
                use_container_width=True
            )
            
            # Network visualization
            st.subheader("🌐 Association Rules Network")
            
            if len(rules_df) > 0:
                pattern_miner = PatternMiner()
                network_html = pattern_miner.create_network_visualization(rules_df)
                if network_html:
                    st.components.v1.html(network_html, height=600)
                else:
                    st.info("Network visualization not available for current rules.")
            
        except Exception as e:
            st.error(f"Error generating association rules: {str(e)}")

def display_insights(data):
    """Display automated insights with caching"""
    st.subheader("💡 Automated Insights")
    
    # Use cached insights generation
    insights = generate_insights_cached(data)
    
    for insight in insights:
        st.markdown(f"""
        <div class="insight-box">
            <strong>{insight['title']}</strong><br>
            {insight['description']}
        </div>
        """, unsafe_allow_html=True)

def display_device_analysis(data):
    """Display device-specific analysis with cached visualizations"""
    st.subheader("📱 Device Usage Analysis")
    
    # Use cached visualizations
    viz_data = create_visualizations_cached(data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Session Duration by Device")
        st.plotly_chart(viz_data['session_duration'], use_container_width=True)
    
    with col2:
        st.subheader("Rating by Device Type")
        st.plotly_chart(viz_data['device_ratings'], use_container_width=True)
    
    # Device statistics table
    st.subheader("📊 Device Statistics")
    device_stats = data.groupby('device_type').agg({
        'user_id': 'count',
        'session_duration': 'mean',
        'rating': 'mean'
    }).round(2)
    device_stats.columns = ['Total Actions', 'Avg Session (sec)', 'Avg Rating']
    st.dataframe(device_stats, use_container_width=True)

def export_results(data):
    """Export analysis results"""
    try:
        # Create export data
        export_data = data.copy()
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"library_analysis_results_{timestamp}.csv"
        
        # Convert to CSV
        csv_data = export_data.to_csv(index=False)
        
        st.sidebar.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv"
        )
        
        st.sidebar.success("✅ Export ready!")
        
    except Exception as e:
        st.sidebar.error(f"❌ Export failed: {str(e)}")

if __name__ == "__main__":
    main()