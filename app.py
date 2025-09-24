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

# Custom CSS for better design consistency and fixed alignment
st.markdown("""
<style>
    /* Remove default padding and margins */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 1rem 0;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
    }
    
    /* 🔑 FIX: Force Streamlit columns to have equal heights */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        align-items: stretch !important;
        gap: 1rem;
    }
    
    [data-testid="stHorizontalBlock"] > div {
        flex: 1;
        display: flex;
        flex-direction: column;
    }
    
    [data-testid="stHorizontalBlock"] > div > div {
        flex: 1;
        display: flex;
        flex-direction: column;
    }

    /* Alternative: More spacious version */
.feature-card {
    background: white;
    border: 2px solid #e9ecef;
    border-radius: 12px;
    padding: 1.5rem 1rem; /* More padding */
    margin: 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
    height: 160px;
    display: flex;
    flex-direction: column;
    justify-content: space-around; /* Even better space distribution */
    align-items: center;
    text-align: center;
    min-height: 160px;
}

.feature-card h4 {
    color: #2E86AB;
    margin: 0.2rem 0; /* Balanced margins */
    font-size: 1rem;
    font-weight: 600;
    line-height: 1.2;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0 0.5rem;
}

.feature-card h3 {
    margin: 0.2rem 0; /* More space around the number */
    font-size: 1.8rem;
    font-weight: bold;
    color: #333;
    line-height: 1.2;
    display: flex;
    align-items: center;
    justify-content: center;
}

.feature-card p {
    margin: 0.3rem 0; /* Balanced margins */
    font-size: 0.85rem;
    color: #666;
    line-height: 1.3;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 0 0.5rem;
}
    
    /* Insight boxes */
    .insight-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #dee2e6 100%);
        border-left: 5px solid #28a745;
        padding: 1.2rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .insight-box strong {
        color: #2E86AB;
        font-size: 1.1rem;
    }
    
    /* Navigation breadcrumbs */
    .breadcrumb {
        background: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        font-size: 0.9rem;
        color: #6c757d;
    }
    
    /* Table improvements */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Network graph container */
    .network-container {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 2rem;
        background: #f8f9fa;
        border-radius: 8px;
    }
    
    /* Sidebar improvements */
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f1f3f6;
        border-radius: 8px 8px 0 0;
        padding: 0 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2E86AB;
        color: white;
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
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Dashboard"
    
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
        
        # Search functionality with better explanation
        st.sidebar.markdown("## 🔍 Book Search & Filter")
        st.sidebar.markdown("""
        **Purpose**: Filter all data and visualizations to focus on specific books.
        
        **How it works**: Enter a book title to see:
        - How often it's borrowed
        - What other books are borrowed with it
        - User ratings and device preferences
        """)
        
        search_term = st.sidebar.text_input(
            "Search books by title",
            placeholder="e.g., 'Python', 'Data Science', 'Machine Learning'",
            help="This will filter all charts and analysis to show only results related to your search term"
        )
        
        if search_term:
            # Show search preview
            matching_books = st.session_state.merged_data[
                st.session_state.merged_data['title'].str.contains(search_term, case=False, na=False)
            ]['title'].unique()
            
            if len(matching_books) > 0:
                st.sidebar.success(f"✅ Found {len(matching_books)} matching books")
                with st.sidebar.expander("📚 Preview matching books"):
                    for book in matching_books[:5]:  # Show first 5
                        st.write(f"• {book}")
                    if len(matching_books) > 5:
                        st.write(f"... and {len(matching_books) - 5} more")
            else:
                st.sidebar.warning("❌ No books found matching your search")
        
        # Clear search button
        if search_term and st.sidebar.button("🗑️ Clear Search"):
            st.rerun()
        
        # Main content with improved tab structure
        display_data = st.session_state.merged_data.copy()
        if search_term:
            display_data = st.session_state.merged_data[
                st.session_state.merged_data['title'].str.contains(search_term, case=False, na=False)
            ]
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔗 Association Rules", "💡 Insights", "📱 Device Analysis"])
        
        with tab1:
            display_dashboard_content(display_data, min_support, min_confidence, min_lift, search_term)
        
        with tab2:
            display_association_rules(display_data, min_support, min_confidence, min_lift)
        
        with tab3:
            display_insights(display_data)
        
        with tab4:
            display_device_analysis(display_data)
        
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

def display_dashboard_content(data, min_support, min_confidence, min_lift, search_term):
    """Display the dashboard content with properly aligned metric cards"""
    
    # Breadcrumb navigation
    st.markdown('<div class="breadcrumb">🏠 Home > 📊 Dashboard</div>', unsafe_allow_html=True)
    
    # Filter data if search term is provided
    display_data = data.copy()
    if search_term:
        display_data = data[data['title'].str.contains(search_term, case=False, na=False)]
        st.info(f"🔍 Showing results for: '{search_term}' ({len(display_data)} records)")
    
    # Create perfectly aligned metric cards
    col1, col2, col3, col4 = st.columns(4)

    # Calculate metrics
    total_borrows = len(display_data[display_data['action_type'] == 'borrow'])
    unique_users = display_data['user_id'].nunique()
    unique_books = display_data['book_id'].nunique()
    avg_rating = display_data['rating'].mean()

    with col1:
        st.markdown(f"""
    <div class="feature-card">
        <h4>📊 Total Activity</h4>
        <h3>{total_borrows:,}</h3>
        <p>Actual borrows only</p>
    </div>
    """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
    <div class="feature-card">
        <h4>👥 Active Users</h4>
        <h3>{unique_users:,}</h3>
        <p>Unique borrowers</p>
    </div>
    """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
    <div class="feature-card">
        <h4>📚 Unique Books</h4>
        <h3>{unique_books:,}</h3>
        <p>Available in library</p>
    </div>
    """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
    <div class="feature-card">
        <h4>⭐ Avg Rating</h4>
        <h3>{avg_rating:.1f}/5.0</h3>
        <p>User satisfaction</p>
    </div>
    """, unsafe_allow_html=True)

    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main dashboard content
    st.markdown("---")
    display_dashboard_charts(display_data)

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
    
    # Breadcrumb navigation
    st.markdown('<div class="breadcrumb">🏠 Home > 🔗 Association Rules</div>', unsafe_allow_html=True)
    
    st.subheader("🔗 Book Association Rules")
    
    # Explanation of what this section does
    with st.expander("ℹ️ What are Association Rules?", expanded=False):
        st.markdown("""
        **Association Rules reveal hidden patterns in borrowing behavior:**
        
        - **"If users borrow Book A, they also tend to borrow Book B"**
        - **Support**: How frequently books appear together (higher = more common)
        - **Confidence**: How often the rule is correct (higher = more reliable)  
        - **Lift**: How much more likely books are borrowed together than by chance (>1.0 = positive relationship)
        
        **Example**: "Users who borrow 'Python Programming' also borrow 'Data Science' with 85% confidence"
        
        Use these insights to:
        - Recommend books to users
        - Understand reading preferences  
        - Plan book acquisitions
        """)
    
    with st.spinner("Mining association rules..."):
        try:
            # Use cached association rules generation
            rules_df = generate_association_rules_cached(
                data, min_support, min_confidence, min_lift
            )
            
            if len(rules_df) == 0:
                st.warning("No association rules found with the current parameters. Try lowering the thresholds.")
                st.info("💡 **Tip**: Lower the Support (0.01-0.03) and Confidence (0.3-0.5) values in the sidebar to find more rules.")
                return
            
            # Display summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📋 Total Rules Found", len(rules_df))
            with col2:
                strong_rules = len(rules_df[rules_df['lift'] >= 2.0])
                st.metric("💪 Strong Rules (Lift ≥ 2.0)", strong_rules)
            with col3:
                avg_confidence = rules_df['confidence'].mean()
                st.metric("🎯 Average Confidence", f"{avg_confidence:.1%}")
            
            # Display rules table (PRE-SORTED by Lift descending)
            st.subheader("📋 Association Rules Table")
            st.markdown("*Automatically sorted by Lift (strongest relationships first)*")
            
            # Format and sort the rules
            display_rules = rules_df.copy()
            
            # Sort by Lift (descending), then Confidence (descending)
            display_rules = display_rules.sort_values(['lift', 'confidence'], ascending=[False, False])
            
            display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
            
            # Round numerical columns
            numerical_cols = ['support', 'confidence', 'lift']
            for col in numerical_cols:
                if col in display_rules.columns:
                    display_rules[col] = display_rules[col].round(3)
            
            # Add rule strength indicators
            display_rules['strength'] = display_rules['lift'].apply(
                lambda x: "🔥 Very Strong" if x >= 3.0 else "💪 Strong" if x >= 2.0 else "📈 Moderate"
            )
            
            # Display table with better column names
            formatted_table = display_rules[[
                'antecedents', 'consequents', 'support', 'confidence', 'lift', 'strength'
            ]].copy()
            
            formatted_table.columns = [
                'If user borrows...', 'Then they also borrow...', 'Support', 'Confidence', 'Lift', 'Strength'
            ]
            
            st.dataframe(
                formatted_table,
                use_container_width=True,
                hide_index=True
            )
            
            # Top 5 strongest rules highlight
            st.subheader("🏆 Top 5 Strongest Relationships")

            # Reset index to ensure proper ordering and create a clean rule number
            top_5 = display_rules.head(5).reset_index(drop=True)

            for idx, rule in top_5.iterrows():
                st.markdown(f"""
                <div class="insight-box">
                    <strong>Rule #{idx + 1}</strong><br>
                    📚 <strong>If users borrow:</strong> {rule['antecedents']}<br>
                    📖 <strong>Then they also borrow:</strong> {rule['consequents']}<br>
                    🎯 <strong>Confidence:</strong> {rule['confidence']:.1%} | 📈 <strong>Lift:</strong> {rule['lift']:.2f}
                </div>
                """, unsafe_allow_html=True)
            
            # Network visualization with better container
            st.subheader("🌐 Association Rules Network")
            st.markdown("*Visual representation of book relationships - larger nodes = more popular books*")
            
            if len(rules_df) > 0:
                pattern_miner = PatternMiner()
                network_html = pattern_miner.create_network_visualization(rules_df[:15])  # Limit to top 15 for better performance
                if network_html:
                    st.markdown('<div class="network-container">', unsafe_allow_html=True)
                    st.components.v1.html(network_html, height=500)  # Reduced height
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("*💡 Tip: Click and drag nodes to explore the network. Hover for details.*")
                else:
                    st.info("Network visualization not available for current rules.")
            
        except Exception as e:
            st.error(f"Error generating association rules: {str(e)}")
            st.info("💡 Try uploading your data files and adjusting the parameters in the sidebar.")

def display_insights(data):
    """Display automated insights with caching"""
    
    # Breadcrumb navigation
    st.markdown('<div class="breadcrumb">🏠 Home > 💡 Automated Insights</div>', unsafe_allow_html=True)
    
    st.subheader("💡 Automated Insights")
    
    st.markdown("""
    **Smart Analysis**: Our system automatically analyzes your library data to uncover hidden patterns, 
    trends, and actionable recommendations for improving user experience.
    """)
    
    try:
        # Use cached insights generation
        insights = generate_insights_cached(data)
        
        if not insights or len(insights) == 0:
            st.warning("No insights generated. This might be due to insufficient data or data quality issues.")
            
            # Provide basic manual insights as fallback
            st.subheader("📊 Basic Data Overview")
            
            total_records = len(data)
            unique_users = data['user_id'].nunique()
            unique_books = data['book_id'].nunique()
            avg_rating = data['rating'].mean() if 'rating' in data.columns else None
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="insight-box">
                    <strong>📈 Library Activity Summary</strong><br>
                    Your library has {total_records:,} total interactions from {unique_users:,} users across {unique_books:,} books.
                    This indicates {'high' if total_records > 500 else 'moderate' if total_records > 100 else 'low'} activity levels.
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if avg_rating:
                    satisfaction = "high" if avg_rating >= 4.0 else "moderate" if avg_rating >= 3.0 else "low"
                    st.markdown(f"""
                    <div class="insight-box">
                        <strong>⭐ User Satisfaction</strong><br>
                        Average rating is {avg_rating:.1f}/5.0, indicating {satisfaction} user satisfaction.
                        {'Keep up the excellent content curation!' if avg_rating >= 4.0 else 'Consider reviewing lower-rated books.' if avg_rating < 3.0 else 'Good overall satisfaction levels.'}
                    </div>
                    """, unsafe_allow_html=True)
            
            return
        
        # Organize insights by priority and category
        high_priority = [i for i in insights if i.get('priority') == 'High']
        medium_priority = [i for i in insights if i.get('priority') == 'Medium'] 
        low_priority = [i for i in insights if i.get('priority') == 'Low']
        
        # Display high priority insights first
        if high_priority:
            st.subheader("🚨 High Priority Insights")
            st.markdown("*These insights require immediate attention*")
            
            for insight in high_priority:
                st.markdown(f"""
                <div class="insight-box" style="border-left-color: #dc3545;">
                    <strong>🔥 {insight['title']}</strong><br>
                    {insight['description']}
                </div>
                """, unsafe_allow_html=True)
        
        # Display medium priority insights
        if medium_priority:
            st.subheader("⚖️ Medium Priority Insights") 
            st.markdown("*Important patterns and trends*")
            
            for insight in medium_priority:
                st.markdown(f"""
                <div class="insight-box" style="border-left-color: #ffc107;">
                    <strong>📊 {insight['title']}</strong><br>
                    {insight['description']}
                </div>
                """, unsafe_allow_html=True)
        
        # Display low priority insights
        if low_priority:
            st.subheader("📝 Additional Insights")
            st.markdown("*Useful observations and trends*")
            
            for insight in low_priority:
                st.markdown(f"""
                <div class="insight-box" style="border-left-color: #28a745;">
                    <strong>💡 {insight['title']}</strong><br>
                    {insight['description']}
                </div>
                """, unsafe_allow_html=True)
        
        # Summary and recommendations
        if insights:
            st.subheader("📋 Summary & Action Items")
            
            action_items = []
            for insight in high_priority:
                if 'engagement' in insight['description'].lower():
                    action_items.append("🎯 Implement user engagement campaigns")
                elif 'recommendation' in insight['description'].lower():
                    action_items.append("🔧 Optimize recommendation algorithm")
                elif 'quality' in insight['description'].lower():
                    action_items.append("📚 Review and curate book collection")
            
            if action_items:
                st.markdown("**Recommended Actions:**")
                for action in set(action_items):  # Remove duplicates
                    st.markdown(f"• {action}")
            else:
                st.success("✅ Your library system is performing well! Continue monitoring these metrics.")
                
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        st.info("💡 This might be due to data format issues. Please check your uploaded files.")

def display_device_analysis(data):
    """Display device-specific analysis with cached visualizations"""
    
    # Breadcrumb navigation  
    st.markdown('<div class="breadcrumb">🏠 Home > 📱 Device Analysis</div>', unsafe_allow_html=True)
    
    st.subheader("📱 Device Usage Analysis")
    
    st.markdown("""
    **Cross-Device Insights**: Understand how users interact with your digital library 
    across different devices and optimize the experience accordingly.
    """)
    
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
    st.subheader("📊 Device Statistics Summary")
    
    try:
        device_stats = data.groupby('device_type').agg({
            'user_id': 'count',
            'session_duration': 'mean',
            'rating': 'mean'
        }).round(2)
        device_stats.columns = ['Total Actions', 'Avg Session (sec)', 'Avg Rating']
        
        # Add insights based on device stats
        st.dataframe(device_stats, use_container_width=True)
        
        # Device insights
        if len(device_stats) > 0:
            best_device = device_stats['Avg Rating'].idxmax()
            longest_session_device = device_stats['Avg Session (sec)'].idxmax()
            most_used_device = device_stats['Total Actions'].idxmax()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="insight-box">
                    <strong>🏆 Highest Rated Device</strong><br>
                    {best_device.title()} users give the highest ratings 
                    ({device_stats.loc[best_device, 'Avg Rating']:.1f}/5.0)
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_session = device_stats.loc[longest_session_device, 'Avg Session (sec)'] / 60
                st.markdown(f"""
                <div class="insight-box">
                    <strong>⏱️ Longest Reading Sessions</strong><br>
                    {longest_session_device.title()} users spend most time reading 
                    ({avg_session:.1f} minutes average)
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                total_actions = device_stats.loc[most_used_device, 'Total Actions']
                st.markdown(f"""
                <div class="insight-box">
                    <strong>📱 Most Popular Device</strong><br>
                    {most_used_device.title()} is the preferred choice 
                    ({total_actions:,} total actions)
                </div>
                """, unsafe_allow_html=True)
                
    except Exception as e:
        st.warning("Unable to generate device statistics. This might be due to missing device data.")
        st.info("💡 Ensure your dataset includes 'device_type' and other relevant columns.")

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