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
import textwrap  # Added for CSS dedenting
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

def get_native_css():
    """Generate CSS that relies on Streamlit's native theme variables (adapts to light/dark automatically)"""
    # Build CSS as a list of small strings using Streamlit's native vars
    css_parts = [
        textwrap.dedent("""
        /* Base layout - Minimal overrides for custom elements, using native vars */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 100%;
            transition: all 0.3s ease;
            background-color: var(--background-color) !important;
            color: var(--text-color) !important;
            font-size: 1rem; /* Base font size for body text */
            line-height: 1.5;
        }
        
        /* Landing page container - Center and space out welcome content */
        .landing-container {
            max-width: 800px;
            margin: 2rem auto;
            padding: 2rem;
            text-align: center;
            background: var(--secondary-background-color) !important;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
            color: var(--text-color) !important;
        }
        
        /* General text sizing for consistency */
        h1 {
            font-size: 2.5rem !important;
            line-height: 1.2;
            margin-bottom: 1rem;
        }
        h2 {
            font-size: 1.8rem !important;
            line-height: 1.3;
            margin-bottom: 0.8rem;
        }
        h3 {
            font-size: 1.4rem !important;
            line-height: 1.3;
            margin-bottom: 0.8rem;
        }
        h4 {
            font-size: 1.2rem !important;
            line-height: 1.3;
            margin-bottom: 0.5rem;
        }
        p {
            font-size: 1rem !important;
            line-height: 1.6;
            margin-bottom: 0.8rem;
        }
        .stMarkdown p {
            font-size: 1rem !important;
            line-height: 1.6;
        }
        .stText p {
            font-size: 1rem !important;
            line-height: 1.6;
        }
        
        /* Columns - Ensure equal height */
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
        """).strip(),
        
        textwrap.dedent("""
        /* Feature cards - Use native vars for theme adaptation */
.feature-card {
    border: 2px solid var(--border-color) !important;
    border-radius: 12px;
    padding: 1rem; /* Reduced padding for more consistent internal space */
    margin: 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    transition: all 0.3s ease;
    height: 160px; /* Fixed card height */
    min-height: 160px;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between; /* Changed to space-between for fixed top/bottom anchoring */
    align-items: center !important;
    text-align: center;
    background: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
    box-sizing: border-box; /* Ensures padding doesn't add to height */
}

.feature-card h4 {
    margin: 0.25rem 0 0.5rem 0; /* Fixed top/bottom margins - anchors at top */
    font-size: 1.1rem !important; /* Fixed size */
    font-weight: 600;
    line-height: 1.2;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0 0.5rem;
    color: var(--primary-color) !important;
    width: 100%; /* Full width to prevent shifting */
    white-space: nowrap; /* Prevent wrapping */
    overflow: hidden; /* Hide overflow if too long */
    text-overflow: ellipsis; /* Ellipsis for very long titles */
}

.feature-card h3 {
    margin: 0; /* No margin - fixed positioning */
    font-size: 32px !important; /* Fixed size (as per your previous request) - non-responsive */
    font-weight: bold;
    line-height: 1.1; /* Tighter line height for consistency */
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-color) !important;
    width: 100%;
    flex-grow: 0; /* Prevent expansion */
    min-height: 40px; /* Fixed space for the number */
}

.feature-card p {
    margin: 0.5rem 0 0 0; /* Fixed bottom margin - anchors at bottom */
    font-size: 0.95rem !important; /* Fixed size */
    line-height: 1.3; /* Consistent line height */
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 0 0.5rem;
    color: var(--secondary-text-color) !important;
    width: 100%;
    white-space: nowrap; /* Prevent wrapping - keeps text on one line */
    overflow: hidden;
    text-overflow: ellipsis; /* Ellipsis if text is too long in narrow columns */
    flex-grow: 0; /* Prevent expansion */
}
        """).strip(),
        
        textwrap.dedent("""
        /* Insight boxes - Use native vars and gradients */
        .insight-box {
            border-left: 5px solid var(--primary-color) !important;
            padding: 1.2rem;
            margin: 1rem 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
            transition: all 0.3s ease;
            background: linear-gradient(135deg, var(--secondary-background-color) 0%, var(--background-color) 50%, var(--secondary-background-color) 100%) !important;
            color: var(--text-color) !important;
            font-size: 1rem; /* Consistent sizing */
            line-height: 1.5;
        }
        .insight-box strong {
            font-size: 1.1rem;
            color: var(--primary-color) !important;
        }
        
        /* Breadcrumb - Use native vars */
        .breadcrumb {
            padding: 0.5rem 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
            font-size: 0.9rem;
            transition: all 0.3s ease;
            background: var(--secondary-background-color) !important;
            color: var(--secondary-text-color) !important;
            border: 1px solid var(--border-color) !important;
        }
        """).strip(),
        
        textwrap.dedent("""
        /* Dataframe overrides - Use native vars */
        .dataframe {
            font-size: 0.95rem; /* Slightly larger for better readability */
            transition: all 0.3s ease;
            background: var(--background-color) !important;
            color: var(--text-color) !important;
        }
        .dataframe th {
            background: var(--secondary-background-color) !important;
            color: var(--text-color) !important;
            border-color: var(--border-color) !important;
            font-size: 0.95rem;
            padding: 0.75rem;
        }
        .dataframe td {
            border-color: var(--border-color) !important;
            color: var(--text-color) !important;
            padding: 0.75rem;
            font-size: 0.95rem;
        }
        [data-testid="stDataFrame"] div[role="grid"] {
            background: var(--background-color) !important;
            color: var(--text-color) !important;
        }
        
        /* Network container - Use native vars */
        .network-container {
            border: 1px solid var(--border-color) !important;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            transition: all 0.3s ease;
            background: var(--background-color) !important;
            color: var(--text-color) !important;
        }
        
        /* Footer - Fixed at bottom, outside content */
        .footer {
            text-align: center;
            padding: 2rem 0;
            border-top: 1px solid var(--border-color) !important;
            margin-top: 3rem;
            border-radius: 8px;
            transition: all 0.3s ease;
            color: var(--secondary-text-color) !important;
            background: var(--secondary-background-color) !important;
            font-size: 0.95rem;
            position: relative; /* Ensures it stays at bottom of page, not tabs */
            width: 100%;
        }
        """).strip(),
        
        textwrap.dedent("""
        /* Sidebar - Minimal sync with native */
        section[data-testid="stSidebar"] {
            transition: all 0.3s ease;
            background-color: var(--sidebar-background-color) !important;
            color: var(--text-color) !important;
            border-right: 1px solid var(--border-color) !important;
            font-size: 1rem;
        }
        section[data-testid="stSidebar"] > div > div {
            color: var(--text-color) !important;
        }
        
        /* Tabs - Enhanced with native colors */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            border-radius: 8px 8px 0 0;
            padding: 0 1rem;
            transition: all 0.3s ease;
            background-color: var(--secondary-background-color) !important;
            color: var(--secondary-text-color) !important;
            border-color: var(--border-color) !important;
            font-size: 1rem;
        }
        .stTabs [aria-selected="true"] {
            background-color: var(--primary-color) !important;
            color: var(--primary-text-color) !important;
            font-weight: bold;
        }
        .stTabs [data-baseweb="tab-panel"] {
            background: var(--background-color) !important;
            color: var(--text-color) !important;
            padding: 1rem;
            border-radius: 0 0 8px 8px;
            font-size: 1rem;
        }
        
        /* Main header - Use native vars */
        .main-header {
            font-size: 2.5rem; /* Consistent with h1 */
            font-weight: bold;
            color: var(--primary-color) !important;
            text-align: center;
            margin-bottom: 1.5rem;
            padding: 1rem 0;
            background: linear-gradient(90deg, var(--secondary-background-color) 0%, var(--background-color) 100%) !important;
            border-radius: 10px;
        }
        
        /* Expander - Use native vars */
        [data-testid="stExpander"] {
            background: var(--secondary-background-color) !important;
            color: var(--text-color) !important;
            border-color: var(--border-color) !important;
            font-size: 1rem;
        }
        [data-testid="stExpander"] h4 {
            font-size: 1.2rem;
        }
        [data-testid="stMetric"] {
            background: var(--secondary-background-color) !important;
            color: var(--text-color) !important;
            font-size: 1rem;
        }
        [data-testid="stMetricLabel"] {
            color: var(--secondary-text-color) !important;
            font-size: 0.95rem;
        }
        [data-testid="stMetricValue"] {
            color: var(--text-color) !important;
            font-size: 1.5rem;
        }
        
        /* Plotly - Minimal overrides using native vars */
        .plotly {
            background: var(--background-color) !important;
            color: var(--text-color) !important;
        }
        .modebar {
            background: var(--secondary-background-color) !important;
            color: var(--text-color) !important;
        }
        .plotly .plotlyjs path, .plotly .plotlyjs line {
            stroke: var(--text-color) !important;
        }
        .plotly .plotlyjs text {
            fill: var(--text-color) !important;
            font-size: 12px; /* Consistent plot text size */
        }
        
        /* General text - Use native vars */
        .stMarkdown {
            color: var(--text-color) !important;
            font-size: 1rem;
        }
        .stText {
            color: var(--text-color) !important;
            font-size: 1rem;
        }
        """).strip()
    ]
    
    # Join all parts into final CSS
    shared_css = '\n\n'.join(css_parts)
    
    return f"<style>{shared_css}</style>"

def main():
    # Initialize session state (removed theme-related state)
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'merged_data' not in st.session_state:
        st.session_state.merged_data = None
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = DataPreprocessor()
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Dashboard"
    
    # Inject native-adapting CSS early (no theme detection needed)
    st.markdown(get_native_css(), unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">📚 Digital Library Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar - Always present, but content varies
    st.sidebar.markdown("## 📂 Data Upload")
    
    # File upload (only show button if files are selected)
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
    
    if library_file and metadata_file and not st.session_state.data_loaded:
        if st.sidebar.button("🔄 Load & Process Data", type="primary"):  # Make it prominent
            with st.spinner("Processing data..."):
                try:
                    # Convert uploaded files to string content for caching
                    library_content = str(library_file.read(), "utf-8")
                    metadata_content = str(metadata_file.read(), "utf-8")
                    
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
                    
                    st.rerun()  # Rerun to update the UI and show tabs
                    
                except Exception as e:
                    st.sidebar.error(f"❌ Error loading data: {str(e)}")
                    st.sidebar.info("💡 Tip: Make sure both files have 'book_id' column for merging")
    
    if st.session_state.data_loaded:
        # Sidebar filters (shown only after data load)
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
        
        # Main content with tabs (only after data loaded)
        display_data = st.session_state.merged_data.copy()
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🔍 Book Search", "🔗 Association Rules", "💡 Insights", "📱 Device Analysis"])
        
        with tab1:
            display_dashboard_content(display_data, min_support, min_confidence, min_lift)
        
        with tab2:
            display_book_search(display_data, min_support, min_confidence, min_lift)
        
        with tab3:
            display_association_rules(display_data, min_support, min_confidence, min_lift)
        
        with tab4:
            display_insights(display_data)
        
        with tab5:
            display_device_analysis(display_data)
        
        # Export functionality (only after data loaded)
        st.sidebar.markdown("## 📤 Export Results")
        if st.sidebar.button("Download Analysis Results"):
            export_results(st.session_state.merged_data)
    
    else:
        # Landing Page: Welcome message (only shown pre-load, centered and isolated)
        st.markdown('<div class="landing-container">', unsafe_allow_html=True)
        st.markdown("""
        <h2>Welcome to Digital Library Analytics! 📖</h2>
        
        <p>This application helps you analyze borrowing patterns and generate book recommendations from your digital library data.</p>
        
        <h3>To get started:</h3>
        <ol>
            <li>Upload your <code>digital_library_dataset.csv</code> file in the sidebar</li>
            <li>Upload your <code>metadata.csv</code> file in the sidebar</li>
            <li>Click "Load & Process Data" to begin analysis</li>
        </ol>
        
        <h3>Features:</h3>
        <ul>
            <li><strong>📊 Dashboard</strong>: Visualize borrowing trends and patterns</li>
            <li><strong>🔍 Book Search</strong>: Explore individual books, their borrows, and analytics</li>
            <li><strong>🔗 Association Rules</strong>: Discover book recommendation relationships</li>
            <li><strong>💡 Insights</strong>: Get automatic insights from your data</li>
            <li><strong>📱 Device Analysis</strong>: Understand user behavior across devices</li>
        </ul>
        """, unsafe_allow_html=True)
        
        # Sample data structure expander (collapsible)
        with st.expander("📋 Expected Data Structure", expanded=False):
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
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Optional: Add a centered call-to-action in main area
        st.info("👆 **Ready to upload?** Use the sidebar on the left to select your CSV files and load the data.")
    
    # Footer - Always at the bottom of the page (outside tabs/landing)
    st.markdown("""
    <div class="footer">
        <p>Developed with Python & Streamlit | For Educational Purposes</p>
        <p>📚 Digital Library Pattern Analysis & Recommendation System</p>
    </div>
    """, unsafe_allow_html=True)

def display_dashboard_content(data, min_support, min_confidence, min_lift):
    """Display the dashboard content with properly aligned metric cards"""
    
    # Breadcrumb navigation
    st.markdown('<div class="breadcrumb">🏠 Home > 📊 Dashboard</div>', unsafe_allow_html=True)
    
    # Create perfectly aligned metric cards
    col1, col2, col3, col4 = st.columns(4)

    # Calculate metrics (full data now)
    total_borrows = len(data[data['action_type'] == 'borrow'])
    unique_users = data['user_id'].nunique()
    unique_books = data['book_id'].nunique()
    avg_rating = data['rating'].mean()

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
    display_dashboard_charts(data)

def display_dashboard_charts(data):
    """Display dashboard charts using cached visualizations with native theme adaptation"""
    
    # Detect current Streamlit theme
    current_theme = st.get_option('theme.base')
    
    # Use cached visualizations
    viz_data = create_visualizations_cached(data)
    
    # Plotly config
    plotly_config = {
        'displayModeBar': False,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Top Borrowed Books")
        fig_top = viz_data['top_books']
        if current_theme == 'dark':
            fig_top.update_layout(template='plotly_dark')
        else:
            fig_top.update_layout(template='plotly_white')
        st.plotly_chart(fig_top, use_container_width=True, config=plotly_config)
    
    with col2:
        st.subheader("📈 Borrowing Trends Over Time")
        fig_trends = viz_data['trends']
        if current_theme == 'dark':
            fig_trends.update_layout(template='plotly_dark')
        else:
            fig_trends.update_layout(template='plotly_white')
        st.plotly_chart(fig_trends, use_container_width=True, config=plotly_config)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("⭐ Rating Distribution")
        fig_ratings = viz_data['ratings']
        if current_theme == 'dark':
            fig_ratings.update_layout(template='plotly_dark')
        else:
            fig_ratings.update_layout(template='plotly_white')
        st.plotly_chart(fig_ratings, use_container_width=True, config=plotly_config)
    
    with col4:
        st.subheader("📱 Device Usage")
        fig_devices = viz_data['devices']
        if current_theme == 'dark':
            fig_devices.update_layout(template='plotly_dark')
        else:
            fig_devices.update_layout(template='plotly_white')
        st.plotly_chart(fig_devices, use_container_width=True, config=plotly_config)

def display_book_search(data, min_support, min_confidence, min_lift):
    """Display the book search tab with list, selection, transactions, analytics, and consequents (recommendations)"""
    
    # Breadcrumb navigation
    st.markdown('<div class="breadcrumb">🏠 Home > 🔍 Book Search</div>', unsafe_allow_html=True)
    
    st.subheader("🔍 Individual Book Search & Analytics")
    
    st.markdown("""
    **Explore Specific Books**: Select a book to view its borrowing transactions, detailed analytics, and recommended books (consequents).
    """)
    
    # Prepare unique books list (with borrow counts for sorting)
    if 'title' in data.columns and 'action_type' in data.columns:
        borrow_data = data[data['action_type'] == 'borrow']
        if not borrow_data.empty:
            book_stats = borrow_data.groupby('title').agg({
                'user_id': 'nunique',  # Unique users
                'action_type': 'count',  # Borrow count
                'rating': 'mean'  # Avg rating
            }).round(2)
            book_stats.columns = ['Unique Users', 'Borrow Count', 'Avg Rating']
            if 'author' in data.columns:
                # Add author (take first non-null author for each title)
                authors = data.groupby('title')['author'].first()
                book_stats = book_stats.join(authors)
                book_stats = book_stats[['author', 'Borrow Count', 'Unique Users', 'Avg Rating']]  # Reorder
            else:
                book_stats = book_stats[['Borrow Count', 'Unique Users', 'Avg Rating']]
            
            # Sort by borrow count descending
            book_stats = book_stats.sort_values('Borrow Count', ascending=False)
        else:
            book_stats = pd.DataFrame()
    else:
        book_stats = pd.DataFrame()
    
    if not book_stats.empty:
        # Display list of all books
        st.subheader("📚 All Books in Library")
        st.markdown("*Sorted by popularity (borrow count descending)*")
        st.dataframe(
            book_stats,
            use_container_width=True,
            height=300  # Scrollable height
        )
        
        # Search and selection
        st.subheader("🔎 Select a Book")
        
        # Fuzzy search text input (filters selectbox options)
        search_input = st.text_input(
            "Search books by title (type to filter)",
            placeholder="e.g., 'Python', 'Data Science'",
            help="Type part of a title to narrow down the list below"
        )
        
        # Get filtered book titles
        all_titles = book_stats.index.tolist()
        if search_input:
            filtered_titles = [title for title in all_titles if search_input.lower() in title.lower()]
        else:
            filtered_titles = all_titles
        
        if filtered_titles:
            selected_book = st.selectbox(
                "Choose a book:",
                options=filtered_titles,
                index=0 if filtered_titles else None,
                help="Select to view transactions and analytics for this book"
            )
        else:
            selected_book = None
            st.warning("No books match your search. Try a different term.")
            return
        
        if selected_book:
            # Filter data for selected book (only borrow actions)
            book_data = data[data['title'] == selected_book]
            borrow_transactions = book_data[book_data['action_type'] == 'borrow']
            
            if len(borrow_transactions) == 0:
                st.warning(f"No borrow transactions found for '{selected_book}'.")
                return
            
            # Borrowed Transactions Table
            st.subheader(f"📋 Borrow Transactions for '{selected_book}'")
            st.markdown(f"*Total: {len(borrow_transactions)} borrows*")
            
            # Select relevant columns for display (limit to top 100 for performance)
            display_cols = ['user_id', 'borrow_timestamp', 'return_timestamp', 'rating', 'device_type', 'session_duration']
            if all(col in borrow_transactions.columns for col in display_cols):
                display_transactions = borrow_transactions[display_cols].head(100).copy()
                # Format timestamps if they are strings
                for col in ['borrow_timestamp', 'return_timestamp']:
                    if col in display_transactions.columns:
                        display_transactions[col] = pd.to_datetime(display_transactions[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(
                    display_transactions,
                    use_container_width=True,
                    height=300
                )
                if len(borrow_transactions) > 100:
                    st.info(f"Showing top 100 transactions. Total borrows: {len(borrow_transactions)}")
            else:
                st.dataframe(borrow_transactions.head(100), use_container_width=True, height=300)
            
            # Analytics
            st.subheader(f"📊 Analytics for '{selected_book}'")
            
            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            total_borrows = len(borrow_transactions)
            unique_users = borrow_transactions['user_id'].nunique()
            avg_rating = borrow_transactions['rating'].mean() if 'rating' in borrow_transactions.columns else 0
            avg_session = borrow_transactions['session_duration'].mean() if 'session_duration' in borrow_transactions.columns else 0
            
            with col1:
                st.metric("📈 Total Borrows", total_borrows)
            with col2:
                st.metric("👥 Unique Users", unique_users)
            with col3:
                st.metric("⭐ Avg Rating", f"{avg_rating:.1f}/5.0")
            with col4:
                st.metric("⏱️ Avg Session Duration", f"{avg_session:.0f} seconds")
            
            # Detect current theme for Plotly
            current_theme = st.get_option('theme.base')
            
            # Device Distribution Pie Chart
            if 'device_type' in borrow_transactions.columns and len(borrow_transactions['device_type'].unique()) > 1:
                st.subheader("📱 Device Distribution")
                device_pie = px.pie(
                    borrow_transactions,
                    names='device_type',
                    title="Borrows by Device",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                if current_theme == 'dark':
                    device_pie.update_layout(template='plotly_dark')
                else:
                    device_pie.update_layout(template='plotly_white')
                st.plotly_chart(device_pie, use_container_width=True, config={'displayModeBar': False})
            
            # Top Users Bar Chart
            if 'user_id' in borrow_transactions.columns:
                st.subheader("👤 Top 5 Users (by Borrow Count)")
                top_users = borrow_transactions.groupby('user_id').size().reset_index(name='Borrows').head(5)
                top_users = top_users.sort_values('Borrows', ascending=True)  # For horizontal bar
                user_bar = px.bar(
                    top_users,
                    x='Borrows',
                    y='user_id',
                    orientation='h',
                    title="Top Users Who Borrowed This Book",
                    color='Borrows',
                    color_continuous_scale='Viridis'
                )
                if current_theme == 'dark':
                    user_bar.update_layout(template='plotly_dark')
                else:
                    user_bar.update_layout(template='plotly_white')
                st.plotly_chart(user_bar, use_container_width=True, config={'displayModeBar': False})
            
            # NEW: Consequents / Recommended Books Section
            st.subheader(f"🔗 Recommended Books for '{selected_book}' (Consequents)")
            st.markdown("""
            **Books that users who borrowed this one also tend to borrow.** 
            Based on association rules analysis (using current parameters: Support={}, Confidence={}, Lift={}).
            """.format(min_support, min_confidence, min_lift))
            
            try:
                # Generate association rules for the whole dataset
                pattern_miner = PatternMiner()
                all_rules = pattern_miner.generate_association_rules(data, min_support, min_confidence, min_lift)
                
                if len(all_rules) == 0:
                    st.warning("No association rules found with current parameters. Try adjusting in the sidebar.")
                    return
                
                # Filter rules where selected_book is in antecedents
                selected_book_set = frozenset([selected_book])
                relevant_rules = all_rules[
                    all_rules['antecedents'].apply(lambda x: selected_book_set.issubset(x))
                ]
                
                if len(relevant_rules) == 0:
                    st.info(f"No direct associations found for '{selected_book}'. It might be a unique or less frequently borrowed book.")
                    st.markdown("💡 **Tip**: Lower the thresholds in the sidebar to find more connections.")
                else:
                    # Sort by confidence descending
                    relevant_rules = relevant_rules.sort_values('confidence', ascending=False)
                    
                    # Extract consequents and their metrics
                    recommendations = []
                    for _, rule in relevant_rules.iterrows():
                        consequents = list(rule['consequents'])
                        confidence = rule['confidence']
                        lift = rule['lift']
                        
                        # Get borrow count for consequents
                        borrow_counts = data[data['action_type'] == 'borrow'].groupby('title').size()
                        
                        for cons in consequents:
                            recommendations.append({
                                'Book': cons,
                                'Confidence': f"{confidence:.1%}",
                                'Lift': f"{lift:.2f}",
                                'Popularity': borrow_counts.get(cons, 0)
                            })
                    
                    # Create DataFrame for recommendations
                    rec_df = pd.DataFrame(recommendations)
                    if not rec_df.empty:
                        # Fix the sorting - convert percentage strings to floats for proper sorting
                        rec_df['confidence_numeric'] = rec_df['Confidence'].str.rstrip('%').astype(float) / 100
                        rec_df = rec_df.sort_values('confidence_numeric', ascending=False).head(10)
                        rec_df = rec_df.drop('confidence_numeric', axis=1)  # Remove helper column
                        
                        st.dataframe(
                            rec_df,
                            use_container_width=True
                        )
                        
                        # Top 3 recommendations highlight
                        if len(rec_df) > 0:
                            st.markdown("**Top Recommendations:**")
                            for idx, row in rec_df.head(3).iterrows():
                                st.markdown(f"""
                                <div class="insight-box">
                                    <strong>📖 {row['Book']}</strong><br>
                                    Confidence: {row['Confidence']} | Lift: {row['Lift']} | Popularity: {row['Popularity']:,} borrows
                                </div>
                                """, unsafe_allow_html=True)
                
                # Simple co-borrowed books as fallback (if no rules)
                if len(relevant_rules) == 0:
                    st.subheader("📊 Alternative: Commonly Co-Borrowed Books")
                    # Find users who borrowed this book and what else they borrowed
                    users_who_borrowed = borrow_transactions['user_id'].unique()
                    co_borrowed = data[
                        (data['user_id'].isin(users_who_borrowed)) & 
                        (data['action_type'] == 'borrow') & 
                        (data['title'] != selected_book)
                    ]['title'].value_counts().head(5)
                    
                    if len(co_borrowed) > 0:
                        co_df = pd.DataFrame({'Book': co_borrowed.index, 'Co-Borrows': co_borrowed.values})
                        st.dataframe(co_df, use_container_width=True)
                        st.info("Based on direct user borrowing patterns (no association rules applied).")
                    else:
                        st.info("No co-borrowing patterns found.")
                        
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
                st.info("💡 Ensure data has sufficient borrowing transactions for analysis.")
                
    else:
        st.warning("No book data available. Ensure 'title' and 'action_type' columns exist in your dataset.")

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
            
            # Compute borrow counts for all unique titles in the data
            borrow_counts = {}
            if 'title' in data.columns and 'action_type' in data.columns:
                borrow_data = data[data['action_type'] == 'borrow']
                if not borrow_data.empty and 'title' in borrow_data.columns:
                    borrow_counts = borrow_data.groupby('title').size().to_dict()
            
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
            
            # Add borrow counts for antecedents and consequents
            def get_borrow_count(items_str):
                if pd.isna(items_str):
                    return 0
                items = [item.strip() for item in items_str.split(',')]
                return sum(borrow_counts.get(item, 0) for item in items)
            
            display_rules['antecedent_borrows'] = display_rules['antecedents'].apply(get_borrow_count)
            display_rules['consequent_borrows'] = display_rules['consequents'].apply(get_borrow_count)
            
            # Round numerical columns
            numerical_cols = ['support', 'confidence', 'lift', 'antecedent_borrows', 'consequent_borrows']
            for col in numerical_cols:
                if col in display_rules.columns:
                    display_rules[col] = display_rules[col].round(3)
            
            # Add rule strength indicators
            display_rules['strength'] = display_rules['lift'].apply(
                lambda x: "🔥 Very Strong" if x >= 3.0 else "💪 Strong" if x >= 2.0 else "📈 Moderate"
            )
            
            # Display table with better column names (including borrow counts)
            formatted_table = display_rules[[
                'antecedents', 'antecedent_borrows', 'consequents', 'consequent_borrows',
                'support', 'confidence', 'lift', 'strength'
            ]].copy()
            
            formatted_table.columns = [
                'Antecedent Books (If user borrows...)', 'Antecedent Borrows',
                'Consequent Books (Then they also borrow...)', 'Consequent Borrows',
                'Support', 'Confidence', 'Lift', 'Strength'
            ]
            
            st.dataframe(
                formatted_table,
                use_container_width=True,
                hide_index=True
            )
            
            # Top 5 strongest rules highlight (with borrow counts)
            st.subheader("🏆 Top 5 Strongest Relationships")

            # Reset index to ensure proper ordering and create a clean rule number
            top_5 = display_rules.head(5).reset_index(drop=True)

            for idx, rule in top_5.iterrows():
                antecedent_count = rule['antecedent_borrows']
                consequent_count = rule['consequent_borrows']
                st.markdown(f"""
                <div class="insight-box">
                    <strong>Rule #{idx + 1}</strong><br>
                    📚 <strong>Antecedent:</strong> {rule['antecedents']} ({antecedent_count} total borrows)<br>
                    📖 <strong>Consequent:</strong> {rule['consequents']} ({consequent_count} total borrows)<br>
                    🎯 <strong>Confidence:</strong> {rule['confidence']:.1%} | 📈 <strong>Lift:</strong> {rule['lift']:.2f}
                </div>
                """, unsafe_allow_html=True)
            
            # Network visualization with better container
            st.subheader("🌐 Association Rules Network")
            st.markdown("*Visual representation of book relationships - larger nodes = more popular books*")
            
            if len(rules_df) > 0:
                pattern_miner = PatternMiner()
                # Pass theme to visualizer if updated in utils (optional)
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
    """Display automated insights with caching (no theme needed, as text-based)"""
    
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
                # Compute activity level outside f-string
                activity_level = "high" if total_records > 500 else "moderate" if total_records > 100 else "low"
                st.markdown(f"""
                <div class="insight-box">
                    <strong>📈 Library Activity Summary</strong><br>
                    Your library has {total_records:,} total interactions from {unique_users:,} users across {unique_books:,} books.
                    This indicates {activity_level} activity levels.
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if avg_rating:
                    # Compute satisfaction outside f-string
                    if avg_rating >= 4.0:
                        satisfaction_level = "high"
                        satisfaction_msg = "Keep up the excellent content curation!"
                    elif avg_rating < 3.0:
                        satisfaction_level = "low"
                        satisfaction_msg = "Consider reviewing lower-rated books."
                    else:
                        satisfaction_level = "moderate"
                        satisfaction_msg = "Good overall satisfaction levels."
                    
                    st.markdown(f"""
                    <div class="insight-box">
                        <strong>⭐ User Satisfaction</strong><br>
                        Average rating is {avg_rating:.1f}/5.0, indicating {satisfaction_level} user satisfaction.
                        {satisfaction_msg}
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
                <div class="insight-box" style="border-left-color: var(--error-color);">
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
                <div class="insight-box" style="border-left-color: var(--warning-color);">
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
                <div class="insight-box" style="border-left-color: var(--success-color);">
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
    """Display device-specific analysis with cached visualizations and native theme adaptation"""
    
    # Breadcrumb navigation  
    st.markdown('<div class="breadcrumb">🏠 Home > 📱 Device Analysis</div>', unsafe_allow_html=True)
    
    st.subheader("📱 Device Usage Analysis")
    
    st.markdown("""
    **Cross-Device Insights**: Understand how users interact with your digital library 
    across different devices and optimize the experience accordingly.
    """)
    
    # Detect current theme for Plotly
    current_theme = st.get_option('theme.base')
    
    # Use cached visualizations
    viz_data = create_visualizations_cached(data)
    
    # Plotly config
    plotly_config = {
        'displayModeBar': False,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Session Duration by Device")
        fig_session = viz_data['session_duration']
        if current_theme == 'dark':
            fig_session.update_layout(template='plotly_dark')
        else:
            fig_session.update_layout(template='plotly_white')
        st.plotly_chart(fig_session, use_container_width=True, config=plotly_config)
    
    with col2:
        st.subheader("Rating by Device Type")
        fig_ratings = viz_data['device_ratings']
        if current_theme == 'dark':
            fig_ratings.update_layout(template='plotly_dark')
        else:
            fig_ratings.update_layout(template='plotly_white')
        st.plotly_chart(fig_ratings, use_container_width=True, config=plotly_config)
    
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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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