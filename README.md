# 📚 Digital Library Analytics Dashboard

A comprehensive Streamlit application for analyzing borrowing patterns and generating book recommendation insights from digital library data.

## 🌐 Live Demo

**🚀 Try it now:** [https://digital-library-analytics-htp3yvnrcxs7d33my8yqqw.streamlit.app/](https://digital-library-analytics-htp3yvnrcxs7d33my8yqqw.streamlit.app/)

Simply upload your CSV files and start exploring your library data insights immediately!

## 🎯 Features

- **📊 Interactive Dashboard**: Real-time visualizations of borrowing trends, user behavior, and book popularity
- **🔗 Association Rule Mining**: Discover book recommendation relationships using Apriori/FP-Growth algorithms
- **💡 AI-Powered Insights**: Automated analysis with actionable recommendations for library improvement
- **📱 Cross-Device Analysis**: Understand user behavior across desktop, mobile, and tablet devices
- **⭐ Quality Analytics**: Analyze user satisfaction and book rating patterns
- **📈 Temporal Patterns**: Identify peak usage times, seasonal trends, and borrowing patterns
- **🌐 Interactive Network Graphs**: Visual representation of book association networks
- **🔍 Smart Search & Filtering**: Filter all analysis by specific books or topics
- **📤 Export Functionality**: Download processed data and analysis reports
- **⚡ High Performance**: Optimized with Streamlit caching for fast loading

## 🚀 Quick Start

### Option 1: Use the Live Demo
1. Visit [https://digital-library-analytics-htp3yvnrcxs7d33my8yqqw.streamlit.app/](https://digital-library-analytics-htp3yvnrcxs7d33my8yqqw.streamlit.app/)
2. Upload your `digital_library_dataset.csv` and `metadata.csv` files
3. Click "🔄 Load & Process Data"
4. Explore the four analysis tabs: Dashboard, Association Rules, Insights, and Device Analysis

### Option 2: Run Locally
```bash
# Clone the repository
git clone https://github.com/your-username/digital-library-analytics.git
cd digital-library-analytics

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 📊 Data Requirements

### Main Dataset (`digital_library_dataset.csv`)
Your transaction data should include:

| Column | Description | Example |
|--------|-------------|---------|
| `user_id` | Unique borrower identifier | U057, U063, U002 |
| `book_id` | Book identifier (links to metadata) | B001, B025, B023 |
| `borrow_timestamp` | Date and time of borrowing | 2024-01-15 10:30:00 |
| `return_timestamp` | Date and time of return | 2024-01-20 14:22:00 or ######### |
| `rating` | User rating (1-5 scale) | 4, 5, 3 |
| `device_type` | Access device | desktop, tablet, mobile |
| `session_duration` | Time spent in seconds | 3594, 1309, 2785 |
| `action_type` | Type of action | borrow, preview |
| `recommendation_score` | Recommendation indicator | 0, 1 |

### Metadata (`metadata.csv`)
Book information for enriching the analysis:

| Column | Description | Example |
|--------|-------------|---------|
| `book_id` | Book identifier (matches main dataset) | B001, B025, B023 |
| `title` | Human-readable book title | Python Programming, Data Science |
| `author` | Author's name | Sarah Johnson, John Smith |
| `year` | Publication year | 2020, 2019, 2021 |

## 🏗️ Project Structure

```
digital_library_project/
│
├── 📊 MAIN APPLICATION
│   └── app.py                          # Streamlit web application
│
├── 🛠️ UTILITY MODULES
│   ├── utils/
│   │   ├── __init__.py                 # Package initialization
│   │   ├── preprocessing.py            # Data cleaning & merging
│   │   ├── pattern_mining.py           # Association rule mining
│   │   ├── visualization.py            # Charts & graphs generation
│   │   └── insights.py                 # Automated insights generation
│
├── 🔧 CONFIGURATION
│   └── .streamlit/
│       └── config.toml                 # Performance optimizations
│
├── 🔧 HELPER SCRIPTS
│   └── data_validation_helper.py       # Data validation tool
│
├── 📋 PROJECT FILES
│   ├── requirements.txt                # Python dependencies
│   ├── README.md                       # This file
│   └── .gitignore                      # Git ignore rules
```

## 🎛️ Key Features Explained

### 📊 Dashboard
- **Interactive Metrics**: Feature cards that guide users to relevant analysis sections
- **Top Borrowed Books**: Horizontal bar chart showing most popular titles
- **Borrowing Trends**: Time series analysis with moving averages and peak annotations
- **Rating Distribution**: User satisfaction analysis across the collection
- **Device Usage**: Cross-platform usage patterns

### 🔗 Association Rules
- **Automated Rule Mining**: Uses Apriori and FP-Growth algorithms
- **Smart Sorting**: Rules automatically sorted by strength (Lift value)
- **Interactive Parameters**: Adjustable support, confidence, and lift thresholds
- **Visual Network**: Interactive graph showing book relationships
- **Top Rules Highlight**: Featured display of strongest associations

### 💡 Automated Insights
- **Priority-Based Analysis**: High, Medium, and Low priority insights
- **Multi-Category Coverage**: User behavior, content quality, temporal patterns
- **Actionable Recommendations**: Specific steps for library improvement
- **Fallback Analysis**: Basic insights when auto-generation isn't possible

### 📱 Device Analysis
- **Cross-Device Behavior**: Session duration and rating patterns by device
- **Usage Statistics**: Comprehensive device performance comparison
- **User Preferences**: Identify optimal devices for different user types

## 🔍 Search & Filtering

The integrated search functionality allows you to:
- **Filter by Book Title**: Enter keywords to focus analysis on specific books
- **Live Preview**: See matching books before applying filters
- **Global Filtering**: Search affects all charts and analysis sections
- **Easy Reset**: Clear search with one click

## ⚡ Performance Features

- **Streamlit Caching**: Lightning-fast loading after initial data processing
- **Optimized Algorithms**: Efficient association rule mining for large datasets
- **Smart Data Processing**: Automatic handling of missing values and data quality issues
- **Responsive Design**: Works seamlessly across desktop and mobile devices

## 📈 Sample Insights Generated

The application automatically identifies patterns such as:

- **📚 Popular Book Combinations**: "Users who borrow 'Python Programming' also borrow 'Data Science' with 85% confidence"
- **📱 Device Preferences**: "Tablet users spend 2x more time reading than mobile users"
- **⏰ Peak Usage Times**: "Library usage peaks at 2:00 PM on weekdays"
- **⭐ Quality Metrics**: "Recommended books have 15% higher ratings than non-recommended books"
- **👥 User Segments**: "10% of users account for 60% of all library activity"

## 🛠️ Installation & Setup

### Dependencies
```bash
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
mlxtend>=0.22.0
matplotlib>=3.7.0
plotly>=5.15.0
networkx>=3.1.0
pyvis>=0.3.2
seaborn>=0.12.0
scikit-learn>=1.3.0
```

### Local Development
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/digital-library-analytics.git
   cd digital-library-analytics
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Validate your data (optional)**
   ```bash
   python data_validation_helper.py
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📊 Usage Examples

### Basic Analysis Workflow
1. **Upload Data**: Use the sidebar to upload your CSV files
2. **Process**: Click "🔄 Load & Process Data" 
3. **Explore Dashboard**: View key metrics and trends
4. **Discover Patterns**: Check Association Rules for book relationships
5. **Get Insights**: Review automated insights and recommendations
6. **Analyze Devices**: Understand cross-platform user behavior
7. **Export Results**: Download processed data and reports

### Advanced Features
- **Parameter Tuning**: Adjust support, confidence, and lift thresholds for different rule sensitivity
- **Targeted Analysis**: Use search to focus on specific books or authors
- **Temporal Analysis**: Identify seasonal patterns and peak usage times
- **Quality Assessment**: Analyze rating patterns and user satisfaction metrics

## 🔧 Configuration Options

### Analysis Parameters
- **Minimum Support**: Frequency threshold for item combinations (0.01-0.5)
- **Minimum Confidence**: Reliability threshold for association rules (0.1-1.0)  
- **Minimum Lift**: Strength threshold for meaningful associations (1.0-5.0)

### Performance Settings
The application includes optimized settings in `.streamlit/config.toml`:
- WebSocket compression for faster data transfer
- Increased message size limits for large datasets
- Optimized file watching and caching

## 🚨 Troubleshooting

### Common Issues

**No Association Rules Found**
- Lower the support threshold (try 0.01-0.03)
- Reduce confidence requirement (try 0.3-0.5)
- Ensure you have sufficient borrowing data

**Slow Performance**
- The first run processes and caches data (may take 30-60 seconds)
- Subsequent interactions are nearly instantaneous
- Consider reducing dataset size if memory issues occur

**Empty Insights Tab**
- Check data quality and completeness
- Ensure required columns are present
- Fallback basic insights will display if auto-generation fails

**Upload Errors**
- Verify CSV file format and required columns
- Check for special characters in file names
- Ensure files are not corrupted or empty

### Data Quality Tips
- Remove or handle missing values in key columns
- Ensure timestamp formats are consistent
- Verify book_id values match between datasets
- Use the validation helper script to check data quality

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Areas for Contribution
- Additional visualization types
- New insight algorithms
- Performance optimizations
- UI/UX improvements
- Documentation enhancements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** for the amazing web app framework
- **MLxtend** for efficient association rule mining algorithms
- **Plotly** for interactive and beautiful visualizations
- **NetworkX** for graph analysis and network visualizations
- **Pandas & NumPy** for powerful data processing capabilities

## 📞 Support & Feedback

- **🌐 Try the Live Demo**: [https://digital-library-analytics-htp3yvnrcxs7d33my8yqqw.streamlit.app/](https://digital-library-analytics-htp3yvnrcxs7d33my8yqqw.streamlit.app/)
- **🐛 Report Issues**: Create an issue on GitHub
- **💡 Suggest Features**: Open a feature request
- **❓ Get Help**: Check the troubleshooting section or create a discussion

## 🔮 Future Enhancements

- **Real-time Data Integration**: Connect to library management systems
- **Advanced ML Models**: Implement collaborative filtering and content-based recommendations
- **Multi-Library Support**: Compare performance across multiple library branches
- **API Integration**: RESTful API for programmatic access
- **Enhanced Visualizations**: 3D network graphs and advanced statistical plots
- **Mobile App**: Native mobile application for on-the-go analysis

---

**🎯 Ready to discover hidden patterns in your library data?**

**[🚀 Launch the Application](https://digital-library-analytics-htp3yvnrcxs7d33my8yqqw.streamlit.app/)** and start exploring your digital library insights today!

---

*Developed with ❤️ using Python & Streamlit | Digital Library Pattern Analysis & Recommendation System v2.0*