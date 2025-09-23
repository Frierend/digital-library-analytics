# 📚 Digital Library Analytics Dashboard

A comprehensive Streamlit application for analyzing borrowing patterns and generating book recommendation insights from digital library data.

## 🚀 Features

- **📊 Interactive Dashboard**: Visualize borrowing trends, device usage, and user behavior
- **🔗 Association Rule Mining**: Discover book recommendation relationships using Apriori/FP-Growth algorithms
- **💡 Automated Insights**: Generate intelligent insights from your library data
- **📱 Device Analysis**: Understand user behavior across different device types
- **⭐ Rating Analysis**: Analyze user satisfaction and book quality metrics
- **📈 Temporal Patterns**: Identify peak usage times and seasonal trends
- **🌐 Network Visualization**: Interactive network graphs of book associations
- **📤 Export Functionality**: Download analysis results and reports

## 🏗️ Project Structure

```
digital_library_project/
│
├── app.py                          # Main Streamlit application
├── data/                           # Data directory
│   ├── digital_library_dataset.csv # Main dataset
│   └── metadata.csv                # Book metadata
│
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── preprocessing.py            # Data cleaning and preprocessing
│   ├── pattern_mining.py           # Association rule mining
│   ├── visualization.py            # Chart and graph generation
│   └── insights.py                 # Automated insight generation
│
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore file
```

## 📋 Data Requirements

### Main Dataset (`digital_library_dataset.csv`)
Required columns:
- `user_id`: Unique borrower identifier
- `book_id`: Book identifier (links to metadata)
- `borrow_timestamp`: Date and time of borrowing
- `return_timestamp`: Date and time of return
- `rating`: User rating (1-5 scale)
- `device_type`: Access device (desktop, tablet, mobile)
- `session_duration`: Time spent in seconds
- `action_type`: Type of action (borrow, preview, etc.)
- `recommendation_score`: Recommendation indicator

### Metadata (`metadata.csv`)
Required columns:
- `book_id`: Book identifier (matches main dataset)
- `title`: Human-readable book title
- `author`: Author's name
- `year`: Publication year

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/digital-library-analytics.git
cd digital-library-analytics
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Create the utils package:**
```bash
mkdir utils
touch utils/__init__.py
```

5. **Add your data files:**
   - Place `digital_library_dataset.csv` and `metadata.csv` in the `data/` directory
   - Or use the file upload feature in the web interface

## 🚀 Usage

1. **Run the Streamlit app:**
```bash
streamlit run app.py
```

2. **Access the dashboard:**
   - Open your browser to `http://localhost:8501`
   - Upload your CSV files using the sidebar
   - Click "Load & Process Data"

3. **Explore the features:**
   - **Dashboard Tab**: View key metrics and visualizations
   - **Association Rules Tab**: Explore book recommendation patterns
   - **Insights Tab**: Read automated insights and recommendations
   - **Device Analysis Tab**: Understand cross-device user behavior

## 📊 Key Features Explained

### Association Rule Mining
- Uses Apriori and FP-Growth algorithms to find patterns in borrowing behavior
- Generates rules like "Users who borrow Book A also borrow Book B"
- Configurable support, confidence, and lift thresholds
- Interactive network visualization of book relationships

### Automated Insights
- Analyzes user engagement and behavior patterns
- Identifies peak usage times and seasonal trends
- Evaluates recommendation system effectiveness
- Provides actionable recommendations for library improvement

### Visualization Components
- Top borrowed books (bar charts)
- Borrowing trends over time (line charts)
- Device usage distribution (pie charts)
- Rating distributions (histograms)
- Session duration analysis (box plots)
- Network graphs of book associations

## 🎛️ Configuration Options

### Analysis Parameters (Sidebar)
- **Minimum Support**: Threshold for frequent itemsets (0.01-0.5)
- **Minimum Confidence**: Threshold for association rules (0.1-1.0)
- **Minimum Lift**: Threshold for rule strength (1.0-5.0)

### Search and Filter
- Search books by title
- Filter results in real-time
- Export filtered datasets

## 📈 Sample Insights Generated

- Library usage overview and user engagement metrics
- Peak usage times and seasonal patterns
- Device preference analysis
- Book popularity and author rankings
- Rating quality assessment
- Recommendation system effectiveness
- User activity distribution patterns

## 🔧 Technical Details

### Technologies Used
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **MLxtend**: Machine learning extensions for association rules
- **Plotly**: Interactive visualizations
- **NetworkX & Pyvis**: Network graph generation
- **NumPy**: Numerical computing

### Algorithm Implementation
- **Apriori Algorithm**: For frequent itemset mining
- **FP-Growth**: Alternative frequent pattern mining
- **Association Rules**: Confidence, support, and lift calculation
- **Data Preprocessing**: Automatic cleaning and validation

## 📝 Data Preprocessing Features

- Automatic data cleaning and validation
- Timestamp parsing and normalization
- Missing value handling
- Data type conversion and validation
- Derived feature generation (reading duration, seasonal patterns)
- Transaction preparation for market basket analysis

## 🚨 Troubleshooting

### Common Issues
1. **File Upload Errors**: Ensure CSV files match the expected structure
2. **No Association Rules**: Lower the support/confidence thresholds
3. **Memory Issues**: Reduce dataset size or increase system memory
4. **Network Visualization Not Loading**: Check browser popup blockers

### Data Quality Checks
- The app validates required columns automatically
- Provides warnings for data quality issues
- Suggests corrections for common problems

## 📊 Export Options

- Download processed datasets as CSV
- Export association rules with metadata
- Save visualization charts as images
- Generate comprehensive analysis reports

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with Streamlit for rapid web app development
- Uses MLxtend library for efficient association rule mining
- Plotly for interactive and responsive visualizations
- NetworkX for graph analysis and visualization

## 📞 Support

For questions or issues:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting section

---

**Developed with Python & Streamlit | For Educational Purposes**

*Digital Library Pattern Analysis & Recommendation System v1.0*