import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handles data loading, cleaning, and preprocessing for the digital library analysis"""
    
    def __init__(self):
        self.merged_data = None
        
    def load_data(self, library_file, metadata_file):
        """Load the library dataset and metadata files"""
        try:
            library_df = pd.read_csv(library_file)
            metadata_df = pd.read_csv(metadata_file)
            return library_df, metadata_df
        except Exception as e:
            raise Exception(f"Error loading data files: {str(e)}")
    
    def clean_library_data(self, df):
        """Clean and preprocess the library dataset"""
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = df_clean.dropna(subset=['user_id', 'book_id', 'action_type'])
        
        # Convert timestamps - handle hashtag placeholder
        timestamp_cols = ['borrow_timestamp', 'return_timestamp']
        for col in timestamp_cols:
            if col in df_clean.columns:
                # Replace hashtag placeholders with NaN
                df_clean[col] = df_clean[col].replace('#########', np.nan)
                # Convert to datetime
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        
        # Clean numeric columns
        if 'rating' in df_clean.columns:
            df_clean['rating'] = pd.to_numeric(df_clean['rating'], errors='coerce')
            # Filter ratings to valid range (1-5)
            df_clean = df_clean[(df_clean['rating'].isna()) | 
                               ((df_clean['rating'] >= 1) & (df_clean['rating'] <= 5))]
        
        if 'session_duration' in df_clean.columns:
            df_clean['session_duration'] = pd.to_numeric(df_clean['session_duration'], errors='coerce')
            # Remove negative durations
            df_clean = df_clean[(df_clean['session_duration'].isna()) | 
                               (df_clean['session_duration'] >= 0)]
        
        if 'recommendation_score' in df_clean.columns:
            df_clean['recommendation_score'] = pd.to_numeric(df_clean['recommendation_score'], errors='coerce')
        
        # Clean categorical columns
        categorical_cols = ['device_type', 'action_type']
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
        
        return df_clean
    
    def clean_metadata(self, df):
        """Clean and preprocess the metadata"""
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = df_clean.dropna(subset=['book_id'])
        
        # Clean string columns
        string_cols = ['title', 'author']
        for col in string_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip()
        
        # Clean year column
        if 'year' in df_clean.columns:
            df_clean['year'] = pd.to_numeric(df_clean['year'], errors='coerce')
            # Filter reasonable years
            current_year = datetime.now().year
            df_clean = df_clean[(df_clean['year'].isna()) | 
                               ((df_clean['year'] >= 1900) & (df_clean['year'] <= current_year))]
        
        return df_clean
    
    def merge_data(self, library_df, metadata_df):
        """Merge library data with metadata based on book_id"""
        try:
            # Clean both datasets
            library_clean = self.clean_library_data(library_df)
            metadata_clean = self.clean_metadata(metadata_df)
            
            print(f"Library dataset: {len(library_clean)} records with {library_clean['book_id'].nunique()} unique books")
            print(f"Metadata: {len(metadata_clean)} records")
            
            # Merge on book_id (left join to keep all library records)
            merged = library_clean.merge(
                metadata_clean,
                on='book_id',
                how='left'
            )
            
            # Check merge success
            books_with_metadata = merged['title'].notna().sum()
            total_records = len(merged)
            merge_success_rate = (books_with_metadata / total_records) * 100
            
            print(f"Merge completed: {books_with_metadata}/{total_records} records ({merge_success_rate:.1f}%) have book metadata")
            
            # Identify books without metadata
            missing_metadata = merged[merged['title'].isna()]['book_id'].unique()
            if len(missing_metadata) > 0:
                print(f"Warning: {len(missing_metadata)} books in library dataset not found in metadata:")
                print(f"Missing book IDs: {list(missing_metadata)[:10]}...")  # Show first 10
            
            # Add derived columns
            merged = self.add_derived_features(merged)
            
            self.merged_data = merged
            return merged
            
        except Exception as e:
            raise Exception(f"Error merging data: {str(e)}")
    
    def add_derived_features(self, df):
        """Add derived features to the merged dataset"""
        df_enhanced = df.copy()
        
        # Add date-based features if borrow_timestamp exists
        if 'borrow_timestamp' in df_enhanced.columns:
            df_enhanced['borrow_date'] = df_enhanced['borrow_timestamp'].dt.date
            df_enhanced['borrow_hour'] = df_enhanced['borrow_timestamp'].dt.hour
            df_enhanced['borrow_day_of_week'] = df_enhanced['borrow_timestamp'].dt.day_name()
            df_enhanced['borrow_month'] = df_enhanced['borrow_timestamp'].dt.month
            df_enhanced['borrow_year'] = df_enhanced['borrow_timestamp'].dt.year
        
        # Calculate reading duration if both timestamps exist
        if all(col in df_enhanced.columns for col in ['borrow_timestamp', 'return_timestamp']):
            df_enhanced['reading_duration'] = (
                df_enhanced['return_timestamp'] - df_enhanced['borrow_timestamp']
            ).dt.total_seconds() / 3600  # Convert to hours
            # Remove negative durations
            df_enhanced.loc[df_enhanced['reading_duration'] < 0, 'reading_duration'] = np.nan
        
        # Add recommendation flag
        if 'recommendation_score' in df_enhanced.columns:
            df_enhanced['is_recommended'] = (df_enhanced['recommendation_score'] > 0).astype(int)
        
        # Add rating categories
        if 'rating' in df_enhanced.columns:
            df_enhanced['rating_category'] = pd.cut(
                df_enhanced['rating'],
                bins=[0, 2, 3, 4, 5],
                labels=['Poor', 'Fair', 'Good', 'Excellent'],
                include_lowest=True
            )
        
        # Add session duration categories
        if 'session_duration' in df_enhanced.columns:
            df_enhanced['session_category'] = pd.cut(
                df_enhanced['session_duration'],
                bins=[0, 300, 900, 1800, float('inf')],
                labels=['Short', 'Medium', 'Long', 'Extended'],
                include_lowest=True
            )
        
        return df_enhanced
    
    def prepare_transaction_data(self, df, action_filter='borrow'):
        """Prepare transaction data for market basket analysis"""
        # Filter by action type
        transactions_df = df[df['action_type'] == action_filter].copy()
        
        if len(transactions_df) == 0:
            return None
        
        # Group by user_id to create transaction lists
        transactions = transactions_df.groupby('user_id')['title'].apply(list).tolist()
        
        # Remove empty transactions
        transactions = [t for t in transactions if len(t) > 0]
        
        return transactions
    
    def get_data_summary(self, df):
        """Generate a summary of the dataset"""
        summary = {
            'total_records': len(df),
            'unique_users': df['user_id'].nunique(),
            'unique_books': df['book_id'].nunique(),
            'date_range': {
                'start': df['borrow_timestamp'].min() if 'borrow_timestamp' in df.columns else None,
                'end': df['borrow_timestamp'].max() if 'borrow_timestamp' in df.columns else None
            },
            'action_types': df['action_type'].value_counts().to_dict() if 'action_type' in df.columns else {},
            'device_types': df['device_type'].value_counts().to_dict() if 'device_type' in df.columns else {},
            'avg_rating': df['rating'].mean() if 'rating' in df.columns else None,
            'avg_session_duration': df['session_duration'].mean() if 'session_duration' in df.columns else None
        }
        return summary
    
    def validate_data(self, df):
        """Validate the merged dataset"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required columns
        required_columns = ['user_id', 'book_id', 'action_type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check for empty dataset
        if len(df) == 0:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Dataset is empty")
        
        # Check for duplicate records
        if df.duplicated().sum() > 0:
            validation_results['warnings'].append(f"Found {df.duplicated().sum()} duplicate records")
        
        # Check data types and ranges
        if 'rating' in df.columns:
            invalid_ratings = df[(df['rating'] < 1) | (df['rating'] > 5)]['rating'].count()
            if invalid_ratings > 0:
                validation_results['warnings'].append(f"Found {invalid_ratings} ratings outside 1-5 range")
        
        if 'session_duration' in df.columns:
            negative_durations = df[df['session_duration'] < 0]['session_duration'].count()
            if negative_durations > 0:
                validation_results['warnings'].append(f"Found {negative_durations} negative session durations")
        
        return validation_results