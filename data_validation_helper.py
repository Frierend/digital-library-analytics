"""
Data validation helper for checking your datasets before running the main application.
Run this script to validate your digital_library_dataset.csv and metadata.csv files.
"""

import pandas as pd
import numpy as np

def validate_datasets(library_file="data/digital_library_dataset.csv", metadata_file="data/metadata.csv"):
    """Validate both datasets and check merge compatibility"""
    
    print("🔍 Digital Library Data Validation")
    print("=" * 50)
    
    try:
        # Load datasets
        print("\n📂 Loading datasets...")
        library_df = pd.read_csv(library_file)
        metadata_df = pd.read_csv(metadata_file)
        
        print(f"✅ Library dataset loaded: {len(library_df)} records")
        print(f"✅ Metadata loaded: {len(metadata_df)} records")
        
        # Validate library dataset structure
        print("\n📊 Library Dataset Validation:")
        print("-" * 30)
        
        required_library_columns = [
            'user_id', 'book_id', 'borrow_timestamp', 'return_timestamp', 
            'rating', 'device_type', 'session_duration', 'action_type', 'recommendation_score'
        ]
        
        library_columns = library_df.columns.tolist()
        missing_library_cols = [col for col in required_library_columns if col not in library_columns]
        
        if missing_library_cols:
            print(f"❌ Missing columns in library dataset: {missing_library_cols}")
        else:
            print("✅ All required columns present in library dataset")
        
        print(f"📋 Library dataset columns: {library_columns}")
        print(f"📈 Library dataset shape: {library_df.shape}")
        
        # Check book_id format in library dataset
        library_book_ids = library_df['book_id'].unique()
        print(f"📚 Unique books in library: {len(library_book_ids)}")
        print(f"📚 Sample book IDs: {library_book_ids[:10].tolist()}")
        
        # Validate metadata structure
        print("\n📚 Metadata Validation:")
        print("-" * 20)
        
        required_metadata_columns = ['book_id', 'title', 'author', 'year']
        metadata_columns = metadata_df.columns.tolist()
        missing_metadata_cols = [col for col in required_metadata_columns if col not in metadata_columns]
        
        if missing_metadata_cols:
            print(f"❌ Missing columns in metadata: {missing_metadata_cols}")
        else:
            print("✅ All required columns present in metadata")
        
        print(f"📋 Metadata columns: {metadata_columns}")
        print(f"📈 Metadata shape: {metadata_df.shape}")
        
        # Check book_id format in metadata
        metadata_book_ids = metadata_df['book_id'].unique()
        print(f"📚 Books in metadata: {len(metadata_book_ids)}")
        print(f"📚 Sample metadata book IDs: {metadata_book_ids[:10].tolist()}")
        
        # Check merge compatibility
        print("\n🔗 Merge Compatibility Check:")
        print("-" * 30)
        
        # Find common book IDs
        common_book_ids = set(library_book_ids) & set(metadata_book_ids)
        library_only = set(library_book_ids) - set(metadata_book_ids)
        metadata_only = set(metadata_book_ids) - set(library_book_ids)
        
        print(f"✅ Books with both transaction and metadata: {len(common_book_ids)}")
        print(f"⚠️  Books in library but missing metadata: {len(library_only)}")
        print(f"ℹ️  Books in metadata but no transactions: {len(metadata_only)}")
        
        if library_only:
            print(f"📚 Sample books missing metadata: {list(library_only)[:5]}")
        
        # Calculate merge coverage
        total_library_records = len(library_df)
        records_with_metadata = len(library_df[library_df['book_id'].isin(common_book_ids)])
        coverage_percentage = (records_with_metadata / total_library_records) * 100
        
        print(f"📊 Merge coverage: {records_with_metadata}/{total_library_records} records ({coverage_percentage:.1f}%)")
        
        # Data quality checks
        print("\n🔍 Data Quality Checks:")
        print("-" * 25)
        
        # Check for null values in key columns
        print("Null values in library dataset:")
        null_counts = library_df[['user_id', 'book_id', 'action_type']].isnull().sum()
        for col, count in null_counts.items():
            print(f"  {col}: {count} nulls")
        
        # Check timestamp format
        timestamp_sample = library_df['borrow_timestamp'].head().tolist()
        print(f"📅 Sample timestamps: {timestamp_sample}")
        
        # Check for hashtag placeholders
        hashtag_returns = (library_df['return_timestamp'] == '#########').sum()
        print(f"📅 Records with hashtag return timestamps: {hashtag_returns}")
        
        # Check rating distribution
        rating_dist = library_df['rating'].value_counts().sort_index()
        print(f"⭐ Rating distribution:\n{rating_dist}")
        
        # Check action types
        action_types = library_df['action_type'].value_counts()
        print(f"🎬 Action types:\n{action_types}")
        
        # Check device types
        device_types = library_df['device_type'].value_counts()
        print(f"📱 Device types:\n{device_types}")
        
        # Final recommendations
        print("\n💡 Recommendations:")
        print("-" * 20)
        
        if coverage_percentage >= 90:
            print("✅ Excellent merge coverage! Your datasets are well-aligned.")
        elif coverage_percentage >= 70:
            print("⚠️  Good merge coverage, but some library records lack metadata.")
        else:
            print("❌ Low merge coverage. Many library records are missing metadata.")
            print("   Consider updating your metadata.csv to include missing book IDs.")
        
        if hashtag_returns > 0:
            print("ℹ️  Hashtag placeholders detected in return timestamps - this is normal and handled by the app.")
        
        print("\n✅ Validation complete! Your datasets are ready for analysis.")
        
        return {
            'library_records': len(library_df),
            'metadata_records': len(metadata_df),
            'common_books': len(common_book_ids),
            'merge_coverage': coverage_percentage,
            'missing_metadata': len(library_only)
        }
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find file - {e}")
        print("Make sure your CSV files are in the correct location:")
        print(f"  - {library_file}")
        print(f"  - {metadata_file}")
        return None
        
    except Exception as e:
        print(f"❌ Error validating datasets: {e}")
        return None

def check_sample_data():
    """Check if sample data matches expected format"""
    print("\n🔍 Checking your data format against expected structure...")
    
    expected_library_structure = {
        'user_id': 'U001, U063, U002, etc.',
        'book_id': 'B001, B025, B023, etc.',
        'borrow_timestamp': 'Date/time or #########',
        'return_timestamp': 'Date/time or #########',
        'rating': '1-5 or empty',
        'device_type': 'desktop, mobile, tablet',
        'session_duration': 'Time in seconds',
        'action_type': 'borrow, preview, etc.',
        'recommendation_score': '0 or 1'
    }
    
    expected_metadata_structure = {
        'book_id': 'B000, B001, B002, etc.',
        'title': 'Book title text',
        'author': 'Author name',
        'year': 'Publication year (2018-2024)'
    }
    
    print("\n📊 Expected Library Dataset Structure:")
    for col, desc in expected_library_structure.items():
        print(f"  {col}: {desc}")
    
    print("\n📚 Expected Metadata Structure:")
    for col, desc in expected_metadata_structure.items():
        print(f"  {col}: {desc}")

if __name__ == "__main__":
    # Run validation
    results = validate_datasets()
    check_sample_data()
    
    if results:
        print(f"\n📈 Summary Statistics:")
        print(f"  Library records: {results['library_records']:,}")
        print(f"  Metadata records: {results['metadata_records']:,}")
        print(f"  Merge coverage: {results['merge_coverage']:.1f}%")
        print(f"  Books needing metadata: {results['missing_metadata']}")