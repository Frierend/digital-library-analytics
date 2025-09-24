import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class InsightGenerator:
    """Generates automated insights from digital library data"""
    
    def __init__(self):
        self.insights = []
    
    def generate_insights(self, df):
        """Generate comprehensive insights from the dataset"""
        self.insights = []
        
        # Basic statistics insights
        self._analyze_basic_stats(df)
        
        # Temporal insights
        self._analyze_temporal_patterns(df)
        
        # Device insights
        self._analyze_device_patterns(df)
        
        # Rating insights
        self._analyze_rating_patterns(df)
        
        # User behavior insights
        self._analyze_user_behavior(df)
        
        # Book popularity insights
        self._analyze_book_popularity(df)
        
        # Recommendation insights
        self._analyze_recommendation_effectiveness(df)
        
        return self.insights
    
    def _add_insight(self, title, description, category="General", priority="Medium"):
        """Add an insight to the list"""
        self.insights.append({
            'title': title,
            'description': description,
            'category': category,
            'priority': priority
        })
    
    def _analyze_basic_stats(self, df):
        """Analyze basic statistics"""
        total_records = len(df)
        unique_users = df['user_id'].nunique() if 'user_id' in df.columns else 0
        unique_books = df['book_id'].nunique() if 'book_id' in df.columns else 0
        
        # Total borrows
        total_borrows = len(df[df['action_type'] == 'borrow']) if 'action_type' in df.columns else 0
        
        self._add_insight(
            "Library Usage Overview",
            f"The library has {total_records:,} total interactions from {unique_users:,} unique users across {unique_books:,} different books, with {total_borrows:,} actual borrows recorded.",
            "Statistics",
            "High"
        )
        
        # User engagement
        if unique_users > 0 and total_records > 0:
            avg_actions_per_user = total_records / unique_users
            if avg_actions_per_user > 10:
                self._add_insight(
                    "High User Engagement",
                    f"Users are highly engaged with an average of {avg_actions_per_user:.1f} actions per user, indicating strong library utilization.",
                    "User Behavior",
                    "Medium"
                )
            elif avg_actions_per_user < 3:
                self._add_insight(
                    "Low User Engagement",
                    f"Users show limited engagement with only {avg_actions_per_user:.1f} actions per user on average. Consider implementing engagement strategies.",
                    "User Behavior",
                    "High"
                )
            else:
                self._add_insight(
                    "Moderate User Engagement",
                    f"Users show moderate engagement with {avg_actions_per_user:.1f} actions per user on average. There's room for improvement.",
                    "User Behavior",
                    "Medium"
                )
        
        # Book utilization
        if unique_books > 0 and total_borrows > 0:
            avg_borrows_per_book = total_borrows / unique_books
            if avg_borrows_per_book < 5:
                self._add_insight(
                    "Low Book Utilization",
                    f"Each book is borrowed only {avg_borrows_per_book:.1f} times on average. Consider promoting underutilized content or reviewing collection relevance.",
                    "Content Strategy",
                    "Medium"
                )
            else:
                self._add_insight(
                    "Good Book Utilization",
                    f"Each book is borrowed {avg_borrows_per_book:.1f} times on average, indicating good collection utilization.",
                    "Content Strategy",
                    "Low"
                )
    
    def _analyze_temporal_patterns(self, df):
        """Analyze temporal patterns in borrowing"""
        if 'borrow_timestamp' not in df.columns:
            return
        
        borrow_df = df[df['action_type'] == 'borrow'].dropna(subset=['borrow_timestamp'])
        
        if len(borrow_df) == 0:
            return
        
        # Analyze peak hours
        if 'borrow_hour' in borrow_df.columns:
            hourly_counts = borrow_df['borrow_hour'].value_counts()
            peak_hour = hourly_counts.idxmax()
            peak_count = hourly_counts.max()
            
            if peak_hour < 6:
                time_desc = "early morning"
            elif peak_hour < 12:
                time_desc = "morning"
            elif peak_hour < 17:
                time_desc = "afternoon" 
            elif peak_hour < 21:
                time_desc = "evening"
            else:
                time_desc = "night"
            
            self._add_insight(
                "Peak Usage Time",
                f"Library usage peaks at {peak_hour}:00 ({time_desc}) with {peak_count} borrows. Consider optimizing system resources during this time.",
                "Temporal",
                "Medium"
            )
        
        # Analyze day of week patterns
        if 'borrow_day_of_week' in borrow_df.columns:
            dow_counts = borrow_df['borrow_day_of_week'].value_counts()
            busiest_day = dow_counts.idxmax()
            quietest_day = dow_counts.idxmin()
            
            self._add_insight(
                "Weekly Usage Pattern",
                f"{busiest_day} is the busiest day for library usage, while {quietest_day} sees the least activity. Plan maintenance and updates accordingly.",
                "Temporal",
                "Low"
            )
    
    def _analyze_device_patterns(self, df):
        """Analyze device usage patterns"""
        if 'device_type' not in df.columns:
            return
        
        device_counts = df['device_type'].value_counts()
        most_popular_device = device_counts.idxmax()
        device_percentage = (device_counts.iloc[0] / len(df)) * 100
        
        self._add_insight(
            "Preferred Access Method",
            f"{most_popular_device.title()} is the preferred access method, accounting for {device_percentage:.1f}% of all library interactions.",
            "Technology",
            "Medium"
        )
        
        # Session duration by device
        if 'session_duration' in df.columns:
            device_sessions = df.groupby('device_type')['session_duration'].mean()
            longest_session_device = device_sessions.idxmax()
            longest_session_time = device_sessions.max() / 60  # Convert to minutes
            
            if longest_session_time > 30:
                self._add_insight(
                    "Extended Reading Sessions",
                    f"Users on {longest_session_device} devices spend the most time reading with average sessions of {longest_session_time:.1f} minutes, indicating deep engagement.",
                    "User Behavior",
                    "Medium"
                )
    
    def _analyze_rating_patterns(self, df):
        """Analyze rating patterns"""
        if 'rating' not in df.columns:
            return
        
        ratings = df['rating'].dropna()
        if len(ratings) == 0:
            return
        
        avg_rating = ratings.mean()
        rating_std = ratings.std()
        
        # Overall satisfaction
        if avg_rating >= 4.0:
            satisfaction_level = "high"
        elif avg_rating >= 3.0:
            satisfaction_level = "moderate"
        else:
            satisfaction_level = "low"
        
        self._add_insight(
            "User Satisfaction",
            f"Average book rating is {avg_rating:.1f} out of 5, indicating {satisfaction_level} user satisfaction with the library collection.",
            "Quality",
            "High"
        )
        
        # Rating distribution
        rating_counts = ratings.value_counts().sort_index()
        five_star_percentage = (rating_counts.get(5, 0) / len(ratings)) * 100
        one_star_percentage = (rating_counts.get(1, 0) / len(ratings)) * 100
        
        if five_star_percentage > 30:
            self._add_insight(
                "Excellent Content Quality",
                f"{five_star_percentage:.1f}% of books receive 5-star ratings, indicating exceptional content quality in the library collection.",
                "Quality",
                "Medium"
            )
        
        if one_star_percentage > 10:
            self._add_insight(
                "Content Quality Concern",
                f"{one_star_percentage:.1f}% of books receive 1-star ratings. Consider reviewing and potentially removing poorly rated content.",
                "Quality",
                "High"
            )
    
    def _analyze_user_behavior(self, df):
        """Analyze user behavior patterns"""
        borrow_df = df[df['action_type'] == 'borrow']
        
        if len(borrow_df) == 0:
            return
        
        # User activity distribution
        user_activity = borrow_df['user_id'].value_counts()
        heavy_users = (user_activity >= user_activity.quantile(0.9)).sum()
        light_users = (user_activity <= user_activity.quantile(0.1)).sum()
        
        heavy_percentage = (heavy_users / len(user_activity)) * 100
        light_percentage = (light_users / len(user_activity)) * 100
        
        self._add_insight(
            "User Activity Distribution",
            f"{heavy_percentage:.1f}% of users are heavy borrowers (top 10%), while {light_percentage:.1f}% are light users (bottom 10%). Focus retention efforts on engaging light users.",
            "User Behavior",
            "Medium"
        )
        
        # Reading completion analysis
        if 'return_timestamp' in df.columns and 'borrow_timestamp' in df.columns:
            completed_borrows = borrow_df.dropna(subset=['return_timestamp', 'borrow_timestamp'])
            completion_rate = (len(completed_borrows) / len(borrow_df)) * 100
            
            if completion_rate > 80:
                self._add_insight(
                    "High Completion Rate",
                    f"{completion_rate:.1f}% of borrowed books are returned, indicating users are actively engaging with the content.",
                    "User Behavior",
                    "Medium"
                )
            elif completion_rate < 50:
                self._add_insight(
                    "Low Completion Rate",
                    f"Only {completion_rate:.1f}% of borrowed books are returned. Users may be abandoning books or system may have tracking issues.",
                    "User Behavior",
                    "High"
                )
    
    def _analyze_book_popularity(self, df):
        """Analyze book popularity patterns"""
        borrow_df = df[df['action_type'] == 'borrow']
        
        if len(borrow_df) == 0:
            return
        
        # Book popularity distribution
        book_popularity = borrow_df['title'].value_counts()
        
        if len(book_popularity) == 0:
            return
        
        # Popular books analysis
        top_10_borrows = book_popularity.head(10).sum()
        total_borrows = len(borrow_df)
        top_10_percentage = (top_10_borrows / total_borrows) * 100
        
        if top_10_percentage > 50:
            self._add_insight(
                "Concentrated Popularity",
                f"Top 10 books account for {top_10_percentage:.1f}% of all borrows, indicating concentrated user interest. Consider expanding similar content.",
                "Content Strategy",
                "Medium"
            )
        
        # Long tail analysis
        single_borrow_books = (book_popularity == 1).sum()
        single_borrow_percentage = (single_borrow_books / len(book_popularity)) * 100
        
        if single_borrow_percentage > 40:
            self._add_insight(
                "Long Tail Distribution",
                f"{single_borrow_percentage:.1f}% of books have been borrowed only once. Consider promoting underutilized content or curating the collection.",
                "Content Strategy",
                "Medium"
            )
        
        # Author popularity
        if 'author' in borrow_df.columns:
            author_popularity = borrow_df['author'].value_counts()
            top_author = author_popularity.index[0]
            top_author_borrows = author_popularity.iloc[0]
            
            self._add_insight(
                "Most Popular Author",
                f"{top_author} is the most popular author with {top_author_borrows} borrows. Consider acquiring more titles from popular authors.",
                "Content Strategy",
                "Low"
            )
    
    def _analyze_recommendation_effectiveness(self, df):
        """Analyze recommendation system effectiveness"""
        if 'is_recommended' not in df.columns or 'rating' not in df.columns:
            return
        
        # Compare ratings for recommended vs non-recommended books
        rec_data = df.dropna(subset=['is_recommended', 'rating'])
        
        if len(rec_data) == 0:
            return
        
        recommended_ratings = rec_data[rec_data['is_recommended'] == 1]['rating']
        non_recommended_ratings = rec_data[rec_data['is_recommended'] == 0]['rating']
        
        if len(recommended_ratings) > 0 and len(non_recommended_ratings) > 0:
            rec_avg = recommended_ratings.mean()
            non_rec_avg = non_recommended_ratings.mean()
            
            if rec_avg > non_rec_avg + 0.2:
                self._add_insight(
                    "Effective Recommendation System",
                    f"Recommended books have higher average ratings ({rec_avg:.1f}) compared to non-recommended books ({non_rec_avg:.1f}), indicating an effective recommendation system.",
                    "Recommendation System",
                    "Medium"
                )
            elif rec_avg < non_rec_avg - 0.2:
                self._add_insight(
                    "Recommendation System Needs Improvement",
                    f"Recommended books have lower average ratings ({rec_avg:.1f}) than non-recommended books ({non_rec_avg:.1f}). The recommendation algorithm may need refinement.",
                    "Recommendation System",
                    "High"
                )
        
        # Recommendation uptake
        rec_percentage = (len(recommended_ratings) / len(rec_data)) * 100
        
        if rec_percentage > 30:
            self._add_insight(
                "High Recommendation Uptake",
                f"{rec_percentage:.1f}% of library interactions involve recommended books, showing users are engaging with the recommendation system.",
                "Recommendation System",
                "Low"
            )
        elif rec_percentage < 10:
            self._add_insight(
                "Low Recommendation Uptake",
                f"Only {rec_percentage:.1f}% of interactions involve recommended books. Consider making recommendations more prominent or improving the algorithm.",
                "Recommendation System",
                "Medium"
            )
    
    def _analyze_seasonal_trends(self, df):
        """Analyze seasonal borrowing trends"""
        if 'borrow_timestamp' not in df.columns:
            return
        
        borrow_df = df[df['action_type'] == 'borrow'].dropna(subset=['borrow_timestamp'])
        
        if len(borrow_df) == 0:
            return
        
        # Add month information
        borrow_df_copy = borrow_df.copy()
        borrow_df_copy['month'] = borrow_df_copy['borrow_timestamp'].dt.month
        
        # Seasonal analysis
        monthly_counts = borrow_df_copy['month'].value_counts().sort_index()
        
        # Define seasons
        spring_months = [3, 4, 5]
        summer_months = [6, 7, 8]
        fall_months = [9, 10, 11]
        winter_months = [12, 1, 2]
        
        spring_borrows = monthly_counts[monthly_counts.index.isin(spring_months)].sum()
        summer_borrows = monthly_counts[monthly_counts.index.isin(summer_months)].sum()
        fall_borrows = monthly_counts[monthly_counts.index.isin(fall_months)].sum()
        winter_borrows = monthly_counts[monthly_counts.index.isin(winter_months)].sum()
        
        seasonal_data = {
            'Spring': spring_borrows,
            'Summer': summer_borrows,
            'Fall': fall_borrows,
            'Winter': winter_borrows
        }
        
        peak_season = max(seasonal_data.keys(), key=lambda x: seasonal_data[x])
        low_season = min(seasonal_data.keys(), key=lambda x: seasonal_data[x])
        
        self._add_insight(
            "Seasonal Usage Pattern",
            f"{peak_season} shows the highest library usage while {low_season} has the lowest. Consider seasonal marketing campaigns and content promotions.",
            "Temporal",
            "Low"
        )
    
    def generate_summary_report(self, df):
        """Generate a comprehensive summary report"""
        # Generate all insights
        insights = self.generate_insights(df)
        
        # Categorize insights
        high_priority = [i for i in insights if i['priority'] == 'High']
        medium_priority = [i for i in insights if i['priority'] == 'Medium']
        low_priority = [i for i in insights if i['priority'] == 'Low']
        
        # Create summary
        report = {
            'total_insights': len(insights),
            'high_priority_count': len(high_priority),
            'medium_priority_count': len(medium_priority),
            'low_priority_count': len(low_priority),
            'insights_by_category': {},
            'top_recommendations': high_priority[:5],  # Top 5 high priority insights
            'all_insights': insights
        }
        
        # Group by category
        categories = set([i['category'] for i in insights])
        for category in categories:
            report['insights_by_category'][category] = [
                i for i in insights if i['category'] == category
            ]
        
        return report
    
    def get_actionable_recommendations(self, df):
        """Get specific actionable recommendations based on insights"""
        insights = self.generate_insights(df)
        recommendations = []
        
        for insight in insights:
            if insight['priority'] == 'High':
                if 'engagement' in insight['description'].lower():
                    recommendations.append("Implement user engagement campaigns and personalized notifications")
                elif 'completion rate' in insight['description'].lower():
                    recommendations.append("Investigate user experience issues and improve book discovery")
                elif 'recommendation' in insight['description'].lower():
                    recommendations.append("Refine recommendation algorithm and A/B test different approaches")
                elif 'quality' in insight['description'].lower():
                    recommendations.append("Review and curate content library based on user feedback")
        
        return recommendations