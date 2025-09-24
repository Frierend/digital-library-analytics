import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class Visualizer:
    """Handles all visualization tasks for the digital library analysis"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        self.theme = {
            'background_color': '#ffffff',
            'grid_color': '#f0f0f0',
            'text_color': '#333333'
        }
    
    def plot_top_borrowed_books(self, df, top_n=10):
        """Create bar chart of top borrowed books with improved clarity"""
        # Filter for borrow actions only (consistent with association rules)
        borrow_df = df[df['action_type'] == 'borrow'].copy()
        
        if len(borrow_df) == 0:
            return self._create_empty_plot("No borrowing data available")
        
        # Count borrows per book title (not book_id)
        book_counts = borrow_df['title'].value_counts().head(top_n)
        
        if len(book_counts) == 0:
            return self._create_empty_plot("No book data available")
        
        # Create horizontal bar chart with better formatting
        fig = px.bar(
            x=book_counts.values,
            y=book_counts.index,
            orientation='h',
            title=f"Top {top_n} Most Borrowed Books (Actual Borrows Only)",
            labels={'x': 'Number of Times Borrowed', 'y': 'Book Title'},
            color=book_counts.values,
            color_continuous_scale='viridis',
            text=book_counts.values  # Add value labels
        )
        
        # Improve layout
        fig.update_layout(
            height=max(400, top_n * 40),  # Dynamic height based on number of books
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False,
            plot_bgcolor='white',
            xaxis_title="Number of Borrows",
            yaxis_title="Book Title",
            font=dict(size=12)
        )
        
        # Add value labels on bars
        fig.update_traces(
            texttemplate='%{text}',
            textposition='outside',
            textfont_size=10
        )
        
        # Improve hover information
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>Borrowed %{x} times<extra></extra>'
        )
        
        return fig
    
    def plot_borrowing_trends(self, df):
        """Create improved line chart of borrowing trends over time"""
        borrow_df = df[df['action_type'] == 'borrow'].copy()
        
        if len(borrow_df) == 0 or 'borrow_timestamp' not in borrow_df.columns:
            return self._create_empty_plot("No borrowing timestamp data available")
        
        # Remove rows with null timestamps
        borrow_df = borrow_df.dropna(subset=['borrow_timestamp'])
        
        if len(borrow_df) == 0:
            return self._create_empty_plot("No valid timestamp data available")
        
        # Group by date and count
        daily_borrows = borrow_df.groupby(borrow_df['borrow_timestamp'].dt.date).size().reset_index()
        daily_borrows.columns = ['date', 'borrows']
        
        # Calculate moving average for smoother trend
        if len(daily_borrows) > 7:
            daily_borrows['moving_avg'] = daily_borrows['borrows'].rolling(window=7, center=True).mean()
        
        # Create improved line chart
        fig = go.Figure()
        
        # Add main trend line
        fig.add_trace(go.Scatter(
            x=daily_borrows['date'],
            y=daily_borrows['borrows'],
            mode='lines+markers',
            name='Daily Borrows',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=6, color='#2E86AB'),
            hovertemplate='<b>%{x}</b><br>Borrows: %{y}<extra></extra>'
        ))
        
        # Add moving average if available
        if 'moving_avg' in daily_borrows.columns:
            fig.add_trace(go.Scatter(
                x=daily_borrows['date'],
                y=daily_borrows['moving_avg'],
                mode='lines',
                name='7-Day Average',
                line=dict(color='#FF6B6B', width=2, dash='dot'),
                hovertemplate='<b>%{x}</b><br>7-Day Avg: %{y:.1f}<extra></extra>'
            ))
        
        # Improve layout
        fig.update_layout(
            title='Daily Borrowing Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Number of Borrows',
            height=400,
            plot_bgcolor='white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add grid for better readability
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # Add annotations for highest and lowest points
        max_idx = daily_borrows['borrows'].idxmax()
        min_idx = daily_borrows['borrows'].idxmin()
        
        fig.add_annotation(
            x=daily_borrows.iloc[max_idx]['date'],
            y=daily_borrows.iloc[max_idx]['borrows'],
            text=f"Peak: {daily_borrows.iloc[max_idx]['borrows']} borrows",
            showarrow=True,
            arrowhead=2,
            arrowcolor='green',
            bgcolor='lightgreen',
            bordercolor='green'
        )
        
        return fig
    
    def plot_device_usage(self, df):
        """Create pie chart of device usage"""
        if 'device_type' not in df.columns:
            return self._create_empty_plot("No device data available")
        
        device_counts = df['device_type'].value_counts()
        
        if len(device_counts) == 0:
            return self._create_empty_plot("No device data available")
        
        # Create pie chart
        fig = px.pie(
            values=device_counts.values,
            names=device_counts.index,
            title='Library Access by Device Type'
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        
        fig.update_layout(height=400)
        
        return fig
    
    def plot_rating_distribution(self, df):
        """Create histogram of rating distribution"""
        if 'rating' not in df.columns:
            return self._create_empty_plot("No rating data available")
        
        # Remove null ratings
        rating_data = df['rating'].dropna()
        
        if len(rating_data) == 0:
            return self._create_empty_plot("No valid rating data available")
        
        # Create histogram
        fig = px.histogram(
            x=rating_data,
            nbins=5,
            title='Distribution of Book Ratings',
            labels={'x': 'Rating', 'y': 'Frequency'},
            color_discrete_sequence=['#F39C12']
        )
        
        fig.update_layout(
            height=400,
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            plot_bgcolor=self.theme['background_color']
        )
        
        return fig
    
    def plot_session_duration_by_device(self, df):
        """Create box plot of session duration by device type"""
        if 'session_duration' not in df.columns or 'device_type' not in df.columns:
            return self._create_empty_plot("No session duration or device data available")
        
        # Remove null values
        plot_data = df.dropna(subset=['session_duration', 'device_type'])
        
        if len(plot_data) == 0:
            return self._create_empty_plot("No valid session duration data available")
        
        # Convert to minutes for better readability
        plot_data = plot_data.copy()
        plot_data['session_minutes'] = plot_data['session_duration'] / 60
        
        # Create box plot
        fig = px.box(
            plot_data,
            x='device_type',
            y='session_minutes',
            title='Session Duration by Device Type',
            labels={'device_type': 'Device Type', 'session_minutes': 'Session Duration (minutes)'}
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor=self.theme['background_color']
        )
        
        return fig
    
    def plot_rating_by_device(self, df):
        """Create violin plot of ratings by device type"""
        if 'rating' not in df.columns or 'device_type' not in df.columns:
            return self._create_empty_plot("No rating or device data available")
        
        # Remove null values
        plot_data = df.dropna(subset=['rating', 'device_type'])
        
        if len(plot_data) == 0:
            return self._create_empty_plot("No valid rating/device data available")
        
        # Create violin plot
        fig = px.violin(
            plot_data,
            x='device_type',
            y='rating',
            title='Rating Distribution by Device Type',
            labels={'device_type': 'Device Type', 'rating': 'Rating'},
            box=True
        )
        
        fig.update_layout(
            height=400,
            yaxis=dict(tickmode='linear', tick0=1, dtick=1),
            plot_bgcolor=self.theme['background_color']
        )
        
        return fig
    
    def plot_hourly_activity(self, df):
        """Create bar chart of borrowing activity by hour"""
        borrow_df = df[df['action_type'] == 'borrow'].copy()
        
        if len(borrow_df) == 0 or 'borrow_hour' not in borrow_df.columns:
            return self._create_empty_plot("No hourly activity data available")
        
        # Remove null values
        hourly_data = borrow_df.dropna(subset=['borrow_hour'])
        
        if len(hourly_data) == 0:
            return self._create_empty_plot("No valid hourly data available")
        
        # Count by hour
        hourly_counts = hourly_data['borrow_hour'].value_counts().sort_index()
        
        # Create bar chart
        fig = px.bar(
            x=hourly_counts.index,
            y=hourly_counts.values,
            title='Borrowing Activity by Hour of Day',
            labels={'x': 'Hour of Day', 'y': 'Number of Borrows'},
            color=hourly_counts.values,
            color_continuous_scale='plasma'
        )
        
        fig.update_layout(
            height=400,
            xaxis=dict(tickmode='linear', tick0=0, dtick=2),
            showlegend=False,
            plot_bgcolor=self.theme['background_color']
        )
        
        return fig
    
    def plot_monthly_trends(self, df):
        """Create line chart of monthly borrowing trends"""
        borrow_df = df[df['action_type'] == 'borrow'].copy()
        
        if len(borrow_df) == 0 or 'borrow_timestamp' not in borrow_df.columns:
            return self._create_empty_plot("No monthly trend data available")
        
        # Remove null timestamps
        borrow_df = borrow_df.dropna(subset=['borrow_timestamp'])
        
        if len(borrow_df) == 0:
            return self._create_empty_plot("No valid timestamp data available")
        
        # Group by year-month
        borrow_df['year_month'] = borrow_df['borrow_timestamp'].dt.to_period('M')
        monthly_counts = borrow_df['year_month'].value_counts().sort_index()
        
        # Create line chart
        fig = px.line(
            x=monthly_counts.index.astype(str),
            y=monthly_counts.values,
            title='Monthly Borrowing Trends',
            labels={'x': 'Month', 'y': 'Number of Borrows'}
        )
        
        fig.update_traces(line_color='#E74C3C', line_width=3, mode='lines+markers')
        fig.update_layout(
            height=400,
            plot_bgcolor=self.theme['background_color']
        )
        
        return fig
    
    def plot_author_popularity(self, df, top_n=10):
        """Create bar chart of most popular authors"""
        borrow_df = df[df['action_type'] == 'borrow'].copy()
        
        if len(borrow_df) == 0 or 'author' not in borrow_df.columns:
            return self._create_empty_plot("No author data available")
        
        # Remove null authors
        borrow_df = borrow_df.dropna(subset=['author'])
        
        if len(borrow_df) == 0:
            return self._create_empty_plot("No valid author data available")
        
        # Count borrows per author
        author_counts = borrow_df['author'].value_counts().head(top_n)
        
        # Create bar chart
        fig = px.bar(
            x=author_counts.values,
            y=author_counts.index,
            orientation='h',
            title=f'Top {top_n} Most Popular Authors',
            labels={'x': 'Number of Borrows', 'y': 'Author'},
            color=author_counts.values,
            color_continuous_scale='sunset'
        )
        
        fig.update_layout(
            height=400,
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False,
            plot_bgcolor=self.theme['background_color']
        )
        
        return fig
    
    def plot_recommendation_effectiveness(self, df):
        """Create comparison chart of recommended vs non-recommended books"""
        if 'is_recommended' not in df.columns or 'rating' not in df.columns:
            return self._create_empty_plot("No recommendation or rating data available")
        
        # Remove null values
        plot_data = df.dropna(subset=['is_recommended', 'rating'])
        
        if len(plot_data) == 0:
            return self._create_empty_plot("No valid recommendation/rating data available")
        
        # Create comparison
        plot_data['recommendation_status'] = plot_data['is_recommended'].map({0: 'Not Recommended', 1: 'Recommended'})
        
        # Create box plot
        fig = px.box(
            plot_data,
            x='recommendation_status',
            y='rating',
            title='Rating Comparison: Recommended vs Non-Recommended Books',
            labels={'recommendation_status': 'Recommendation Status', 'rating': 'Rating'},
            color='recommendation_status'
        )
        
        fig.update_layout(
            height=400,
            yaxis=dict(tickmode='linear', tick0=1, dtick=1),
            showlegend=False,
            plot_bgcolor=self.theme['background_color']
        )
        
        return fig
    
    def _create_empty_plot(self, message):
        """Create an empty plot with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor='center',
            yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=400,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor=self.theme['background_color']
        )
        return fig
    
    def create_summary_dashboard(self, df):
        """Create a comprehensive summary dashboard"""
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Top Books', 'Device Usage', 'Ratings', 'Hourly Activity'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Add plots (simplified versions for dashboard)
        try:
            # Top books
            borrow_df = df[df['action_type'] == 'borrow']
            top_books = borrow_df['title'].value_counts().head(5)
            fig.add_trace(
                go.Bar(x=top_books.values, y=top_books.index, orientation='h'),
                row=1, col=1
            )
            
            # Device usage
            device_counts = df['device_type'].value_counts()
            fig.add_trace(
                go.Pie(labels=device_counts.index, values=device_counts.values),
                row=1, col=2
            )
            
            # Ratings
            ratings = df['rating'].dropna()
            fig.add_trace(
                go.Histogram(x=ratings, nbinsx=5),
                row=2, col=1
            )
            
            # Hourly activity (if available)
            if 'borrow_hour' in df.columns:
                hourly = df[df['action_type'] == 'borrow']['borrow_hour'].value_counts().sort_index()
                fig.add_trace(
                    go.Bar(x=hourly.index, y=hourly.values),
                    row=2, col=2
                )
        
        except Exception as e:
            print(f"Error creating dashboard: {str(e)}")
        
        fig.update_layout(height=800, showlegend=False, title_text="Library Analytics Dashboard")
        return fig