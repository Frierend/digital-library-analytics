"""
Digital Library Analytics Utils Package

This package contains utility modules for the digital library analytics application:
- preprocessing: Data cleaning and preparation
- pattern_mining: Association rule mining and market basket analysis  
- visualization: Chart and graph generation
- insights: Automated insight generation
"""

from .preprocessing import DataPreprocessor
from .pattern_mining import PatternMiner
from .visualization import Visualizer
from .insights import InsightGenerator

__all__ = ['DataPreprocessor', 'PatternMiner', 'Visualizer', 'InsightGenerator']
__version__ = '1.0.0'