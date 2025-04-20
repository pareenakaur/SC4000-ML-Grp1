"""
Submission Blending for Model Ensembling
========================================
This script combines predictions from multiple models with specialized handling for outliers.

Input Files:
1. bestline_submission_without_outliers.csv (source: code6_5.py)
2. bestline_submission_outliers_likelihood2.csv (source: code6.py)
3. bestline_submission_main_2.csv (source: code3.py) 

Output Files:
1. final_submission_threshold0.5.csv
2. final_submission_threshold0.05.csv
3. final_submission_threshold0.005.csv
4. final_submission.csv
"""

import datetime
import functools
import gc
import os
import sys
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

import numpy as np
import pandas as pd
from tqdm import tqdm


# Decorator for warning suppression
def suppress_warnings(func):
    """Decorator to suppress common warnings during function execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Store original filters
        original_filters = warnings.filters.copy()
        
        # Suppress warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter('ignore', UserWarning)
        warnings.simplefilter('ignore', DeprecationWarning)
        warnings.simplefilter('ignore', RuntimeWarning)
        
        try:
            # Execute function
            return func(*args, **kwargs)
        finally:
            # Restore original filters
            warnings.filters = original_filters
    
    return wrapper


# Configuration management with dataclass
@dataclass
class BlendingConfig:
    """Configuration for submission blending operations"""
    # Base directory for all files

    
    # Input files
    non_outlier_path: str
    outlier_likelihood_path: str
    main_submission_path: str
    
    
    # Thresholds for different blending strategies
    thresholds: List[float] = field(default_factory=lambda: [0.5, 0.05, 0.005])
    
    # Number of top outliers for final submission
    top_outliers: int = 25000
    
    def __post_init__(self):
        """Validate and adjust paths after initialization"""
        # Convert string paths to Path objects
        
        # Ensure paths are absolute
        if not self.non_outlier_path.startswith('/'):
            self.non_outlier_path = str(self.non_outlier_path)
        
        if not self.outlier_likelihood_path.startswith('/'):
            self.outlier_likelihood_path = str(self.outlier_likelihood_path)
            
        if not self.main_submission_path.startswith('/'):
            self.main_submission_path = str(self.main_submission_path)


# Strategy pattern for different blending approaches
class BlendingStrategy(ABC):
    """Abstract base class for different submission blending strategies"""
    @abstractmethod
    def select_outliers(self, outlier_df: pd.DataFrame, config: BlendingConfig) -> pd.DataFrame:
        """Select outliers based on specific strategy"""
        pass
    
    @abstractmethod
    def get_output_filename(self, config: BlendingConfig) -> str:
        """Get appropriate output filename for this strategy"""
        pass


class ThresholdBasedStrategy(BlendingStrategy):
    """Strategy to select outliers based on a threshold value"""
    def __init__(self, threshold: float):
        self.threshold = threshold
    
    def select_outliers(self, outlier_df: pd.DataFrame, config: BlendingConfig) -> pd.DataFrame:
        """Select outliers with likelihood above threshold"""
        # Sort by target (likelihood) in descending order
        sorted_df = outlier_df.sort_values(by='target', ascending=False)
        
        # Filter by threshold
        outliers = sorted_df[sorted_df['target'] > self.threshold]
        
        return outliers
    
    def get_output_filename(self, config: BlendingConfig) -> str:
        """Create output filename with threshold value"""
        return str(f'final_submission_threshold{self.threshold}.csv')


class TopNStrategy(BlendingStrategy):
    """Strategy to select the top N most likely outliers"""
    def __init__(self, n: int):
        self.n = n
    
    def select_outliers(self, outlier_df: pd.DataFrame, config: BlendingConfig) -> pd.DataFrame:
        """Select top N outliers by likelihood"""
        # Sort by target (likelihood) in descending order and take top N
        sorted_df = outlier_df.sort_values(by='target', ascending=False)
        outliers = sorted_df.head(self.n)[['card_id']]
        
        return outliers
    
    def get_output_filename(self, config: BlendingConfig) -> str:
        """Create output filename for top N strategy"""
        return str('final_submission.csv')


# Command to execute blending operation
class BlendSubmissions:
    """Command to blend predictions from multiple submissions"""
    def __init__(self, config: BlendingConfig, strategy: BlendingStrategy):
        self.config = config
        self.strategy = strategy
        self.non_outlier_submission = None
        self.outlier_likelihood = None
        self.main_submission = None
        self.outlier_ids = None
        self.blended_submission = None
    
    def load_submissions(self):
        """Load all required submission files"""
        print(f'[INFO] Loading submission files')
        
        self.non_outlier_submission = pd.read_csv(self.config.non_outlier_path)
        self.outlier_likelihood = pd.read_csv(self.config.outlier_likelihood_path)
        self.main_submission = pd.read_csv(self.config.main_submission_path)
        
        return self
    
    def identify_outliers(self):
        """Identify outliers using the selected strategy"""
        print(f'[INFO] Identifying outliers using {self.strategy.__class__.__name__}')
        
        self.outlier_ids = self.strategy.select_outliers(self.outlier_likelihood, self.config)
        print(f'[INFO] Found {len(self.outlier_ids)} outliers')
        
        return self
    
    def blend_predictions(self):
        """Blend predictions by replacing values for outlier IDs"""
        print(f'[INFO] Blending predictions')
        
        # Make a copy of the non-outlier submission
        self.blended_submission = self.non_outlier_submission.copy()
        
        # Get outlier card IDs
        outlier_card_ids = self.outlier_ids['card_id'].values
        
        # Filter main submission to only include outliers
        outlier_predictions = self.main_submission[
            self.main_submission['card_id'].isin(outlier_card_ids)
        ]
        
        # Create a progress bar for the update operation
        progress_desc = f"Updating targets for {len(outlier_card_ids)} outliers"
        
        # Replace values for outliers all at once (vectorized operation)
        mask = self.blended_submission['card_id'].isin(outlier_card_ids)
        
        # Show progress with tqdm
        with tqdm(total=1, desc=progress_desc, leave=False) as pbar:
            self.blended_submission.loc[mask, 'target'] = outlier_predictions['target'].values
            pbar.update(1)
        
        return self
    
    def save_submission(self):
        """Save the blended submission to file"""
        output_path = self.strategy.get_output_filename(self.config)
        
        print(f'[INFO] Saving blended submission to {output_path}')
        self.blended_submission.to_csv(output_path, index=False)
        
        return self
    
    def execute(self):
        """Execute the entire blending process"""
        return (self
                .load_submissions()
                .identify_outliers()
                .blend_predictions()
                .save_submission())


# Orchestrator for multiple blending operations
class SubmissionBlender:
    """Orchestrates multiple blending operations with different strategies"""
    def __init__(self, config: BlendingConfig):
        self.config = config
        self.strategies = []
    
    def add_strategy(self, strategy: BlendingStrategy):
        """Add a blending strategy to execute"""
        self.strategies.append(strategy)
        return self
    
    def blend_all(self):
        """Execute all blending strategies"""
        for strategy in self.strategies:
            print(f'\n[STRATEGY] Executing {strategy.__class__.__name__}')
            
            command = BlendSubmissions(self.config, strategy)
            command.execute()
            
            # Clean up to reduce memory usage
            gc.collect()
        
        print('\n[SUCCESS] All blending operations completed')
        return self


@suppress_warnings
def main():
    """Main function to orchestrate the blending process"""
    # Configure the blending operations
    config = BlendingConfig(
        non_outlier_path='bestline_submission_without_outliers.csv',
        outlier_likelihood_path='enhanced_submission_outliers_likelihood.csv',
        main_submission_path='code3_bestline_submission_main_2.csv',
        
    )
    
    # Create and configure the blender
    blender = SubmissionBlender(config)
    
    # Add threshold-based strategies
    for threshold in config.thresholds:
        blender.add_strategy(ThresholdBasedStrategy(threshold))
    
    # Add top-N strategy
    blender.add_strategy(TopNStrategy(config.top_outliers))
    
    # Execute all blending operations
    blender.blend_all()


if __name__ == "__main__":
    main()