"""
PRiSM Plotting Architecture

This module provides the refactored plotting architecture with explicit index space
handling, decoupled components, and centralized services.

Components:
- IndexMapper: Handles conversions between OHE, Collapsed, and Dense index spaces
- FeatureMetadataRegistry: Central repository for feature properties
- FeatureMetadata: Dataclass for storing feature metadata
- PlottingPipeline: Orchestrates data preparation and transformation
- NomogramRenderer: Pure rendering without LASSO dependencies
- PlotFormatter: Formatting utilities for nomogram plots
- save_nomogram_csv: CSV export for nomogram data
"""

from prism.plotting.csv_export import save_nomogram_csv
from prism.plotting.formatter import (
    PlotFormatter,
    create_continuous_formatter,
    create_log_formatter,
    create_response_value_formatter,
    format_value,
    get_nice_ticks,
)
from prism.plotting.index_mapper import IndexMapper
from prism.plotting.json_export import (
    NomogramData,
    load_nomogram_json,
    save_nomogram_json,
)
from prism.plotting.metadata import FeatureMetadata, FeatureMetadataRegistry
from prism.plotting.pipeline import PlottingPipeline
from prism.plotting.renderer import NomogramRenderer

__all__ = [
    'IndexMapper',
    'FeatureMetadata',
    'FeatureMetadataRegistry',
    'PlottingPipeline',
    'NomogramRenderer',
    'PlotFormatter',
    'create_continuous_formatter',
    'create_log_formatter',
    'create_response_value_formatter',
    'format_value',
    'get_nice_ticks',
    'save_nomogram_csv',
    'save_nomogram_json',
    'load_nomogram_json',
    'NomogramData',
]
