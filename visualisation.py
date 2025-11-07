"""
Visualization functions for NotchedBeam project

This module provides plotting functions for crack analysis and visualization,
including comprehensive plotting configuration and styling.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import matplotlib.font_manager as fm
from typing import Dict, List, Tuple, Optional
from ipywidgets import (
    Dropdown,
    VBox,
    interactive_output,
)
from IPython.display import display
import layout as lo

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================

# Global visualization settings
GLOBAL_FONT_SIZE = 20
GLOBAL_FIG_SIZE = (8, 8)  # Quadratic figure
GLOBAL_DPI = 100

# Default color cycle for plots
DEFAULT_COLOR_CYCLE = [
    lo.COLORS['red'],
    lo.COLORS['orange'], 
    lo.COLORS['gold'],
    lo.COLORS['blue'],
    lo.COLORS['lightblue'],
    lo.COLORS['indigo'],
]

# General plotting styles
PLOT_STYLES = {
    'figure_size': (10, 6),
    'dpi': 300,
    'font_size': 12,
    'line_width': 2,
    'marker_size': 6,
    'grid_alpha': 0.3,
    'legend_frameon': True,
    'legend_fancybox': True,
    'legend_shadow': True
}

# Crack annotator specific styles
CRACK_ANNOTATOR_STYLES = {
    'figure_size': (12, 8),
    'dpi': 100,
    'font_size': 14,
    'title_font_size': 14,
    'instruction_font_size': 14,
    'line_width': 2,
    'marker_size': 6,
    'grid_alpha': 0.3,
    'legend_frameon': False,
    'legend_fancybox': True,
    'legend_shadow': False,
    'legend_facecolor': 'white'
}

# Visualization-specific styles (for crack heatmap, rose diagram, etc.)
VISUALIZATION_STYLES = {
    'figure_size': (8, 8),  # Quadratic
    'dpi': 100,
    'font_size': 20,
    'line_width': 3,
    'marker_size': 8,
    'grid_alpha': 0.0,  # No grid
    'legend_frameon': False,
    'legend_fancybox': False,
    'legend_shadow': False,
    'legend_facecolor': 'none',
    'legend_edgecolor': 'none',
    'tick_direction': 'in',  # Ticks pointing inside
    'spine_visibility': {'top': False, 'right': False, 'bottom': True, 'left': True}
}

# ============================================================================
# STYLE SETUP FUNCTIONS
# ============================================================================

def setup_font():
    """Setup font configuration with Minion Pro or similar serif font."""
    # List of preferred fonts in order of preference
    preferred_fonts = [
        'Minion Pro',
        'Minion Pro Regular',
        'Times New Roman',
        'Times',
        'Liberation Serif',
        'DejaVu Serif',
        'serif'
    ]
    
    # Get available fonts
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # Find the first available preferred font
    selected_font = None
    for font in preferred_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        print(f"Using font: {selected_font}")
        return selected_font
    else:
        print("Using default serif font")
        return 'serif'

# Set the font
SELECTED_FONT = setup_font()

def setup_matplotlib_style():
    """Configure matplotlib with general project-specific styling."""
    plt.rcParams.update({
        'figure.figsize': PLOT_STYLES['figure_size'],
        'figure.dpi': PLOT_STYLES['dpi'],
        'font.size': PLOT_STYLES['font_size'],
        'font.family': SELECTED_FONT,
        'lines.linewidth': PLOT_STYLES['line_width'],
        'lines.markersize': PLOT_STYLES['marker_size'],
        'grid.alpha': PLOT_STYLES['grid_alpha'],
        'legend.frameon': PLOT_STYLES['legend_frameon'],
        'legend.fancybox': PLOT_STYLES['legend_fancybox'],
        'legend.shadow': PLOT_STYLES['legend_shadow'],
        'axes.prop_cycle': plt.cycler('color', DEFAULT_COLOR_CYCLE)
    })

def setup_crack_annotator_style():
    """Configure matplotlib with crack annotator specific styling."""
    plt.rcParams.update({
        'figure.figsize': CRACK_ANNOTATOR_STYLES['figure_size'],
        'figure.dpi': CRACK_ANNOTATOR_STYLES['dpi'],
        'font.size': CRACK_ANNOTATOR_STYLES['font_size'],
        'font.family': SELECTED_FONT,
        'lines.linewidth': CRACK_ANNOTATOR_STYLES['line_width'],
        'lines.markersize': CRACK_ANNOTATOR_STYLES['marker_size'],
        'grid.alpha': CRACK_ANNOTATOR_STYLES['grid_alpha'],
        'legend.frameon': CRACK_ANNOTATOR_STYLES['legend_frameon'],
        'legend.fancybox': CRACK_ANNOTATOR_STYLES['legend_fancybox'],
        'legend.shadow': CRACK_ANNOTATOR_STYLES['legend_shadow'],
        'legend.facecolor': CRACK_ANNOTATOR_STYLES['legend_facecolor'],
        'axes.prop_cycle': plt.cycler('color', DEFAULT_COLOR_CYCLE)
    })

def setup_visualization_style():
    """Setup global matplotlib style for visualizations (crack heatmap, rose diagram, etc.)."""
    plt.rcParams.update({
        'font.size': VISUALIZATION_STYLES['font_size'],
        'font.family': SELECTED_FONT,
        'axes.labelsize': VISUALIZATION_STYLES['font_size'],
        'axes.titlesize': VISUALIZATION_STYLES['font_size'],
        'xtick.labelsize': VISUALIZATION_STYLES['font_size'],
        'ytick.labelsize': VISUALIZATION_STYLES['font_size'],
        'legend.fontsize': VISUALIZATION_STYLES['font_size'],
        'figure.titlesize': VISUALIZATION_STYLES['font_size'],
        'lines.linewidth': VISUALIZATION_STYLES['line_width'],
        'lines.markersize': VISUALIZATION_STYLES['marker_size'],
        'axes.grid': False,  # No grid lines
        'xtick.direction': VISUALIZATION_STYLES['tick_direction'],
        'ytick.direction': VISUALIZATION_STYLES['tick_direction'],
        'legend.frameon': VISUALIZATION_STYLES['legend_frameon'],
        'legend.fancybox': VISUALIZATION_STYLES['legend_fancybox'],
        'legend.shadow': VISUALIZATION_STYLES['legend_shadow'],
        'legend.facecolor': VISUALIZATION_STYLES['legend_facecolor'],
        'legend.edgecolor': VISUALIZATION_STYLES['legend_edgecolor'],
        'axes.prop_cycle': plt.cycler('color', DEFAULT_COLOR_CYCLE)
    })

# ============================================================================
# PLOTTING UTILITY FUNCTIONS
# ============================================================================

def apply_axis_styling(ax, style_config=None):
    """Apply consistent axis styling to a matplotlib axis."""
    if style_config is None:
        style_config = VISUALIZATION_STYLES
    
    # Hide/show spines based on configuration
    spine_visibility = style_config.get('spine_visibility', {
        'top': False, 'right': False, 'bottom': True, 'left': True
    })
    
    for spine_name, visible in spine_visibility.items():
        ax.spines[spine_name].set_visible(visible)
    
    # Set tick parameters
    ax.tick_params(
        axis='both', 
        which='major', 
        labelsize=style_config.get('font_size', GLOBAL_FONT_SIZE),
        pad=10,
        length=8,
        width=1.5
    )

def create_legend_with_style(ax, style_config=None):
    """Create a legend with consistent styling."""
    if style_config is None:
        style_config = VISUALIZATION_STYLES
    
    legend = ax.legend(
        bbox_to_anchor=(1.05, 1), 
        loc='upper left', 
        fontsize=style_config.get('font_size', GLOBAL_FONT_SIZE)
    )
    
    # Apply legend styling
    legend.set_frame_on(style_config.get('legend_frameon', False))
    if not style_config.get('legend_frameon', False):
        legend.get_frame().set_facecolor('none')
        legend.get_frame().set_edgecolor('none')
    
    return legend

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def distance_force_plot(df_or_path, title="Distance vs Force", figsize=None, dpi=100):
    """
    Create a plot showing distance vs force curves.
    
    Parameters:
    -----------
    df_or_path : pd.DataFrame or str
        Either a DataFrame containing:
        - w: distance data as lists (in mm)
        - P: force data as lists (in N)
        - AFN: Automated File Number 
        OR a filepath to a Parquet file containing the same data
        
    title : str, optional
        Title for the plot (default: "Distance vs Force")
        
    figsize : tuple, optional
        Figure size in inches (default: uses VISUALIZATION_STYLES['figure_size'])
        
    dpi : int, optional
        Figure resolution (default: 100)
        
    Returns:
    --------
    None
        Displays interactive plot with AFN dropdown
    """
    
    # Handle input: DataFrame or filepath
    if isinstance(df_or_path, str):
        # Load from Parquet file
        try:
            df = pd.read_parquet(df_or_path, engine='fastparquet')
            print(f"Loaded data from Parquet: {df_or_path}")
        except Exception as e:
            print(f"Error loading Parquet file {df_or_path}: {e}")
            return None
    else:
        # Use DataFrame directly
        df = df_or_path
    
    # Check if the DataFrame is empty
    if df.empty:
        print("No data to visualize")
        return None
    
    # Check if required columns exist
    if 'w_data' not in df.columns or 'P_data' not in df.columns:
        print("Error: Required columns 'w_data' and 'P_data' not found in data")
        return None
    
    # Check if AFN column exists
    if 'AFN' not in df.columns:
        print("Error: AFN column required")
        return None
    
    # Get list of available AFNs (filter out NA values)
    afn_list = sorted(df['AFN'].dropna().unique())
    
    if not afn_list:
        print("No AFN values found in data")
        return None
    
    # Create AFN dropdown
    afn_dropdown = Dropdown(
        options=afn_list,
        value=afn_list[0],
        description='AFN:',
        continuous_update=False
    )
    
    # Plotting function that will be called when AFN changes
    def plot_afn(afn):
        """Plot distance vs force for selected AFN."""
        # Get data for selected AFN
        row = df[df['AFN'] == afn]
        
        if row.empty:
            print(f"No data found for AFN {afn}")
            return
        
        row = row.iloc[0]
        w = row['w_data']
        P = row['P_data']
        
        # Check if data is valid
        if w is None or P is None or len(w) == 0 or len(P) == 0:
            print(f"No valid data for AFN {afn}")
            return
        
        # Setup visualization style
        setup_visualization_style()
        
        # Use global figure size if none specified
        fig_size = figsize if figsize is not None else VISUALIZATION_STYLES['figure_size']
        
        # Create figure
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        
        # Plot the curve
        ax.plot(w, P, color=lo.COLORS['blue'], alpha=1, linewidth=2.0)
        
        # Set labels and title
        ax.set_xlabel(r'Distance (mm) $\longrightarrow$', fontsize=VISUALIZATION_STYLES['font_size'], labelpad=15)
        ax.set_ylabel(r'Force (N) $\longrightarrow$', fontsize=VISUALIZATION_STYLES['font_size'], labelpad=15)
        ax.set_title(f'{title} - AFN {afn}', fontsize=VISUALIZATION_STYLES['font_size'], pad=20)
        
        # Set tick label sizes and padding
        ax.tick_params(axis='both', which='major', labelsize=VISUALIZATION_STYLES['font_size'], 
                       pad=10, direction='in', length=6)
        
        # Show all four spines (complete frame)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        
        # Remove grid
        ax.grid(False)
        
        # Tight layout
        plt.tight_layout()
        plt.show()
    
    # Connect dropdown to plotting function
    out = interactive_output(plot_afn, {'afn': afn_dropdown})
    
    # Display UI
    controls = VBox([afn_dropdown])
    display(out, controls)
    
    # Return None to ensure no extra output
    return None
