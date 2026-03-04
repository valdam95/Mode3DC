"""
Visualization functions for NotchedBeam project

This module provides plotting functions for crack analysis and visualization,
including comprehensive plotting configuration and styling.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
import numpy as np

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================

# Global visualization settings
GLOBAL_FONT_SIZE = 20
GLOBAL_FIG_SIZE = (8, 7.5)
GLOBAL_DPI = 100
COLORS = lo.COLORS

# ============================================================================
# GLOBAL PLOT SERIES COLORS AND MARKERS
# ============================================================================

PLOTLINE_ALPHA = 0.8
PLOTPOINT_ALPHA = 0.8

TICK_WIDTH = 1
TICK_LENGTH = 10
TICK_COLOR = 'grey'
FRAME_THICKNESS = 1

MARKER_FC1 = '^'
MARKER_FC2 = 's'
MARKER_RG1 = 'o'
MARKER_RG2 = 'D'

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
    'figure_size': (8, 7.5),
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
    'figure_size': (8, 7.5),
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

# Default error-bar layout (color stays local to each plot/function)
ERRORBAR_STYLES = {
    'line_width': 1,
    'cap_size': 0,
    'alpha': 0.5,
}

# ============================================================================
# STYLE SETUP FUNCTIONS
# ============================================================================


def setup_font():
    """Setup font configuration with Minion Pro or similar serif font.
    
    Also configures math mode to use a compatible serif font:
    - If Minion Pro is available, uses STIX for math mode (serif math font)
    - Otherwise, uses the same serif font for both text and math mode
    
    Returns:
        tuple: (selected_font, math_font_set) where math_font_set is the math font configuration
    """
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
        
        # Configure math mode font
        if 'Minion Pro' in selected_font:
            # Minion Pro doesn't have a math variant, use STIX (serif math font) as equivalent
            math_font_set = 'stix'
            print("Using STIX font for math mode (compatible with Minion Pro)")
        else:
            # For other serif fonts, try to use matching math font
            # Use STIX as a general serif math font fallback
            math_font_set = 'stix'
            print("Using STIX font for math mode")
        
        # Set math font globally
        plt.rcParams['mathtext.fontset'] = math_font_set
        plt.rcParams['mathtext.default'] = 'it'  # Use italic as default
        
        return selected_font, math_font_set
    else:
        print("Using default serif font")
        math_font_set = 'cm'
        plt.rcParams['mathtext.fontset'] = math_font_set
        plt.rcParams['mathtext.default'] = 'it'
        return 'serif', math_font_set

# Set the font
SELECTED_FONT, MATH_FONT_SET = setup_font()

def setup_matplotlib_style():
    """Configure matplotlib with general project-specific styling."""
    plt.rcParams.update({
        'figure.figsize': PLOT_STYLES['figure_size'],
        'figure.dpi': PLOT_STYLES['dpi'],
        'font.size': PLOT_STYLES['font_size'],
        'font.family': SELECTED_FONT,
        'mathtext.fontset': MATH_FONT_SET,
        'mathtext.default': 'it',
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
        'mathtext.fontset': MATH_FONT_SET,
        'mathtext.default': 'it',
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
        'mathtext.fontset': MATH_FONT_SET,
        'mathtext.default': 'it',
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


def _alpha_equivalent_color(color, alpha, background='white'):
    """
    Return a solid color equivalent to drawing `color` with `alpha` on `background`.
    """
    try:
        fg = np.array(mcolors.to_rgb(color), dtype=float)
        bg = np.array(mcolors.to_rgb(background), dtype=float)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        mixed = alpha * fg + (1.0 - alpha) * bg
        return tuple(mixed.tolist())
    except Exception:
        return color


def draw_errorbar_layout(
    ax,
    x,
    y,
    xerr=None,
    yerr=None,
    color='0.45',
    zorder=1,
    use_alpha_equivalent=True,
):
    """
    Draw uncapped error bars with project-default layout while preserving local color.
    """
    if use_alpha_equivalent:
        try:
            error_color = _alpha_equivalent_color(color, ERRORBAR_STYLES['alpha'])
        except NameError:
            # Fallback for notebook/string execution where helper may be out of scope.
            import matplotlib.colors as _mcolors
            fg = np.array(_mcolors.to_rgb(color), dtype=float)
            bg = np.array(_mcolors.to_rgb('white'), dtype=float)
            a = float(np.clip(ERRORBAR_STYLES['alpha'], 0.0, 1.0))
            error_color = tuple((a * fg + (1.0 - a) * bg).tolist())
        error_alpha = 1.0
    else:
        error_color = color
        error_alpha = ERRORBAR_STYLES['alpha']

    ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt='none',
        ecolor=error_color,
        elinewidth=ERRORBAR_STYLES['line_width'],
        capsize=ERRORBAR_STYLES['cap_size'],
        alpha=error_alpha,
        zorder=zorder,
    )

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def distance_force_plot(df_or_path, title="Distance vs Force", figsize=(8, 7.5), dpi=100):
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
        Figure size in inches (default: (8, 7.5))
        
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

#--- just temp function -- 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import layout as lo

def extract_pc_force(value):
    """Return the second element of P_c if available (force component)."""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return value[1]
    return None

def plot_pe_vs_afn_compare(
    parquet_path,
    title="Peak Force Comparison: Estimated P_e vs. Actual P_c",
    alpha=0.8,
    marker_size=2.3,
):
    """
    Scatter comparison of estimated peak load P_e vs. actual signal peak P_c.
    - Colors encode measurement date (layout palette).
    - Circles: P_c (actual from signal)
    - Diamonds: P_e (estimated)
    - AFN labels shown next to P_c.
    - Black outline for negative phi, white outline for non-negative phi.
    
    Parameters:
    -----------
    parquet_path : str
        Path to Parquet data file (e.g., "path/to/M3DC_raw.parquet")
    title : str, optional
        Title for the plot (default: "Peak Force Comparison: Estimated P_e vs. Actual P_c")
    """
    import os
    
    # Load data from Parquet file
    try:
        df = pd.read_parquet(parquet_path, engine='fastparquet')
        print(f"Loaded data from Parquet: {parquet_path}")
    except Exception as e:
        print(f"Error loading Parquet file {parquet_path}: {e}")
        return None
    
    if df.empty:
        print("DataFrame is empty.")
        return None
    
    required = {'AFN', 'P_c', 'date'}
    if not required.issubset(df.columns):
        missing = ', '.join(required - set(df.columns))
        print(f"Missing required columns: {missing}")
        return None
    has_pe = 'P_e' in df.columns
    
    # Check if phi column exists (optional)
    has_phi = 'phi' in df.columns
    
    # Select columns for plotting (include phi if available)
    plot_columns = ['AFN', 'P_c', 'date']
    if has_pe:
        plot_columns.append('P_e')
    if has_phi:
        plot_columns.append('phi')
    
    # Keep rows where actual signal peak can be plotted; P_e is optional.
    plot_data = df[plot_columns].dropna(subset=['AFN', 'P_c', 'date'])
    if plot_data.empty:
        print("No rows with non-null AFN, P_c, and date.")
        return None
    
    plot_data['AFN'] = pd.to_numeric(plot_data['AFN'], errors='coerce')
    plot_data['date'] = pd.to_datetime(plot_data['date'], errors='coerce')
    plot_data['P_c_force'] = plot_data['P_c'].apply(extract_pc_force)
    if has_pe:
        plot_data['P_e'] = pd.to_numeric(plot_data['P_e'], errors='coerce')
    plot_data = plot_data.dropna(subset=['AFN', 'date', 'P_c_force'])
    if plot_data.empty:
        print("No valid P_c force values found.")
        return None
    
    palette = [
        lo.COLORS['red'],
        lo.COLORS['orange'],
        lo.COLORS['gold'],
        lo.COLORS['blue'],
        lo.COLORS['lightblue'],
        lo.COLORS['indigo'],
    ]
    
    unique_dates = plot_data['date'].sort_values().unique()
    color_map = {date: palette[i % len(palette)] for i, date in enumerate(unique_dates)}
    
    setup_visualization_style()
    plt.rcParams['font.family'] = SELECTED_FONT
    plt.rcParams['mathtext.fontset'] = MATH_FONT_SET
    plt.rcParams['mathtext.default'] = 'it'

    fig, ax = plt.subplots(figsize=(8, 7.5), dpi=100)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    for date in unique_dates:
        subset = plot_data[plot_data['date'] == date].copy()
        
        # Handle phi column (optional)
        if has_phi:
            subset['phi_numeric'] = pd.to_numeric(subset['phi'], errors='coerce')
            subset_neg = subset[subset['phi_numeric'] < 0]
            subset_pos = subset[subset['phi_numeric'] >= 0]
            subset_na = subset[subset['phi_numeric'].isna()]
        else:
            # If phi doesn't exist, treat all as non-negative (white outline)
            subset['phi_numeric'] = np.nan
            subset_neg = pd.DataFrame()  # Empty - no negative phi
            subset_pos = subset.copy()    # All points
            subset_na = pd.DataFrame()    # Empty
        
        color = color_map[date]
        date_label = date.strftime('%Y-%m-%d')
        label_added = False
        
        def scatter_points(dataframe, marker, face_alpha, edge_col, label=None):
            if dataframe.empty:
                return False
            ax.scatter(
                dataframe['AFN'],
                dataframe['P_c_force'] if marker == 'o' else dataframe['P_e'],
                color=color,
                marker=marker,
                alpha=face_alpha,
                s=marker_size ** 2 * 10,
                edgecolors=edge_col,
                linewidths=1.0,
                label=label
            )
            return True
        
        if scatter_points(subset_pos, 'o', alpha, 'white', date_label):
            label_added = True
        if scatter_points(subset_neg, 'o', alpha, 'black', None if label_added else date_label):
            label_added = True
        if scatter_points(subset_na, 'o', alpha, 'white', None if label_added else date_label):
            label_added = True
        
        if has_pe:
            subset_pos_pe = subset_pos.dropna(subset=['P_e'])
            subset_neg_pe = subset_neg.dropna(subset=['P_e'])
            subset_na_pe = subset_na.dropna(subset=['P_e'])
            scatter_points(subset_pos_pe, 'D', alpha, 'white')
            scatter_points(subset_neg_pe, 'D', alpha, 'black')
            scatter_points(subset_na_pe, 'D', alpha, 'white')
        
        # Add AFN labels next to points
        for _, row in subset.iterrows():
            ax.text(
                row['AFN'],
                row['P_c_force'],
                f"{int(row['AFN'])}",
                fontsize=8,
                color='black',
                ha='left',
                va='bottom'
            )
    
    ax.set_xlabel('AFN')
    ax.set_ylabel('Force (N)')
    ax.set_title(title)
    ax.tick_params(
        axis='both', which='major',
        labelsize=VISUALIZATION_STYLES['font_size'],
        pad=10, width=1, length=10,
        bottom=True, top=True, left=True, right=True,
        labelcolor='black', color='grey',
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    ax.grid(False)
    
    # Legend for dates (colors)
    date_legend = ax.legend(title='Date', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Legend for marker meaning
    marker_handles = [
        mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=8, label='actual'),
    ]
    if has_pe:
        marker_handles.append(
            mlines.Line2D([], [], color='gray', marker='D', linestyle='None', markersize=8, label='estimate')
        )
    marker_legend = ax.legend(handles=marker_handles, loc='lower right')
    ax.add_artist(date_legend)
    
    plt.tight_layout()
    plt.show()
    
    return None

def plot_temperature_density_vs_date(
    masters_parquet_path,
    title=None,
    figsize=(8, 7.5),
    dpi=100,
    alpha=0.8,
    marker_size=2.3,
    show_legend=True,
    tick_width=1,
    tick_length=10,
    labelpad_x=15,
    labelpad_y=15,
    frame_thickness=1,
    temperature_unit='°C',
    density_unit='kg/m³',
    ylim_temperature=None,
    ylim_density=None,
    box_width=4,
    day_spacing=10,
):
    """
    Plot slab temperature, weak-layer temperature, and slab density vs date.

    Creates a two-panel figure with box plots per experiment date:
      • Top panel:    slab density  ρ_slab  (row-wise mean of rho_1 … rho_4)
      • Bottom panel: slab temperature  T_slab  (mean of T_s1, T_s2) and
                      weak-layer temperature  T_wl  (mean of T_s2, T_s3),
                      shown side by side for each date.

    Date positions along the x-axis are proportional to the actual calendar
    distance between experiment days.

    Parameters
    ----------
    masters_parquet_path : str
        Path to the master Parquet file containing at least the columns
        date, T_s1, T_s2, T_s3, rho_1, rho_2, rho_3, rho_4.
    title : str, optional
        Figure title.  If *None* no title is shown.
    figsize : tuple, default (8, 7.5)
        Figure size in inches.
    dpi : int, default 100
        Figure resolution.
    alpha : float, default 0.8
        Box face transparency.
    marker_size : float, default 2.3
        Outlier marker size.
    show_legend : bool, default True
        Whether to show the legend in the temperature panel.
    tick_width : float, default 1
        Tick mark width.
    tick_length : float, default 10
        Tick mark length in points.
    labelpad_x : float, default 15
        X-axis label padding.
    labelpad_y : float, default 15
        Y-axis label padding.
    frame_thickness : float, default 1
        Spine line width.
    temperature_unit : str, default '°C'
        Unit label for the temperature axis.
    density_unit : str, default 'kg/m³'
        Unit label for the density axis.
    ylim_temperature : tuple, optional
        (ymin, ymax) for the temperature panel.
    ylim_density : tuple, optional
        (ymin, ymax) for the density panel.
    box_width : float, default 4
        Width of each box plot.
    day_spacing : float, default 10
        Position units per calendar day (controls horizontal spread).

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    from matplotlib.ticker import MultipleLocator
    import matplotlib.patches as mpatches

    # --- load data --------------------------------------------------------
    try:
        df = pd.read_parquet(masters_parquet_path, engine='fastparquet')
        print(f"Loaded data from Parquet: {masters_parquet_path}")
    except Exception as e:
        print(f"Error loading Parquet file {masters_parquet_path}: {e}")
        return None

    if df.empty:
        print("No data to visualize")
        return None

    required_columns = ['date', 'T_s1', 'T_s2', 'T_s3',
                        'rho_1', 'rho_2', 'rho_3', 'rho_4']
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        print(f"Error: Required columns not found: {missing}")
        return None

    # --- derived quantities -----------------------------------------------
    df = df.copy()
    df['T_slab']   = df[['T_s1', 'T_s2']].mean(axis=1)
    df['T_wl']     = df[['T_s2', 'T_s3']].mean(axis=1)
    df['rho_slab'] = df[['rho_1', 'rho_2', 'rho_3', 'rho_4']].mean(axis=1)

    # parse dates
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')

    unique_dates = sorted(df['date'].unique())
    if not unique_dates:
        print("Error: No valid dates found")
        return None

    # --- proportional date positions --------------------------------------
    origin = unique_dates[0]
    base_offset = 20
    date_positions = {}
    for d in unique_dates:
        day_delta = (d - origin) / np.timedelta64(1, 'D')
        date_positions[d] = base_offset + day_delta * day_spacing

    # Temperature box offset (T_slab left, T_wl right)
    offset_temp = (day_spacing - box_width) / 2 * 0.8

    # --- style setup ------------------------------------------------------
    setup_visualization_style()
    # Re-apply text/math font here in case other notebook cells changed rcParams.
    plt.rcParams['font.family'] = SELECTED_FONT
    plt.rcParams['mathtext.fontset'] = MATH_FONT_SET
    plt.rcParams['mathtext.default'] = 'it'

    color_rho    = lo.COLORS['indigo']
    color_T_slab = lo.COLORS['orange']
    color_T_wl   = lo.COLORS['blue']

    # --- figure -----------------------------------------------------------
    fig, (ax_density, ax_temp) = plt.subplots(
        2, 1, figsize=figsize, dpi=dpi, sharex=True,
        gridspec_kw={'hspace': 0.15},
        constrained_layout=True,
    )
    fig.patch.set_facecolor('white')
    ax_density.set_facecolor('white')
    ax_temp.set_facecolor('white')

    # --- collect data per date --------------------------------------------
    rho_data, rho_pos = [], []
    ts_data,  ts_pos  = [], []
    tw_data,  tw_pos  = [], []
    tick_pos_list, tick_lbl_list = [], []

    for d in unique_dates:
        pos = date_positions[d]
        sub = df[df['date'] == d]

        rho_vals = sub['rho_slab'].dropna().values
        if len(rho_vals):
            rho_data.append(rho_vals)
            rho_pos.append(pos)

        ts_vals = sub['T_slab'].dropna().values
        if len(ts_vals):
            ts_data.append(ts_vals)
            ts_pos.append(pos - offset_temp)

        tw_vals = sub['T_wl'].dropna().values
        if len(tw_vals):
            tw_data.append(tw_vals)
            tw_pos.append(pos + offset_temp)

        tick_pos_list.append(pos)
        tick_lbl_list.append(pd.Timestamp(d).strftime('%d. %b'))

    # --- draw box plots ---------------------------------------------------
    def _draw_bp(ax, data, positions, color, marker='o'):
        if not data:
            return
        bp = ax.boxplot(
            data, positions=positions, widths=box_width,
            patch_artist=True, zorder=2,
            medianprops=dict(color='black', linewidth=1),
            boxprops=dict(linewidth=0),
            whiskerprops=dict(linewidth=1, color='grey'),
            capprops=dict(linewidth=1, color='grey'),
            flierprops=dict(marker=marker, markersize=marker_size,
                            markeredgewidth=0, alpha=alpha),
        )
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(alpha)
        for flier in bp['fliers']:
            flier.set_markerfacecolor(color)
            flier.set_markeredgewidth(0)
            flier.set_marker(marker)
            flier.set_zorder(1)

    _draw_bp(ax_density, rho_data, rho_pos, color_rho)
    _draw_bp(ax_temp,    ts_data,  ts_pos,  color_T_slab, marker='o')
    _draw_bp(ax_temp,    tw_data,  tw_pos,  color_T_wl,   marker='s')

    # --- labels -----------------------------------------------------------
    ax_temp.set_xlabel(
        'Date', fontsize=VISUALIZATION_STYLES['font_size'],
        labelpad=labelpad_x, color='black')
    ax_density.set_ylabel(
        f'Density $\\rho_{{\\mathrm{{slab}}}}$ ({density_unit})',
        fontsize=VISUALIZATION_STYLES['font_size'],
        labelpad=labelpad_y, color='black')
    ax_temp.set_ylabel(
        f'Temperature ({temperature_unit})',
        fontsize=VISUALIZATION_STYLES['font_size'],
        labelpad=labelpad_y, color='black')

    if title is not None:
        fig.suptitle(title, fontsize=VISUALIZATION_STYLES['font_size'], y=0.995)

    # --- x-axis -----------------------------------------------------------
    if tick_pos_list:
        ax_temp.set_xticks(tick_pos_list)
        ax_temp.set_xticklabels(tick_lbl_list, rotation=45, ha='right')
        margin = 15
        ax_density.set_xlim(min(tick_pos_list) - margin,
                            max(tick_pos_list) + margin)
        ax_temp.set_xlim(min(tick_pos_list) - margin,
                         max(tick_pos_list) + margin)

    # --- y-axis limits & locators -----------------------------------------
    if ylim_density is not None:
        ax_density.set_ylim(ylim_density)
    if ylim_temperature is not None:
        ax_temp.set_ylim(ylim_temperature)

    ax_temp.yaxis.set_major_locator(MultipleLocator(2))
    ax_density.yaxis.set_major_locator(MultipleLocator(50))

    # --- tick & spine styling ---------------------------------------------
    for ax in (ax_density, ax_temp):
        ax.tick_params(
            axis='both', which='major',
            labelsize=VISUALIZATION_STYLES['font_size'],
            pad=10, width=tick_width, length=tick_length,
            bottom=True, top=True, left=True, right=True,
            labelcolor='black', color='grey',
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(frame_thickness)
        ax.grid(False)

    # --- legend -----------------------------------------------------------
    if show_legend:
        handles = [
            mpatches.Patch(facecolor=color_T_slab, edgecolor='none',
                           alpha=alpha, label='$T_{\\mathrm{slab}}$'),
            mpatches.Patch(facecolor=color_T_wl, edgecolor='none',
                           alpha=alpha, label='$T_{\\mathrm{wl}}$'),
        ]
        ax_temp.legend(
            handles=handles, loc='lower left',
            fontsize=VISUALIZATION_STYLES['font_size'],
            frameon=False)

    fig.set_size_inches(figsize)
    return fig


def plot_column_over_afn(parquet_path, title=None):
    """
    Interactive plot of any column value over AFN with dropdown selection.
    - Colors encode measurement date (layout palette).
    - No outlines on scatter points.
    - AFN labels shown next to points.
    - Excludes AFNs below 100.
    - Dropdown to select which column to plot (y-axis).
    
    Parameters:
    -----------
    parquet_path : str
        Path to Parquet data file (e.g., "path/to/M3DC_raw.parquet")
    title : str, optional
        Title for the plot (default: uses column name)
    """
    import os
    
    # Load data from Parquet file
    try:
        df = pd.read_parquet(parquet_path, engine='fastparquet')
        print(f"Loaded data from Parquet: {parquet_path}")
    except Exception as e:
        print(f"Error loading Parquet file {parquet_path}: {e}")
        return None
    
    if df.empty:
        print("DataFrame is empty.")
        return None
    
    # Auto-detect info parquet path
    info_path = parquet_path.replace('.parquet', '_info.parquet')
    df_info = None
    if os.path.exists(info_path):
        try:
            df_info = pd.read_parquet(info_path, engine='fastparquet')
            print(f"Loaded info from Parquet: {info_path}")
        except Exception as e:
            print(f"Warning: Could not load info file {info_path}: {e}")
    
    # Check required columns
    if 'AFN' not in df.columns or 'date' not in df.columns:
        print("Error: Required columns 'AFN' and 'date' not found in data")
        return None
    
    # Get numeric columns (exclude AFN, date, and list-type columns)
    exclude_cols = {'AFN', 'date', 'w_data', 'P_data', 'sample_geometry_raw', 'sample_geometry_FEM'}
    numeric_cols = [col for col in df.columns 
                   if col not in exclude_cols 
                   and df[col].dtype in ['int64', 'float64', 'Int64', 'Float64']]
    
    if not numeric_cols:
        print("Error: No numeric columns found to plot")
        return None
    
    # Create column dropdown
    column_dropdown = Dropdown(
        options=sorted(numeric_cols),
        value=numeric_cols[0],
        description='Column:',
        continuous_update=False
    )
    
    # Plotting function that will be called when column changes
    def plot_column(column_name):
        """Plot column value over AFN for selected column."""
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found in data")
            return
        
        # Prepare data
        plot_data = df[['AFN', 'date', column_name]].dropna(subset=['AFN', 'date', column_name])
        if plot_data.empty:
            print(f"No rows with non-null AFN, date, and {column_name}.")
            return
        
        # Filter out AFNs below 100
        plot_data = plot_data[plot_data['AFN'] >= 100].copy()
        if plot_data.empty:
            print(f"No AFNs >= 100 found with valid {column_name} values.")
            return
        
        plot_data['date'] = pd.to_datetime(plot_data['date'], errors='coerce')
        plot_data = plot_data.dropna(subset=['date', column_name])
        if plot_data.empty:
            print(f"No valid {column_name} values found.")
            return
        
        palette = [
            lo.COLORS['red'],
            lo.COLORS['orange'],
            lo.COLORS['gold'],
            lo.COLORS['blue'],
            lo.COLORS['lightblue'],
            lo.COLORS['indigo'],
        ]
        
        unique_dates = plot_data['date'].sort_values().unique()
        color_map = {date: palette[i % len(palette)] for i, date in enumerate(unique_dates)}
        
        setup_visualization_style()
        plt.rcParams['font.family'] = SELECTED_FONT
        plt.rcParams['mathtext.fontset'] = MATH_FONT_SET
        plt.rcParams['mathtext.default'] = 'it'

        fig, ax = plt.subplots(figsize=(8, 7.5), dpi=100)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        for date in unique_dates:
            subset = plot_data[plot_data['date'] == date].copy()
            
            color = color_map[date]
            date_label = date.strftime('%Y-%m-%d')
            
            # Plot points without any outlines
            ax.scatter(
                subset['AFN'],
                subset[column_name],
                c=color,
                marker='o',
                alpha=0.8,
                edgecolors='none',  # No outlines - hardcoded
                s=2.3 ** 2 * 10,
                label=date_label
            )
            
            # Add AFN labels next to points
            for _, row in subset.iterrows():
                ax.text(
                    row['AFN'],
                    row[column_name],
                    f" {int(row['AFN'])}",
                    fontsize=8,
                    color='black',
                    ha='left',
                    va='center'
                )
        
        # Extract unit from info DataFrame if available
        ylabel = column_name
        if df_info is not None:
            try:
                unit_row = df_info[df_info['Abreviation'] == column_name]
                if not unit_row.empty:
                    unit = unit_row.iloc[0]['Units']
                    if pd.notna(unit) and str(unit).strip() != '' and str(unit).lower() != 'none':
                        ylabel = f'{column_name} ({unit})'
            except Exception:
                pass
        
        plot_title = title 
        
        ax.set_xlabel('AFN')
        ax.set_ylabel(ylabel)
        ax.set_title(plot_title)
        ax.tick_params(
            axis='both', which='major',
            labelsize=VISUALIZATION_STYLES['font_size'],
            pad=10, width=1, length=10,
            bottom=True, top=True, left=True, right=True,
            labelcolor='black', color='grey',
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        ax.grid(False)
        
        # Legend for dates (colors)
        ax.legend(title='Date', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    # Connect dropdown to plotting function
    out = interactive_output(plot_column, {'column_name': column_dropdown})
    
    # Display UI
    controls = VBox([column_dropdown])
    display(out, controls)
    
    return None


def plot_mode_I_III_interaction(
    df_or_path,
    x_component='G1',
    y_component='G3',
    depict_AFN=False,
    mark_infinite=False,
    alpha=0.8,
    marker_size=2.3,
    tick_width=1,
    tick_length=10,
    labelpad_x=15,
    labelpad_y=15,
    frame_thickness=1,
    x_lim=None,
    y_lim=None,
    title=None,
    figsize=(8, 7.5),
    dpi=100,
    opt_param=None,
    fit_resolution=1000,
    show_plot=True,
    return_optimization=False,
):
    """
    Quick scatter plot for G-components with uncertainty whiskers (no end caps).

    Parameters
    ----------
    df_or_path : pd.DataFrame or str
        DataFrame with G-columns, or path to parquet file.
    x_component : str, default 'G1'
        One of 'G1', 'G2', 'G3' used for x-axis.
    y_component : str, default 'G2'
        One of 'G1', 'G2', 'G3' used for y-axis.
    depict_AFN : bool, default False
        If True, write AFN labels next to each datapoint (requires 'AFN' column).
    mark_infinite : bool, default False
        If True, points with invalid `L` (None/NaN or non-numeric type) are
        highlighted with a red outline.
    alpha : float, default 0.8
        Marker and errorbar transparency.
    marker_size : float, default 2.3
        Marker size scale.
    tick_width : float, default 1
        Width of tick marks.
    tick_length : float, default 10
        Length of tick marks in points.
    labelpad_x : float, default 15
        Padding (spacing) between x-axis label and tick labels.
    labelpad_y : float, default 15
        Padding (spacing) between y-axis label and tick labels.
    frame_thickness : float, default 1
        Thickness (linewidth) of the plot frame (spines).
    x_lim : tuple, optional
        x-axis limits.
    y_lim : tuple, optional
        y-axis limits.
    title : str, optional
        Plot title.
    figsize : tuple, default (8, 7.5)
        Figure size.
    dpi : int, default 120
        Figure DPI.
    opt_param : dict, optional
        Precomputed optimization results (e.g. from
        data_config.optimize_mode_I_III_interaction). If provided and plotting
        G1 vs G3, overlays the red geometric fit curve from this dict.
    fit_resolution : int, default 1000
        Number of x-samples for plotting fitted interaction curves.
    show_plot : bool, default True
        If False, skip rendering the figure (useful for optimization-only calls).
    return_optimization : bool, default False
        If True, return optimization values/stats as a dictionary.
    """
    component_map = {
        'G1': 'G1c',
        'G2': 'G2c',
        'G3': 'G3c',
    }

    x_key = str(x_component).upper().strip()
    y_key = str(y_component).upper().strip()
    if x_key not in component_map or y_key not in component_map:
        print("Error: x_component and y_component must be one of: 'G1', 'G2', 'G3'.")
        return None

    x_col = component_map[x_key]
    y_col = component_map[y_key]
    x_err_col = f"{x_col}_uncertainty"
    y_err_col = f"{y_col}_uncertainty"

    if isinstance(df_or_path, pd.DataFrame):
        df = df_or_path.copy()
    else:
        try:
            df = pd.read_parquet(df_or_path, engine='fastparquet')
            print(f"Loaded data from Parquet: {df_or_path}")
        except Exception as e:
            print(f"Error loading parquet: {e}")
            return None

    required_cols = [x_col, y_col, x_err_col, y_err_col]
    # Avoid duplicate requests.
    required_cols = list(dict.fromkeys(required_cols))
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return None

    # Build a clean 1D numeric table even if source DataFrame has duplicate column names.
    plot_df = pd.DataFrame(index=df.index)
    for col in required_cols:
        raw_col = df[col]
        if isinstance(raw_col, pd.DataFrame):
            # If duplicate column names exist, use the first occurrence.
            raw_col = raw_col.iloc[:, 0]
        plot_df[col] = pd.to_numeric(raw_col, errors='coerce')
    if 'AFN' in df.columns:
        raw_afn_col = df['AFN']
        if isinstance(raw_afn_col, pd.DataFrame):
            raw_afn_col = raw_afn_col.iloc[:, 0]
        plot_df['AFN_num'] = pd.to_numeric(raw_afn_col, errors='coerce')
    if depict_AFN and 'AFN' in df.columns:
        raw_afn_col = df['AFN']
        if isinstance(raw_afn_col, pd.DataFrame):
            raw_afn_col = raw_afn_col.iloc[:, 0]
        plot_df['AFN'] = raw_afn_col.astype(str)
    elif depict_AFN:
        print("Warning: depict_AFN=True but 'AFN' column was not found. Labels are skipped.")
    if mark_infinite and 'L' in df.columns:
        raw_l_col = df['L']
        if isinstance(raw_l_col, pd.DataFrame):
            raw_l_col = raw_l_col.iloc[:, 0]

        def _is_valid_numeric_value(v):
            if pd.isna(v) or isinstance(v, bool):
                return False
            return isinstance(v, (int, float, np.integer, np.floating))

        plot_df['L_invalid'] = ~raw_l_col.apply(_is_valid_numeric_value)
    elif mark_infinite:
        print("Warning: mark_infinite=True but 'L' column was not found. Red outlines are skipped.")
    plot_df = plot_df.dropna(subset=[x_col, y_col])
    if plot_df.empty:
        print("No valid rows to plot.")
        return None

    # Missing uncertainty values are interpreted as zero whisker length.
    plot_df[x_err_col] = plot_df[x_err_col].fillna(0.0).clip(lower=0.0)
    plot_df[y_err_col] = plot_df[y_err_col].fillna(0.0).clip(lower=0.0)

    setup_visualization_style()
    plt.rcParams['font.family'] = SELECTED_FONT
    plt.rcParams['mathtext.fontset'] = MATH_FONT_SET
    plt.rcParams['mathtext.default'] = 'it'

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    def _get_alpha_equiv_color_from_base(base_color, alpha_val):
        # Match the behavior from the ERR polar fit example:
        # use precomputed alpha-equivalent colors when available.
        color_name = None
        for name, value in lo.COLORS.items():
            if str(value).lower() == str(base_color).lower():
                color_name = name
                break
        if (
            color_name is not None
            and hasattr(lo, 'COLORS_ALPHA_EQUIV')
            and color_name in lo.COLORS_ALPHA_EQUIV
            and alpha_val in lo.COLORS_ALPHA_EQUIV[color_name]
        ):
            return lo.COLORS_ALPHA_EQUIV[color_name][alpha_val]
        return _alpha_equivalent_color(base_color, alpha_val)

    base_color = lo.COLORS['blue']
    errorbar_color = _get_alpha_equiv_color_from_base(base_color, 0.5)

    # Plot error bars first (background), same style as requested example.
    x_vals = plot_df[x_col].to_numpy(dtype=float)
    y_vals = plot_df[y_col].to_numpy(dtype=float)
    x_err_vals = plot_df[x_err_col].to_numpy(dtype=float)
    y_err_vals = plot_df[y_err_col].to_numpy(dtype=float)
    for i in range(len(x_vals)):
        if np.isfinite(x_err_vals[i]) and x_err_vals[i] > 0:
            ax.plot(
                [x_vals[i] - x_err_vals[i], x_vals[i] + x_err_vals[i]],
                [y_vals[i], y_vals[i]],
                color=errorbar_color,
                alpha=1.0,
                linewidth=ERRORBAR_STYLES['line_width'],
                zorder=0,
            )
        if np.isfinite(y_err_vals[i]) and y_err_vals[i] > 0:
            ax.plot(
                [x_vals[i], x_vals[i]],
                [y_vals[i] - y_err_vals[i], y_vals[i] + y_err_vals[i]],
                color=errorbar_color,
                alpha=1.0,
                linewidth=ERRORBAR_STYLES['line_width'],
                zorder=0,
            )

    # Plot markers after error bars with alpha-equivalent marker color.
    ax.scatter(
        x_vals,
        y_vals,
        c=base_color,
        s=marker_size ** 2 * 10,
        alpha=0.8,
        edgecolors='none',
        zorder=1,
    )

    if mark_infinite and 'L_invalid' in plot_df.columns:
        invalid_mask = plot_df['L_invalid'].fillna(False)
        if invalid_mask.any():
            ax.scatter(
                plot_df.loc[invalid_mask, x_col],
                plot_df.loc[invalid_mask, y_col],
                facecolors='none',
                edgecolors='red',
                linewidths=1.5,
                s=(marker_size ** 2 * 10) * 1.15,
                alpha=alpha,
                zorder=3,
            )

    ax.set_xlabel(
        r'Mode I energy release rate $\mathcal{G}_{\mathrm{I}}$ (J/m$^2$) $\longrightarrow$',
        fontsize=VISUALIZATION_STYLES['font_size'],
        labelpad=labelpad_x,
        color='black',
    )
    ax.set_ylabel(
        r'Mode III energy release rate $\mathcal{G}_{\mathrm{III}}$ (J/m$^2$) $\longrightarrow$',
        fontsize=VISUALIZATION_STYLES['font_size'],
        labelpad=labelpad_y,
        color='black',
    )

    optimization_result = None
    if x_col == 'G1c' and y_col == 'G3c':
        if isinstance(opt_param, dict):
            optimization_result = opt_param
            n_samples = max(int(fit_resolution), 50)

            geo = optimization_result.get('geo_fit', {})
            n_geo = float(geo.get('n', np.nan))
            m_geo = float(geo.get('m', np.nan))
            giiic_geo = float(geo.get('giiic', np.nan))
            gic_ref = float(optimization_result.get('gic_ref', np.nan))
            if np.isfinite(n_geo) and n_geo > 0 and np.isfinite(m_geo) and m_geo > 0 and np.isfinite(giiic_geo) and giiic_geo > 0 and np.isfinite(gic_ref) and gic_ref > 0:
                x_geo = np.linspace(1e-12, gic_ref, n_samples)
                with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
                    term_geo = 1.0 - (x_geo / gic_ref) ** n_geo
                    y_geo = np.full_like(x_geo, np.nan, dtype=float)
                    msk = term_geo >= 0.0
                    y_geo[msk] = giiic_geo * (term_geo[msk] ** (1.0 / max(m_geo, 1e-12)))
                vgeo = np.isfinite(x_geo) & np.isfinite(y_geo) & (y_geo >= 0.0)
                if np.any(vgeo):
                    ax.plot(
                        x_geo[vgeo],
                        y_geo[vgeo],
                        color=lo.COLORS.get('red', '#d71149'),
                        linewidth=2.0,
                        alpha=0.95,
                        linestyle='-',
                        zorder=4,
                    )
        else:
            print("[plot_mode_I_III_interaction] No opt_param provided; plotting scatter only (no fit overlays).")
    else:
        print("[plot_mode_I_III_interaction] Fit overlay skipped (requires x_component='G1' and y_component='G3').")

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if title is not None:
        ax.set_title(title, fontsize=VISUALIZATION_STYLES['font_size'], pad=20)

    if depict_AFN and 'AFN' in plot_df.columns:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_off = max((x_max - x_min) * 0.008, 1e-9)
        y_off = max((y_max - y_min) * 0.008, 1e-9)
        for _, row in plot_df.iterrows():
            ax.text(
                row[x_col] + x_off,
                row[y_col] + y_off,
                row['AFN'],
                fontsize=max(VISUALIZATION_STYLES['font_size'] - 4, 8),
                color='black',
                alpha=0.9,
                zorder=3,
            )

    # Keep identical data scaling on both axes so geometric distances are comparable.
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(
        axis='both', which='major',
        labelsize=VISUALIZATION_STYLES['font_size'],
        pad=10,
        width=tick_width,
        length=tick_length,
        color=TICK_COLOR,
        bottom=True, top=True, labeltop=False,
        left=True, right=True, labelright=False,
        labelcolor='black',
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(frame_thickness)
    ax.grid(False)
    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    if return_optimization:
        return optimization_result
    return None


def plot_GIII_ratio_vs_G(
    df_or_path,
    x_lim=(0,np.pi/2),
    title=None,
    figsize=(8, 7.5),
    dpi=100,
    show_legend=False,
    alpha=PLOTPOINT_ALPHA,
    marker_size=2.3,
    tick_width=TICK_WIDTH,
    tick_length=TICK_LENGTH,
    labelpad_x=15,
    labelpad_y=15,
    frame_thickness=FRAME_THICKNESS,
    Gc_lim=0.8,
    free_fit_variant='C',
    n_starts=49,
    y_lim=None,
):
    """
    Plot elliptic radial coordinate over elliptic mode mixity angle:
      x = phi^G = arctan((GIII*GIc)/(GI*GIIIc))
      y = r^G

    Uses:
    - GIII = G3c
    - GI + GIII = G1c + G3c

    Parameters
    ----------
    alpha : float, default 0.8
        Marker transparency.
    marker_size : float, default 2.3
        Marker size scale.
    tick_width : float, default 1
        Width (thickness) of the tick marks.
    tick_length : float, default 10
        Length of the tick marks in points.
    labelpad_x : float, default 15
        Padding (spacing) between x-axis label and tick labels.
    labelpad_y : float, default 15
        Padding (spacing) between y-axis label and tick labels.
    frame_thickness : float, default 1
        Thickness (linewidth) of the plot frame (spines).
    Gc_lim : float, default 0.9
        Threshold to compute fixed critical values from data means:
        - GIIIc = mean(G3c) for rows with G3c/(G1c+G3c) >= Gc_lim
        - GIc   = mean(G1c) for rows with G1c/(G1c+G3c) >= Gc_lim
    free_fit_variant : {'B', 'C'}, default 'B'
        Residual variant for the free (red) n,m fit:
        - 'B': (GI/GIc)^n + (GIII/GIIIc)^m - 1
        - 'C': (GI/GIc)^(1/n) + (GIII/GIIIc)^(1/m) - 1
    n_starts : int, default 49
        Number of initial guesses for multi-start local optimization of n,m.
    y_lim : tuple, optional
        y-axis limits as (ymin, ymax). If None, automatic limits are used.
    """
    if isinstance(df_or_path, pd.DataFrame):
        df = df_or_path.copy()
    else:
        try:
            df = pd.read_parquet(df_or_path, engine='fastparquet')
            print(f"Loaded data from Parquet: {df_or_path}")
        except Exception as e:
            print(f"Error loading parquet: {e}")
            return None

    required_cols = ['G1c', 'G3c']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return None

    plot_df = df.copy()
    plot_df['G1c'] = pd.to_numeric(plot_df['G1c'], errors='coerce')
    plot_df['G3c'] = pd.to_numeric(plot_df['G3c'], errors='coerce')
    g1_unc_col = 'G1c_uncertainty' if 'G1c_uncertainty' in plot_df.columns else None
    g3_unc_col = 'G3c_uncertainty' if 'G3c_uncertainty' in plot_df.columns else None
    if g1_unc_col is not None:
        plot_df['G1c_unc'] = pd.to_numeric(plot_df[g1_unc_col], errors='coerce').fillna(0.0).clip(lower=0.0)
    else:
        plot_df['G1c_unc'] = 0.0
    if g3_unc_col is not None:
        plot_df['G3c_unc'] = pd.to_numeric(plot_df[g3_unc_col], errors='coerce').fillna(0.0).clip(lower=0.0)
    else:
        plot_df['G3c_unc'] = 0.0

    plot_df['G13_total'] = plot_df['G1c'] + plot_df['G3c']
    plot_df = plot_df.dropna(subset=['G1c', 'G3c', 'G13_total'])
    plot_df = plot_df[plot_df['G13_total'] > 0]
    if plot_df.empty:
        print("No valid rows to plot (requires G1+G3 > 0).")
        return None

    # Fixed GIc/GIIIc from Gc_lim subsets (uncertainty-weighted means),
    # with robust full-data fallback.
    ratio_iii = plot_df['G3c'] / plot_df['G13_total']
    ratio_i = plot_df['G1c'] / plot_df['G13_total']
    ratio_iii_mask = ratio_iii >= Gc_lim
    ratio_i_mask = ratio_i >= Gc_lim

    def _weighted_mean_with_unc(x_series, sx_series, min_sigma=1e-6):
        x = pd.to_numeric(x_series, errors='coerce').to_numpy(dtype=float)
        sx = pd.to_numeric(sx_series, errors='coerce').to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(sx) & (sx >= 0.0)
        if not np.any(mask):
            return np.nan, np.nan
        x = x[mask]
        sx = np.clip(sx[mask], min_sigma, None)
        w = 1.0 / (sx ** 2)
        mu = float(np.sum(w * x) / np.maximum(np.sum(w), 1e-12))
        mu_unc = float(np.sqrt(1.0 / np.maximum(np.sum(w), 1e-12)))
        return mu, mu_unc

    gi_subset = plot_df.loc[ratio_i_mask] if ratio_i_mask.any() else plot_df
    giii_subset = plot_df.loc[ratio_iii_mask] if ratio_iii_mask.any() else plot_df

    gic_ref, gic_unc_weighted = _weighted_mean_with_unc(gi_subset['G1c'], gi_subset['G1c_unc'])
    giiic_ref, giiic_unc_weighted = _weighted_mean_with_unc(giii_subset['G3c'], giii_subset['G3c_unc'])

    if not np.isfinite(gic_ref) or gic_ref <= 0:
        gic_ref = 1.0
    if not np.isfinite(gic_unc_weighted):
        gic_unc_weighted = 0.0
    if not np.isfinite(giiic_ref) or giiic_ref <= 0:
        giiic_ref = 1.0
    if not np.isfinite(giiic_unc_weighted):
        giiic_unc_weighted = 0.0

    # Prepare data and weights for exponent-law diagnostics.
    g1_vals = plot_df['G1c'].to_numpy(dtype=float)
    g3_vals = plot_df['G3c'].to_numpy(dtype=float)
    sg1_vals = plot_df['G1c_unc'].to_numpy(dtype=float)
    sg3_vals = plot_df['G3c_unc'].to_numpy(dtype=float)
    sigma_weight = np.sqrt(sg1_vals ** 2 + sg3_vals ** 2)
    sigma_eff = np.maximum(sigma_weight, 1e-6)
    weights_raw = 1.0 / (sigma_eff ** 2)
    weights = weights_raw.copy()
    weights = weights / np.maximum(np.sum(weights), 1e-12)

    def _a_value(n_val, m_val):
        u = np.clip(g1_vals / max(gic_ref, 1e-12), 0.0, None)
        v = np.clip(g3_vals / max(giiic_ref, 1e-12), 0.0, None)
        return np.power(u, n_val) + np.power(v, m_val)

    def _c_value(n_val, m_val):
        u = np.clip(g1_vals / max(gic_ref, 1e-12), 1e-12, None)
        v = np.clip(g3_vals / max(giiic_ref, 1e-12), 1e-12, None)
        return np.power(u, 1.0 / max(n_val, 1e-12)) + np.power(v, 1.0 / max(m_val, 1e-12))

    def _r_value():
        # Elliptic radial coordinate used for plotting:
        # r^G = sqrt((GI/GIc)^2 + (GIII/GIIIc)^2)
        u = np.clip(g1_vals / max(gic_ref, 1e-12), 0.0, None)
        v = np.clip(g3_vals / max(giiic_ref, 1e-12), 0.0, None)
        return np.sqrt(u ** 2 + v ** 2)

    def _r_opt_from_phi(phi_val, n_val, m_val, variant='B'):
        c = max(np.cos(phi_val), 1e-12)
        s = max(np.sin(phi_val), 1e-12)

        def f(rv):
            if variant == 'C':
                return (rv * c) ** (1.0 / max(n_val, 1e-12)) + (rv * s) ** (1.0 / max(m_val, 1e-12)) - 1.0
            return (rv * c) ** n_val + (rv * s) ** m_val - 1.0

        lo_r, hi_r = 0.0, 1.0
        while f(hi_r) < 0.0 and hi_r < 1e6:
            hi_r *= 2.0
        for _ in range(60):
            mid = 0.5 * (lo_r + hi_r)
            if f(mid) < 0.0:
                lo_r = mid
            else:
                hi_r = mid
        return 0.5 * (lo_r + hi_r)

    # Base coordinates and propagated uncertainties in (phi, r) space.
    g1_safe = np.where(np.abs(g1_vals) > 1e-12, g1_vals, 1e-12)
    ratio_safe = (g3_vals * gic_ref) / (g1_safe * giiic_ref)
    phi_vals_base = np.arctan(ratio_safe)
    phi_vals_base = np.where(g1_vals == 0, np.pi / 2, phi_vals_base)
    r_quad_vals_base = np.sqrt((g1_vals / gic_ref) ** 2 + (g3_vals / giiic_ref) ** 2)

    denom = 1.0 + ratio_safe ** 2
    dx_dg1 = -(g3_vals * gic_ref) / ((g1_safe ** 2) * giiic_ref * denom)
    dx_dg3 = gic_ref / (g1_safe * giiic_ref * denom)
    x_unc_base = np.sqrt((dx_dg1 * sg1_vals) ** 2 + (dx_dg3 * sg3_vals) ** 2)

    r_safe = np.where(r_quad_vals_base > 1e-12, r_quad_vals_base, 1e-12)
    u_np = np.clip(g1_vals / gic_ref, 0.0, None)
    v_np = np.clip(g3_vals / giiic_ref, 0.0, None)
    dr_du = u_np / r_safe
    dr_dv = v_np / r_safe
    dr_dg1 = dr_du * (1.0 / gic_ref)
    dr_dg3 = dr_dv * (1.0 / giiic_ref)
    y_unc_base = np.sqrt((dr_dg1 * sg1_vals) ** 2 + (dr_dg3 * sg3_vals) ** 2)

    x_unc_base = np.clip(np.nan_to_num(x_unc_base, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    y_unc_base = np.clip(np.nan_to_num(y_unc_base, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)

    def _objective_nm(n_val, m_val):
        if n_val < 1e-3 or n_val > 10.0 or m_val < 1e-3 or m_val > 10.0:
            return np.inf
        r_model = np.array(
            [_r_opt_from_phi(ph, n_val=n_val, m_val=m_val, variant=free_fit_variant) for ph in phi_vals_base],
            dtype=float
        )
        eps = 1e-4
        phi_p = np.clip(phi_vals_base + eps, 0.0, np.pi / 2)
        phi_m = np.clip(phi_vals_base - eps, 0.0, np.pi / 2)
        r_model_p = np.array(
            [_r_opt_from_phi(ph, n_val=n_val, m_val=m_val, variant=free_fit_variant) for ph in phi_p],
            dtype=float
        )
        r_model_m = np.array(
            [_r_opt_from_phi(ph, n_val=n_val, m_val=m_val, variant=free_fit_variant) for ph in phi_m],
            dtype=float
        )
        dphi = np.maximum(phi_p - phi_m, 1e-9)
        dr_dphi = (r_model_p - r_model_m) / dphi
        sigma_eff2 = y_unc_base ** 2 + (dr_dphi * x_unc_base) ** 2
        sigma_eff2 = np.maximum(sigma_eff2, 1e-10)
        residual = r_quad_vals_base - r_model
        return float(np.sum((residual ** 2) / sigma_eff2))

    def _calc_case_stats(
        n_val,
        m_val,
        case_name,
        method='fixed_exponent',
        objective_override=None,
        variant='B',
        n_params=0,
    ):
        n_val = float(n_val)
        m_val = float(m_val)
        objective_value = _objective_nm(n_val, m_val)
        if objective_override is not None and np.isfinite(objective_override):
            objective_value = float(objective_override)
        try:
            if variant == 'C':
                expr_vals = _c_value(n_val=n_val, m_val=m_val)
            else:
                expr_vals = _a_value(n_val=n_val, m_val=m_val)
        except Exception:
            expr_vals = np.full_like(g1_vals, np.nan, dtype=float)
        a_residual_vals = expr_vals - 1.0
        weighted_rmse_a = float(np.sqrt(np.sum(weights * a_residual_vals ** 2)))
        chi2 = float(np.sum(weights_raw * a_residual_vals ** 2))
        dof = max(len(a_residual_vals) - int(n_params), 1)
        reduced_chi2 = float(chi2 / dof)
        phi_vals_local = plot_df['phi_elliptic'].to_numpy(dtype=float)
        r_quad_vals_local = plot_df['r_quadratic'].to_numpy(dtype=float)
        r_model_vals = np.array(
            [_r_opt_from_phi(ph, n_val=n_val, m_val=m_val, variant=variant) for ph in phi_vals_local],
            dtype=float
        )
        weighted_rmse_r = float(np.sqrt(np.sum(weights * (r_quad_vals_local - r_model_vals) ** 2)))
        return {
            'name': case_name,
            'n': n_val,
            'm': m_val,
            'objective': float(objective_value),
            'method': method,
            'rmse_a': float(weighted_rmse_a),
            'rmse_r': float(weighted_rmse_r),
            'chi2': chi2,
            'reduced_chi2': reduced_chi2,
        }

    # Elliptic mode mixity angle:
    # phi^G = arctan((GIII * GIc) / (GI * GIIIc))
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_phi = (plot_df['G3c'] * gic_ref) / (plot_df['G1c'] * giiic_ref)
        plot_df['phi_elliptic'] = np.arctan(ratio_phi)
        plot_df['phi_elliptic'] = np.where(plot_df['G1c'] == 0, np.pi / 2, plot_df['phi_elliptic'])

    # Keep radial coordinate for diagnostics/plotting.
    plot_df['r_elliptic'] = _r_value()
    # Plot y-values in quadratic radial space:
    # r^G = sqrt((GI/GIc)^2 + (GIII/GIIIc)^2)
    plot_df['r_quadratic'] = np.sqrt((plot_df['G1c'] / gic_ref) ** 2 + (plot_df['G3c'] / giiic_ref) ** 2)

    # Fit n and m independently with GIc/GIIIc fixed from Gc_lim means.
    n_opt = 2.0
    m_opt = 2.0
    nm_opt_method = 'initial'
    nm_opt_objective = np.nan
    nm_starts_total = 0
    nm_starts_success = 0
    nm_boundary_hits = 0
    nm_top = []
    fit_variant = str(free_fit_variant).upper().strip()
    if fit_variant not in {'B', 'C'}:
        fit_variant = 'B'
    free_fit_variant = fit_variant
    try:
        from scipy.optimize import minimize
        seed_axis = np.array([1e-3, 1e-2, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0], dtype=float)
        seed_pairs = [(float(a), float(b)) for a in seed_axis for b in seed_axis]
        if int(n_starts) > len(seed_pairs):
            rng = np.random.default_rng(42)
            extra = int(n_starts) - len(seed_pairs)
            random_pairs = list(zip(rng.uniform(1e-3, 10.0, extra), rng.uniform(1e-3, 10.0, extra)))
            seed_pairs.extend([(float(a), float(b)) for a, b in random_pairs])
        elif int(n_starts) < len(seed_pairs):
            seed_pairs = seed_pairs[:max(int(n_starts), 1)]

        runs = []
        for n0, m0 in seed_pairs:
            nm_starts_total += 1
            res_nm = minimize(
                lambda x: _objective_nm(float(x[0]), float(x[1])),
                x0=np.array([n0, m0], dtype=float),
                method='L-BFGS-B',
                bounds=[(1e-3, 10.0), (1e-3, 10.0)],
            )
            if res_nm.success and np.all(np.isfinite(res_nm.x)) and np.isfinite(res_nm.fun):
                nm_starts_success += 1
                n_hat = float(res_nm.x[0])
                m_hat = float(res_nm.x[1])
                if (
                    abs(n_hat - 1e-3) < 1e-4 or abs(n_hat - 10.0) < 1e-4
                    or abs(m_hat - 1e-3) < 1e-4 or abs(m_hat - 10.0) < 1e-4
                ):
                    nm_boundary_hits += 1
                runs.append((float(res_nm.fun), n_hat, m_hat))

        if runs:
            runs.sort(key=lambda t: t[0])
            nm_top = runs[:5]
            nm_opt_objective, n_opt, m_opt = runs[0]
            nm_opt_method = 'multistart:scipy:L-BFGS-B'
        else:
            nm_opt_method = 'scipy_failed_fallback_grid'
    except Exception:
        nm_opt_method = 'scipy_unavailable_fallback_grid'

    if not np.isfinite(nm_opt_objective):
        n_grid = np.linspace(1e-3, 10.0, 121)
        m_grid = np.linspace(1e-3, 10.0, 121)
        best_obj = np.inf
        best_n = n_opt
        best_m = m_opt
        for n_try in n_grid:
            for m_try in m_grid:
                obj = _objective_nm(float(n_try), float(m_try))
                if obj < best_obj:
                    best_obj = obj
                    best_n = float(n_try)
                    best_m = float(m_try)
        n_opt = best_n
        m_opt = best_m
        nm_opt_objective = float(best_obj)
        nm_opt_method = 'grid_search'
        nm_starts_total = max(nm_starts_total, len(n_grid) * len(m_grid))
        nm_starts_success = max(nm_starts_success, 1)

    # Structured feedback with uncertainty estimates from the Gc_lim subsets.
    gi_seed = plot_df.loc[ratio_i_mask, 'G1c'] if ratio_i_mask.any() else plot_df['G1c']
    giii_seed = plot_df.loc[ratio_iii_mask, 'G3c'] if ratio_iii_mask.any() else plot_df['G3c']
    gic_std = float(pd.to_numeric(gi_seed, errors='coerce').std(ddof=1) / np.sqrt(max(len(gi_seed), 1)))
    giiic_std = float(pd.to_numeric(giii_seed, errors='coerce').std(ddof=1) / np.sqrt(max(len(giii_seed), 1)))
    if not np.isfinite(gic_std):
        gic_std = 0.0
    if not np.isfinite(giiic_std):
        giiic_std = 0.0
    r_vals = plot_df['r_elliptic'].to_numpy(dtype=float)
    fit_nm2 = _calc_case_stats(2.0, 2.0, 'Model A (fixed n=m=2)', variant='B', n_params=0)
    fit_nm1 = _calc_case_stats(1.0, 1.0, 'Model B (fixed n=m=1)', variant='B', n_params=0)
    fit_nmfree = _calc_case_stats(
        n_opt, m_opt,
        f"Model C (fitted n,m: n={n_opt:.4f}, m={m_opt:.4f})",
        method=nm_opt_method,
        objective_override=nm_opt_objective,
        variant=free_fit_variant,
        n_params=2,
    )

    def _print_fit_block(fit_result, model_label):
        print(f"  {model_label}")
        print(f"    exponents        : n={fit_result['n']:.4f}, m={fit_result['m']:.4f}")
        print(f"    GIc              : {gic_ref:.4f} +/- {gic_unc_weighted:.4f} J/m^2 (weighted)")
        print(f"    GIIIc            : {giiic_ref:.4f} +/- {giiic_unc_weighted:.4f} J/m^2 (weighted)")
        print(f"    optimizer        : {fit_result['method']}")
        print(f"    objective        : {fit_result['objective']:.6g}")
        print(f"    weighted_RMSE_a  : {fit_result['rmse_a']:.6g}")
        print(f"    weighted_RMSE_r  : {fit_result['rmse_r']:.6g}")
        print(f"    chi^2            : {fit_result['chi2']:.6g}")
        print(f"    reduced chi^2    : {fit_result['reduced_chi2']:.6g}")
        print(f"    r (mean/med/std) : {np.mean(r_vals):.4f} / {np.median(r_vals):.4f} / {np.std(r_vals):.4f}")

    print("")
    print("=" * 72)
    print("plot_GIII_ratio_vs_G - calibration report")
    print("-" * 72)
    print(f"  Data points used   : N={len(plot_df)}")
    print(f"  Gc threshold       : Gc_lim={Gc_lim:.3f}")
    print(f"  subset counts      : N_GIc={int(np.sum(ratio_i_mask))}, N_GIIIc={int(np.sum(ratio_iii_mask))}")
    print(f"  Uncertainty basis  : inverse-variance weighted mean (1/sigma^2)")
    print(f"  subset SE (legacy) : GIc={gic_std:.4f}, GIIIc={giiic_std:.4f} J/m^2")
    print(f"  free-fit residual  : variant {free_fit_variant}")
    print("-" * 72)
    _print_fit_block(fit_nm2, "Model A (fixed n=m=2)")
    print("")
    _print_fit_block(fit_nm1, "Model B (fixed n=m=1)")
    print("")
    _print_fit_block(fit_nmfree, fit_nmfree['name'])
    print("-" * 72)
    print(f"  free-fit diagnostics:")
    print(f"    starts total/success : {nm_starts_total} / {nm_starts_success}")
    print(f"    boundary hits        : {nm_boundary_hits}")
    if len(nm_top) > 1:
        obj_gap = float(nm_top[1][0] - nm_top[0][0])
        print(f"    objective gap (2nd-1st): {obj_gap:.6g}")
    if nm_top:
        bests = ", ".join([f"({n_i:.3f},{m_i:.3f}|{o_i:.4g})" for o_i, n_i, m_i in nm_top[:3]])
        print(f"    top minima (n,m|obj): {bests}")
    print("=" * 72)
    print("")

    # Propagate uncertainties (independent G1c/G3c assumption):
    # x = atan((G3*GIc)/(G1*GIIIc))
    # y = ((G1/GIc)^n + (G3/GIIIc)^m)
    s = plot_df['G13_total']
    g1 = plot_df['G1c']
    g3 = plot_df['G3c']
    sg1 = plot_df['G1c_unc']
    sg3 = plot_df['G3c_unc']
    g1_safe = np.where(np.abs(g1.to_numpy(dtype=float)) > 1e-12, g1.to_numpy(dtype=float), 1e-12)
    ratio_safe = (g3.to_numpy(dtype=float) * gic_ref) / (g1_safe * giiic_ref)
    denom = 1.0 + ratio_safe ** 2
    dx_dg1 = -(g3.to_numpy(dtype=float) * gic_ref) / ((g1_safe ** 2) * giiic_ref * denom)
    dx_dg3 = gic_ref / (g1_safe * giiic_ref * denom)
    plot_df['x_unc'] = np.sqrt((dx_dg1 * sg1) ** 2 + (dx_dg3 * sg3) ** 2)

    # y-uncertainty propagation for quadratic radial coordinate:
    # r = sqrt((GI/GIc)^2 + (GIII/GIIIc)^2)
    g1_np = g1.to_numpy(dtype=float)
    g3_np = g3.to_numpy(dtype=float)
    sg1_np = sg1.to_numpy(dtype=float)
    sg3_np = sg3.to_numpy(dtype=float)
    u_np = np.clip(g1_np / gic_ref, 0.0, None)
    v_np = np.clip(g3_np / giiic_ref, 0.0, None)
    r_np = np.sqrt(u_np ** 2 + v_np ** 2)
    r_safe = np.where(r_np > 1e-12, r_np, 1e-12)
    # dr/du = u/r, dr/dv = v/r
    dr_du = u_np / r_safe
    dr_dv = v_np / r_safe
    dr_dg1 = dr_du * (1.0 / gic_ref)
    dr_dg3 = dr_dv * (1.0 / giiic_ref)
    y_unc = np.sqrt((dr_dg1 * sg1_np) ** 2 + (dr_dg3 * sg3_np) ** 2)
    plot_df['y_unc'] = y_unc
    plot_df['x_unc'] = pd.to_numeric(plot_df['x_unc'], errors='coerce').fillna(0.0).clip(lower=0.0)
    plot_df['y_unc'] = pd.to_numeric(plot_df['y_unc'], errors='coerce').fillna(0.0).clip(lower=0.0)

    setup_visualization_style()
    plt.rcParams['font.family'] = SELECTED_FONT
    plt.rcParams['mathtext.fontset'] = MATH_FONT_SET
    plt.rcParams['mathtext.default'] = 'it'

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    def _get_alpha_equiv_color_from_base(base_color, alpha_val):
        color_name = None
        for name, value in lo.COLORS.items():
            if str(value).lower() == str(base_color).lower():
                color_name = name
                break
        if (
            color_name is not None
            and hasattr(lo, 'COLORS_ALPHA_EQUIV')
            and color_name in lo.COLORS_ALPHA_EQUIV
            and alpha_val in lo.COLORS_ALPHA_EQUIV[color_name]
        ):
            return lo.COLORS_ALPHA_EQUIV[color_name][alpha_val]
        return _alpha_equivalent_color(base_color, alpha_val)

    base_color = lo.COLORS['blue']
    marker_color = base_color  # real alpha used below
    errorbar_color = _get_alpha_equiv_color_from_base(base_color, 0.5)
    x_vals = plot_df['phi_elliptic'].to_numpy(dtype=float)
    y_vals = plot_df['r_quadratic'].to_numpy(dtype=float)
    x_err_vals = plot_df['x_unc'].to_numpy(dtype=float)
    y_err_vals = plot_df['y_unc'].to_numpy(dtype=float)

    # Bars first (background), same behavior as plot_mode_I_III_interaction.
    for i in range(len(x_vals)):
        if np.isfinite(x_err_vals[i]) and x_err_vals[i] > 0:
            ax.plot(
                [x_vals[i] - x_err_vals[i], x_vals[i] + x_err_vals[i]],
                [y_vals[i], y_vals[i]],
                color=errorbar_color,
                alpha=1.0,
                linewidth=ERRORBAR_STYLES['line_width'],
                zorder=0,
            )
        if np.isfinite(y_err_vals[i]) and y_err_vals[i] > 0:
            ax.plot(
                [x_vals[i], x_vals[i]],
                [y_vals[i] - y_err_vals[i], y_vals[i] + y_err_vals[i]],
                color=errorbar_color,
                alpha=1.0,
                linewidth=ERRORBAR_STYLES['line_width'],
                zorder=0,
            )

    ax.scatter(
        x_vals,
        y_vals,
        c=marker_color,
        s=marker_size ** 2 * 10,
        alpha=alpha,
        edgecolors='none',
        label=r'$r^{\mathcal{G}}$ data',
        zorder=1,
    )

    phi_fit = np.linspace(0.0, np.pi / 2, 250)
    r_fit_nm2 = np.array([_r_opt_from_phi(ph, n_val=2.0, m_val=2.0, variant='B') for ph in phi_fit], dtype=float)
    r_fit_nm1 = np.array([_r_opt_from_phi(ph, n_val=1.0, m_val=1.0, variant='B') for ph in phi_fit], dtype=float)
    r_fit_nmfree = np.array([_r_opt_from_phi(ph, n_val=n_opt, m_val=m_opt, variant=free_fit_variant) for ph in phi_fit], dtype=float)
    ax.plot(
        phi_fit,
        r_fit_nm2,
        color=base_color,
        linewidth=1.5,
        alpha=0.9,
        zorder=2,
        label='optimized law (n=m=2)' if show_legend else None,
    )
    ax.plot(
        phi_fit,
        r_fit_nm1,
        color=lo.COLORS['orange'],
        linewidth=1.5,
        alpha=0.9,
        zorder=2,
        label='optimized law (n=m=1)' if show_legend else None,
    )
    ax.plot(
        phi_fit,
        r_fit_nmfree,
        color=lo.COLORS['red'],
        linewidth=1.5,
        alpha=0.9,
        zorder=2,
        label=f'optimized law (n={n_opt:.2f}, m={m_opt:.2f})' if show_legend else None,
    )

    ax.set_xlabel(
        r'Elliptic mode mixity angle $\varphi^{\,\mathcal{G}}$ (rad) $\longrightarrow$',
        fontsize=VISUALIZATION_STYLES['font_size'],
        labelpad=labelpad_x,
        color='black',
    )
    ax.set_ylabel(
        r'Elliptic radial coordinate $r^{\,\mathcal{G}}$ $\longrightarrow$',
        fontsize=VISUALIZATION_STYLES['font_size'],
        labelpad=labelpad_y,
        color='black',
    )
    if title is not None:
        ax.set_title(title, fontsize=VISUALIZATION_STYLES['font_size'], pad=20)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.tick_params(
        axis='both', which='major',
        labelsize=VISUALIZATION_STYLES['font_size'],
        pad=10,
        width=tick_width,
        length=tick_length,
        color=TICK_COLOR,
        bottom=True, top=True, labeltop=False,
        left=True, right=True, labelright=False,
        labelcolor='black',
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(frame_thickness)
    ax.grid(False)
    if show_legend:
        ax.legend()
    plt.tight_layout()
    plt.show()
    return None


def plot_load_config_ERR(
    df_or_path,
    title=None,
    figsize=(8, 7.5),
    dpi=100,
    alpha=0.8,
    marker_size=2.3,
    tick_width=1,
    tick_length=10,
    labelpad_x=15,
    labelpad_y=15,
    frame_thickness=1,
    total_weights_rel_err=0.01,
    exclude_pst=True,
    only_pst=False,
    switch_x_axis_dir=False,
    phi_filter='all',
    jitter_offset=0.25,
    x_lim=None,
    y_lim=None,
    show_rel_G=False,
    axis_step_equivalence=(10.0, 0.10),
):
    """
    Raw datapoints with uncertainty bars over additional applied force.

    x-axis:
        F_w = total_weights * g   [N]
    y-axis:
        G_I (red), G_II (orange), G_III (blue)

    Parameters
    ----------
    phi_filter : str, default 'all'
        Inclination filter based on column `phi`:
        - 'all': no inclination filtering
        - 'upslope': keep only phi > 0
        - 'downslope': keep only phi < 0
    jitter_offset : float, default 0.25
        Constant horizontal jitter offset in x-data units (N), used uniformly
        for all weight bundles and for mode center separation.
    x_lim : tuple or None, default None
        Optional x-axis limits as (xmin, xmax).
    y_lim : tuple or None, default None
        Optional y-axis limits as (ymin, ymax).
    show_rel_G : bool, default False
        If True, normalize mode ERR values by `Gc` and plot mode ratios
        (G1c/Gc, G2c/Gc, G3c/Gc) instead of absolute ERR values.
    axis_step_equivalence : tuple, default (10.0, 0.05)
        Visual x/y distance equivalence defined as (x_step, y_step):
        one x-step has the same on-screen length as one y-step.
    """
    if isinstance(df_or_path, pd.DataFrame):
        df = df_or_path.copy()
    else:
        try:
            df = pd.read_parquet(df_or_path, engine='fastparquet')
            print(f"Loaded data from Parquet: {df_or_path}")
        except Exception as e:
            print(f"Error loading parquet: {e}")
            return None

    required_cols = ['total weights', 'G1c', 'G2c', 'G3c']
    if show_rel_G:
        required_cols.append('Gc')
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return None

    plot_df = df.copy()

    pst_afn_set = {1, 2, 3, 4, 5, 6, 7}
    if 'AFN' in plot_df.columns and (exclude_pst or only_pst):
        afn_numeric = pd.to_numeric(plot_df['AFN'], errors='coerce')
        is_pst = afn_numeric.isin(pst_afn_set)
        if bool(only_pst):
            plot_df = plot_df[is_pst].copy()
        elif bool(exclude_pst):
            plot_df = plot_df[~is_pst].copy()
    elif (exclude_pst or only_pst) and 'AFN' not in plot_df.columns:
        print("Warning: AFN column not found; PST filtering was skipped.")

    phi_filter_norm = str(phi_filter).strip().lower()
    if phi_filter_norm not in {'all', 'upslope', 'downslope'}:
        print(f"Warning: Unknown phi_filter='{phi_filter}'. Using 'all'.")
        phi_filter_norm = 'all'
    if phi_filter_norm != 'all':
        if 'phi' not in plot_df.columns:
            print("Warning: phi column not found; inclination filtering was skipped.")
        else:
            phi_vals = pd.to_numeric(plot_df['phi'], errors='coerce')
            if phi_filter_norm == 'upslope':
                plot_df = plot_df[phi_vals > 0].copy()
            elif phi_filter_norm == 'downslope':
                plot_df = plot_df[phi_vals < 0].copy()

    # Use plain float dtypes to avoid pd.NA ambiguity in masks/comparisons.
    plot_df['total weights'] = pd.to_numeric(plot_df['total weights'], errors='coerce').astype('float64')
    plot_df['G1c'] = pd.to_numeric(plot_df['G1c'], errors='coerce').astype('float64')
    plot_df['G2c'] = pd.to_numeric(plot_df['G2c'], errors='coerce').astype('float64')
    plot_df['G3c'] = pd.to_numeric(plot_df['G3c'], errors='coerce').astype('float64')
    if show_rel_G:
        plot_df['Gc'] = pd.to_numeric(plot_df['Gc'], errors='coerce').astype('float64')
    g1_unc_col = 'G1c_uncertainty' if 'G1c_uncertainty' in plot_df.columns else None
    g2_unc_col = 'G2c_uncertainty' if 'G2c_uncertainty' in plot_df.columns else None
    g3_unc_col = 'G3c_uncertainty' if 'G3c_uncertainty' in plot_df.columns else None
    tw_unc_col = 'total weights_uncertainty' if 'total weights_uncertainty' in plot_df.columns else None
    if g1_unc_col is not None:
        plot_df['G1c_unc'] = pd.to_numeric(plot_df[g1_unc_col], errors='coerce').fillna(0.0).clip(lower=0.0)
    else:
        plot_df['G1c_unc'] = 0.0
    if g2_unc_col is not None:
        plot_df['G2c_unc'] = pd.to_numeric(plot_df[g2_unc_col], errors='coerce').fillna(0.0).clip(lower=0.0)
    else:
        plot_df['G2c_unc'] = 0.0
    if g3_unc_col is not None:
        plot_df['G3c_unc'] = pd.to_numeric(plot_df[g3_unc_col], errors='coerce').fillna(0.0).clip(lower=0.0)
    else:
        plot_df['G3c_unc'] = 0.0
    if tw_unc_col is not None:
        plot_df['total_weights_unc'] = pd.to_numeric(plot_df[tw_unc_col], errors='coerce').fillna(0.0).clip(lower=0.0)
    else:
        plot_df['total_weights_unc'] = (plot_df['total weights'].abs() * abs(float(total_weights_rel_err))).astype('float64')

    # Additional applied force from total mass.
    # total weights are stored in kg; convert to force with g=9.81 m/s^2.
    plot_df['F_w'] = plot_df['total weights'] * 9.81
    plot_df = plot_df.dropna(subset=['G1c', 'G2c', 'G3c'])
    if show_rel_G:
        plot_df = plot_df.dropna(subset=['Gc'])
        plot_df = plot_df[plot_df['Gc'].abs() > 1e-12]
    if plot_df.empty:
        print("No valid rows to plot (check filters and required columns).")
        return None

    # Keep datapoints without valid force calculation and place them at x=0.
    plot_df['F_w_plot'] = pd.to_numeric(plot_df['F_w'], errors='coerce').fillna(0.0)
    # x uncertainty from total-weights uncertainty (kg) propagated to force (N).
    plot_df['F_w_unc'] = (9.81 * pd.to_numeric(plot_df['total_weights_unc'], errors='coerce')).fillna(0.0).clip(lower=0.0)

    if show_rel_G:
        gc_safe = plot_df['Gc'].to_numpy(dtype=float)
        plot_df['G1_plot'] = plot_df['G1c'].to_numpy(dtype=float) / gc_safe
        plot_df['G2_plot'] = plot_df['G2c'].to_numpy(dtype=float) / gc_safe
        plot_df['G3_plot'] = plot_df['G3c'].to_numpy(dtype=float) / gc_safe
        plot_df['G1_plot_unc'] = plot_df['G1c_unc'].to_numpy(dtype=float) / np.abs(gc_safe)
        plot_df['G2_plot_unc'] = plot_df['G2c_unc'].to_numpy(dtype=float) / np.abs(gc_safe)
        plot_df['G3_plot_unc'] = plot_df['G3c_unc'].to_numpy(dtype=float) / np.abs(gc_safe)
    else:
        plot_df['G1_plot'] = plot_df['G1c']
        plot_df['G2_plot'] = plot_df['G2c']
        plot_df['G3_plot'] = plot_df['G3c']
        plot_df['G1_plot_unc'] = plot_df['G1c_unc']
        plot_df['G2_plot_unc'] = plot_df['G2c_unc']
        plot_df['G3_plot_unc'] = plot_df['G3c_unc']

    # Match style behavior used across the project.
    setup_visualization_style()
    plt.rcParams['font.family'] = SELECTED_FONT
    plt.rcParams['mathtext.fontset'] = MATH_FONT_SET
    plt.rcParams['mathtext.default'] = 'it'

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Raw points with bars-first style (same pattern as mode interaction plots).
    mode_i_color = lo.COLORS.get('red', lo.COLORS.get('darkred', '#c54325'))
    mode_ii_color = lo.COLORS.get('orange', '#f58c00')
    mode_iii_color = lo.COLORS.get('blue', '#0077BB')
    x_vals = plot_df['F_w_plot'].to_numpy(dtype=float)
    x_err_vals = plot_df['F_w_unc'].to_numpy(dtype=float)
    x_groups = np.round(x_vals, 2)
    jitter_offset = abs(float(jitter_offset))
    # Constant strip/beeswarm offsets for all weight bundles.
    strip_half_width = jitter_offset
    if switch_x_axis_dir:
        # Mirror jitter orientation when axis direction is reversed.
        mode_center_offsets = {'G1c': jitter_offset, 'G2c': 0.0, 'G3c': -jitter_offset}
    else:
        mode_center_offsets = {'G1c': -jitter_offset, 'G2c': 0.0, 'G3c': jitter_offset}
    # Keep draw order fixed: I -> II -> III.
    series_specs = [
        ('G1_plot', 'G1_plot_unc', mode_i_color, 1, '^'),  # Mode I: triangle
        ('G2_plot', 'G2_plot_unc', mode_ii_color, 2, 's'),  # Mode II: rectangle/square
        ('G3_plot', 'G3_plot_unc', mode_iii_color, 3, 'o'),  # Mode III: round
    ]
    rng = np.random.default_rng()
    for y_col, y_unc_col, series_color, z, marker_shape in series_specs:
        y_vals = plot_df[y_col].to_numpy(dtype=float)
        y_err_vals = plot_df[y_unc_col].to_numpy(dtype=float)
        x_plot_vals = x_vals.copy()
        mode_offset = mode_center_offsets.get(y_col, 0.0)
        # Build random strip positions per force group.
        for g in np.sort(np.unique(x_groups[np.isfinite(x_groups)])):
            idx = np.where(np.isclose(x_groups, g, atol=1e-12))[0]
            if idx.size == 0:
                continue
            offsets = rng.uniform(-strip_half_width, strip_half_width, size=idx.size)
            x_plot_vals[idx] = x_vals[idx] + mode_offset + offsets
        errorbar_color = _alpha_equivalent_color(series_color, ERRORBAR_STYLES['alpha'])
        for i in range(len(x_plot_vals)):
            if np.isfinite(x_err_vals[i]) and x_err_vals[i] > 0:
                ax.plot(
                    [x_plot_vals[i] - x_err_vals[i], x_plot_vals[i] + x_err_vals[i]],
                    [y_vals[i], y_vals[i]],
                    color=errorbar_color,
                    alpha=1.0,
                    linewidth=ERRORBAR_STYLES['line_width'],
                    zorder=0,
                )
            if np.isfinite(y_err_vals[i]) and y_err_vals[i] > 0:
                ax.plot(
                    [x_plot_vals[i], x_plot_vals[i]],
                    [y_vals[i] - y_err_vals[i], y_vals[i] + y_err_vals[i]],
                    color=errorbar_color,
                    alpha=1.0,
                    linewidth=ERRORBAR_STYLES['line_width'],
                    zorder=0,
                )
        ax.scatter(
            x_plot_vals,
            y_vals,
            c=series_color,
            s=marker_size ** 2 * 10,
            alpha=alpha,
            edgecolors='none',
            marker=marker_shape,
            zorder=z,
        )

    n_depicted = len(plot_df)
    print("\n[plot_load_config_ERR] Depicted datapoints")
    print(f"  N = {n_depicted}")

    if switch_x_axis_dir:
        x_label = r'$\longleftarrow$ Additional weights $F_{\mathrm{w}}$ (N)'
    else:
        x_label = r'Additional weights $F_{\mathrm{w}}$ (N) $\longrightarrow$'
    ax.set_xlabel(
        x_label,
        fontsize=VISUALIZATION_STYLES['font_size'],
        labelpad=labelpad_x,
        color='black',
    )
    if show_rel_G:
        y_label = r'Mode ratio $\psi$ $\longrightarrow$'
    else:
        y_label = r'Energy release rate $\mathcal{G}$ (J/m$^2$) $\longrightarrow$'
    ax.set_ylabel(
        y_label,
        fontsize=VISUALIZATION_STYLES['font_size'],
        labelpad=labelpad_y,
        color='black',
    )
    if title is not None:
        ax.set_title(title, fontsize=VISUALIZATION_STYLES['font_size'])

    x_valid = plot_df['F_w_plot'].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if x_valid.size > 0:
        x_min_auto = float(np.min(x_valid))
        x_max_auto = float(np.max(x_valid))
    else:
        x_min_auto, x_max_auto = 0.0, 1.0
    x_span = max(x_max_auto - x_min_auto, 1.0)
    x_margin = 0.06 * x_span
    if x_lim is not None:
        x_lo, x_hi = float(x_lim[0]), float(x_lim[1])
    else:
        x_lo = x_min_auto - x_margin
        x_hi = x_max_auto + x_margin
    if switch_x_axis_dir:
        # Right-to-left axis direction (zero on the right, positive values to the left).
        ax.set_xlim(max(x_lo, x_hi), min(x_lo, x_hi))
    else:
        ax.set_xlim(min(x_lo, x_hi), max(x_lo, x_hi))
    if y_lim is not None:
        ax.set_ylim(float(y_lim[0]), float(y_lim[1]))

    # Enforce configurable visual spacing equivalence:
    # one x_step has the same on-screen length as one y_step.
    try:
        x_step_eq, y_step_eq = axis_step_equivalence
        x_step_eq = abs(float(x_step_eq))
        y_step_eq = abs(float(y_step_eq))
    except Exception:
        x_step_eq, y_step_eq = 10.0, 0.05
    if x_step_eq <= 0 or y_step_eq <= 0:
        x_step_eq, y_step_eq = 10.0, 0.05
    ax.set_aspect(x_step_eq / y_step_eq, adjustable='box')

    import matplotlib.ticker as mticker
    ax.xaxis.set_major_locator(mticker.MultipleLocator(x_step_eq))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(y_step_eq))

    # Draw grouped mean values (per applied weight) as black markers with mode shape.
    plot_df['F_w_group'] = plot_df['F_w_plot'].round(2)
    grouped_means = plot_df.groupby('F_w_group', dropna=True)[['G1_plot', 'G2_plot', 'G3_plot']].mean(numeric_only=True)
    # Uncertainty of grouped means from per-point uncertainties:
    # sigma_mean = sqrt(sum(sigma_i^2)) / n
    grouped_mean_errs = {}
    for plot_col, unc_col in [('G1_plot', 'G1_plot_unc'), ('G2_plot', 'G2_plot_unc'), ('G3_plot', 'G3_plot_unc')]:
        err_ser = plot_df.groupby('F_w_group', dropna=True)[unc_col].apply(
            lambda s: (np.sqrt(np.sum(np.square(pd.to_numeric(s, errors='coerce').dropna().to_numpy(dtype=float)))) / max(len(s.dropna()), 1))
        )
        grouped_mean_errs[plot_col] = err_ser
    mean_specs = [('G1_plot', '^', 4), ('G2_plot', 's', 5), ('G3_plot', 'o', 6)]
    mean_cols_map = {
        fr'mode I': 'G1_plot',
        fr'mode II': 'G2_plot',
        fr'mode III': 'G3_plot',
    }
    print("[plot_load_config_ERR] Grouped mean values (black markers)")
    value_unit = "-" if show_rel_G else "J/m^2"
    for mode_name, col_name in mean_cols_map.items():
        series_mean_vals = grouped_means[col_name].dropna()
        if len(series_mean_vals) == 0:
            print(f"  {mode_name}: no grouped means available")
            continue
        overall_mean = float(series_mean_vals.mean())
        print(f"  {mode_name}:")
        print(f"    overall grouped mean = {overall_mean:.4f} {value_unit}")
        per_weight_text = ", ".join(
            [f"{xv:.2f}N:{yv:.4f}" for xv, yv in zip(series_mean_vals.index, series_mean_vals.values)]
        )
        print(f"    per weight -> {per_weight_text}")
    for y_col, _, z in mean_specs:
        series_mean = grouped_means[y_col].dropna()
        if len(series_mean) >= 2:
            ax.plot(
                series_mean.index.to_numpy(dtype=float),
                series_mean.to_numpy(dtype=float),
                color='black',
                linewidth=1.5,
                alpha=0.3,
                zorder=z - 0.2,
            )
    for x_group, mean_row in grouped_means.iterrows():
        if not np.isfinite(x_group):
            continue
        for y_col, marker_shape, z in mean_specs:
            y_mean = mean_row.get(y_col, np.nan)
            if not np.isfinite(y_mean):
                continue
            ax.scatter(
                [x_group],
                [y_mean],
                c='black',
                s=marker_size ** 2 * 10,
                marker=marker_shape,
                alpha=1.0,
                edgecolors='none',
                zorder=z,
            )
            # For PST-only mode-ratio plots: annotate mean ± error next to black mean markers.
            if only_pst and show_rel_G:
                y_err = np.nan
                if y_col in grouped_mean_errs and x_group in grouped_mean_errs[y_col].index:
                    y_err = grouped_mean_errs[y_col].loc[x_group]
                if np.isfinite(y_err):
                    ax.text(
                        x_group + 0.15,
                        y_mean,
                        f"{y_mean:.2f}±{y_err:.2f}",
                        fontsize=VISUALIZATION_STYLES['font_size'],
                        color='black',
                        alpha=0.95,
                        ha='left',
                        va='center',
                        zorder=z + 0.1,
                    )

    # Upper-left mode label text (no background), in respective series colors.
    text_x = 0.02
    text_y0 = 0.98
    text_dy = 0.055
    ax.text(
        text_x, text_y0, "mode I",
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=VISUALIZATION_STYLES['font_size'],
        color=mode_i_color,
        alpha=1.0,
        zorder=15,
    )
    ax.text(
        text_x, text_y0 - text_dy, "mode II",
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=VISUALIZATION_STYLES['font_size'],
        color=mode_ii_color,
        alpha=1.0,
        zorder=15,
    )
    ax.text(
        text_x, text_y0 - 2 * text_dy, "mode III",
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=VISUALIZATION_STYLES['font_size'],
        color=mode_iii_color,
        alpha=1.0,
        zorder=15,
    )
    ax.text(
        text_x, text_y0 - 3 * text_dy, "mean value",
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=VISUALIZATION_STYLES['font_size'],
        color='black',
        alpha=1.0,
        zorder=15,
    )

    # Layout style like parameter-study plot:
    # no left/top/right frame, no tick marks, dotted horizontal guides in back.
    ax.tick_params(
        axis='x', which='major',
        labelsize=VISUALIZATION_STYLES['font_size'],
        pad=10, width=tick_width, length=tick_length,
        bottom=True, top=False, labelbottom=True,
        labelcolor='black', color='grey',
    )
    ax.tick_params(
        axis='y', which='major',
        labelsize=VISUALIZATION_STYLES['font_size'],
        pad=10, width=0, length=0,
        left=False, right=False, labelleft=True, labelright=False,
        labelcolor='black', color='grey',
    )
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_linewidth(frame_thickness)
    ax.spines['bottom'].set_color('black')

    y_lo, y_hi = ax.get_ylim()
    y_min, y_max = (min(y_lo, y_hi), max(y_lo, y_hi))
    y_tick_vals = [
        yv for yv in ax.get_yticks()
        if np.isfinite(yv)
        and (y_min - 1e-12) <= yv <= (y_max + 1e-12)
        and not np.isclose(yv, 0.0, atol=1e-12)
    ]
    for yv in y_tick_vals:
        ax.axhline(
            y=yv,
            color='grey',
            linewidth=2,
            linestyle=(0, (0.05, 2.0)),
            dash_capstyle='round',
            alpha=0.7,
            zorder=-5,
        )
    ax.grid(False)
    plt.show()
    return fig


def plot_mode_III_ratio_to_GIGIII(
    df_or_path,
    title=None,
    figsize=(8, 7.5),
    dpi=100,
    alpha=0.8,
    marker_size=2.3,
    tick_width=1,
    tick_length=10,
    labelpad_x=15,
    labelpad_y=15,
    frame_thickness=1,
    x_lim=None,
    y_lim=None,
    n=None,
    m=None,
    GIIIc=None,
    fit_curve_samples=1200,
):
    """
    Plot Mode III ratio over total mixed-mode energy release rate.

    x-axis:
        psi_III = G_III / (G_I + G_III)
    y-axis:
        G_I + G_III

    Optional red interaction overlay:
        (G_I/GIc)^n + (G_III/GIIIc)^m = 1
    with GIc automatically estimated from PST experiments (AFN 1..7).
    """
    if isinstance(df_or_path, pd.DataFrame):
        df = df_or_path.copy()
    else:
        try:
            df = pd.read_parquet(df_or_path, engine='fastparquet')
            print(f"Loaded data from Parquet: {df_or_path}")
        except Exception as e:
            print(f"Error loading parquet: {e}")
            return None

    required_cols = ['G1c', 'G3c']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return None

    plot_df = df.copy()
    plot_df['G1c'] = pd.to_numeric(plot_df['G1c'], errors='coerce')
    plot_df['G3c'] = pd.to_numeric(plot_df['G3c'], errors='coerce')
    g1_unc_col = 'G1c_uncertainty' if 'G1c_uncertainty' in plot_df.columns else None
    g3_unc_col = 'G3c_uncertainty' if 'G3c_uncertainty' in plot_df.columns else None
    if g1_unc_col is not None:
        plot_df['G1c_unc'] = pd.to_numeric(plot_df[g1_unc_col], errors='coerce').fillna(0.0).clip(lower=0.0)
    else:
        plot_df['G1c_unc'] = 0.0
    if g3_unc_col is not None:
        plot_df['G3c_unc'] = pd.to_numeric(plot_df[g3_unc_col], errors='coerce').fillna(0.0).clip(lower=0.0)
    else:
        plot_df['G3c_unc'] = 0.0

    plot_df['G13_total'] = plot_df['G1c'] + plot_df['G3c']
    plot_df = plot_df.dropna(subset=['G1c', 'G3c', 'G13_total'])
    plot_df = plot_df[plot_df['G13_total'] > 0]
    if plot_df.empty:
        print("No valid rows to plot (requires G1c, G3c with G1c+G3c > 0).")
        return None

    # GIc reference from PST subset (AFN 1..7), same rule as in
    # plot_mode_I_III_interaction; fallback to all rows if PST subset is empty.
    gic_ref = np.nan
    gic_ref_sem = np.nan
    gic_ref_n = 0
    if 'AFN' in plot_df.columns:
        afn_num = pd.to_numeric(plot_df['AFN'], errors='coerce')
        pst_mask = afn_num.round().isin([1, 2, 3, 4, 5, 6, 7])
        gic_seed = plot_df.loc[pst_mask, 'G1c'] if pst_mask.any() else plot_df['G1c']
    else:
        pst_mask = pd.Series(False, index=plot_df.index)
        gic_seed = plot_df['G1c']
    gic_seed = pd.to_numeric(gic_seed, errors='coerce').dropna()
    if not gic_seed.empty:
        gic_ref = float(gic_seed.mean())
        gic_ref_n = int(gic_seed.shape[0])
        if gic_ref_n > 1:
            gic_ref_sem = float(gic_seed.std(ddof=1) / np.sqrt(gic_ref_n))
        else:
            gic_ref_sem = 0.0

    # Coordinates.
    plot_df['psi_III'] = plot_df['G3c'] / plot_df['G13_total']
    plot_df['G13_plot'] = plot_df['G13_total']

    # Uncertainty propagation:
    # psi_III = G3 / (G1 + G3), G13 = G1 + G3
    g1 = plot_df['G1c'].to_numpy(dtype=float)
    g3 = plot_df['G3c'].to_numpy(dtype=float)
    sg1 = plot_df['G1c_unc'].to_numpy(dtype=float)
    sg3 = plot_df['G3c_unc'].to_numpy(dtype=float)
    s = np.maximum(g1 + g3, 1e-12)
    dpsi_dg1 = -g3 / (s ** 2)
    dpsi_dg3 = g1 / (s ** 2)
    x_unc = np.sqrt((dpsi_dg1 * sg1) ** 2 + (dpsi_dg3 * sg3) ** 2)
    y_unc = np.sqrt(sg1 ** 2 + sg3 ** 2)
    plot_df['x_unc'] = np.clip(np.nan_to_num(x_unc, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    plot_df['y_unc'] = np.clip(np.nan_to_num(y_unc, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)

    setup_visualization_style()
    plt.rcParams['font.family'] = SELECTED_FONT
    plt.rcParams['mathtext.fontset'] = MATH_FONT_SET
    plt.rcParams['mathtext.default'] = 'it'

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    def _get_alpha_equiv_color_from_base(base_color, alpha_val):
        color_name = None
        for name, value in lo.COLORS.items():
            if str(value).lower() == str(base_color).lower():
                color_name = name
                break
        if (
            color_name is not None
            and hasattr(lo, 'COLORS_ALPHA_EQUIV')
            and color_name in lo.COLORS_ALPHA_EQUIV
            and alpha_val in lo.COLORS_ALPHA_EQUIV[color_name]
        ):
            return lo.COLORS_ALPHA_EQUIV[color_name][alpha_val]
        return _alpha_equivalent_color(base_color, alpha_val)

    base_color = lo.COLORS['blue']
    errorbar_color = _get_alpha_equiv_color_from_base(base_color, 0.5)
    x_vals = plot_df['psi_III'].to_numpy(dtype=float)
    y_vals = plot_df['G13_plot'].to_numpy(dtype=float)
    x_err_vals = plot_df['x_unc'].to_numpy(dtype=float)
    y_err_vals = plot_df['y_unc'].to_numpy(dtype=float)

    # Error bars first (background), then markers on top.
    for i in range(len(x_vals)):
        if np.isfinite(x_err_vals[i]) and x_err_vals[i] > 0:
            ax.plot(
                [x_vals[i] - x_err_vals[i], x_vals[i] + x_err_vals[i]],
                [y_vals[i], y_vals[i]],
                color=errorbar_color,
                alpha=1.0,
                linewidth=ERRORBAR_STYLES['line_width'],
                zorder=0,
            )
        if np.isfinite(y_err_vals[i]) and y_err_vals[i] > 0:
            ax.plot(
                [x_vals[i], x_vals[i]],
                [y_vals[i] - y_err_vals[i], y_vals[i] + y_err_vals[i]],
                color=errorbar_color,
                alpha=1.0,
                linewidth=ERRORBAR_STYLES['line_width'],
                zorder=0,
            )

    ax.scatter(
        x_vals,
        y_vals,
        c=base_color,
        s=marker_size ** 2 * 10,
        alpha=0.8,
        edgecolors='none',
        zorder=1,
    )

    # Optional red interaction curve from user-provided n, m, GIIIc
    # with GIc estimated automatically from PST rows.
    if n is not None and m is not None and GIIIc is not None:
        try:
            n_val = float(n)
            m_val = float(m)
            giiic_val = float(GIIIc)
        except Exception:
            n_val = m_val = giiic_val = np.nan
        if (
            np.isfinite(n_val) and np.isfinite(m_val) and np.isfinite(giiic_val)
            and n_val > 0.0 and m_val > 0.0 and giiic_val > 0.0
            and np.isfinite(gic_ref) and gic_ref > 0.0
        ):
            t = np.linspace(0.0, 1.0, max(int(fit_curve_samples), 200))
            gi_curve = gic_ref * t
            with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
                giii_curve = giiic_val * np.power(np.clip(1.0 - np.power(t, n_val), 0.0, None), 1.0 / m_val)
            gsum_curve = gi_curve + giii_curve
            psi_curve = np.divide(giii_curve, np.maximum(gsum_curve, 1e-12))
            valid_curve = np.isfinite(psi_curve) & np.isfinite(gsum_curve)
            if np.any(valid_curve):
                # Sort by psi for visually clean plotting in this coordinate system.
                order = np.argsort(psi_curve[valid_curve])
                ax.plot(
                    psi_curve[valid_curve][order],
                    gsum_curve[valid_curve][order],
                    color=lo.COLORS.get('red', '#d71149'),
                    linewidth=2.0,
                    alpha=0.95,
                    linestyle='-',
                    zorder=3,
                )
            print("")
            print("[plot_mode_III_ratio_to_GIGIII] Red interaction overlay")
            print(f"  GIc (auto, PST AFN 1..7): {gic_ref:.4f} +/- {gic_ref_sem:.4f} J/m^2 (n={gic_ref_n})")
            print(f"  n={n_val:.4f}, m={m_val:.4f}, GIIIc={giiic_val:.4f} J/m^2")
        else:
            print("[plot_mode_III_ratio_to_GIGIII] Red interaction overlay skipped: invalid n/m/GIIIc or GIc reference.")

    ax.set_xlabel(
        r'Mode III ratio $\mathcal{G}_{\mathrm{III}}/(\mathcal{G}_{\mathrm{I}}+\mathcal{G}_{\mathrm{III}})$ $\longrightarrow$',
        fontsize=VISUALIZATION_STYLES['font_size'],
        labelpad=labelpad_x,
        color='black',
    )
    ax.set_ylabel(
        r'Energy release rate $\mathcal{G}_{\mathrm{I}}+\mathcal{G}_{\mathrm{III}}$ (J/m$^2$) $\longrightarrow$',
        fontsize=VISUALIZATION_STYLES['font_size'],
        labelpad=labelpad_y,
        color='black',
    )
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if title is not None:
        ax.set_title(title, fontsize=VISUALIZATION_STYLES['font_size'], pad=20)

    ax.tick_params(
        axis='both', which='major',
        labelsize=VISUALIZATION_STYLES['font_size'],
        pad=10,
        width=tick_width,
        length=tick_length,
        color=TICK_COLOR,
        bottom=True, top=True, labeltop=False,
        left=True, right=True, labelright=False,
        labelcolor='black',
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(frame_thickness)
    ax.grid(False)
    plt.tight_layout()
    plt.show()
    return None


def plot_pc_vs_g1(
    df_or_path,
    title=None,
    figsize=(8, 7.5),
    dpi=100,
    alpha=0.8,
    marker_size=2.3,
    show_legend=False,
    tick_width=1,
    tick_length=10,
    labelpad_x=15,
    labelpad_y=15,
    frame_thickness=1,
    xlim=None,
    ylim=None,
):
    """
    Plot fracture-limit moment a*P_c over Mode-I fracture energy G_I.

    x-axis:
        G_I = G1c [J/m^2]
    y-axis:
        M_c = a * P_c [N m], with a provided in mm
    """
    if isinstance(df_or_path, pd.DataFrame):
        df = df_or_path.copy()
    else:
        try:
            df = pd.read_parquet(df_or_path, engine='fastparquet')
            print(f"Loaded data from Parquet: {df_or_path}")
        except Exception as e:
            print(f"Error loading parquet: {e}")
            return None

    required_cols = ['G1c', 'P_c', 'a']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return None

    plot_df = df.copy()
    plot_df['G1c'] = pd.to_numeric(plot_df['G1c'], errors='coerce')
    plot_df['a'] = pd.to_numeric(plot_df['a'], errors='coerce').fillna(0.0)

    def _pc_force_value(v):
        pc_force = extract_pc_force(v)
        if pc_force is not None:
            return pc_force
        try:
            return float(v)
        except Exception:
            return np.nan

    plot_df['P_c_force'] = plot_df['P_c'].apply(_pc_force_value)
    plot_df['P_c_force'] = pd.to_numeric(plot_df['P_c_force'], errors='coerce').fillna(0.0)
    # a is given in mm, so a*P_c yields N*mm; convert to N*m by dividing by 1000.
    plot_df['M_c_Nm'] = (plot_df['a'] * plot_df['P_c_force']) / 1000.0
    plot_df = plot_df.dropna(subset=['G1c'])
    if plot_df.empty:
        print("No valid rows to plot (requires numeric G1c and P_c force).")
        return None

    setup_visualization_style()
    plt.rcParams['font.family'] = SELECTED_FONT
    plt.rcParams['mathtext.fontset'] = MATH_FONT_SET
    plt.rcParams['mathtext.default'] = 'it'

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.scatter(
        plot_df['G1c'],
        plot_df['M_c_Nm'],
        alpha=alpha,
        s=marker_size ** 2 * 10,
        facecolor=lo.COLORS['blue'],
        edgecolors='none',
        linewidths=0,
        marker='o',
        label=r'$a\,P_c$ vs $G_I$' if show_legend else '',
        zorder=2,
    )

    ax.set_xlabel(
        r'Mode-I fracture energy $\mathcal{G}_{\mathrm{I}}$ (J/m$^2$) $\longrightarrow$',
        fontsize=VISUALIZATION_STYLES['font_size'],
        labelpad=labelpad_x,
        color='black',
    )
    ax.set_ylabel(
        r'Fracture limit moment $a\,P_c$ (N m) $\longrightarrow$',
        fontsize=VISUALIZATION_STYLES['font_size'],
        labelpad=labelpad_y,
        color='black',
    )

    if title is not None:
        ax.set_title(title, fontsize=VISUALIZATION_STYLES['font_size'], pad=20)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.tick_params(
        axis='both', which='major',
        labelsize=VISUALIZATION_STYLES['font_size'],
        pad=10,
        width=tick_width,
        length=tick_length,
        bottom=True, top=True, labeltop=False,
        left=True, right=True, labelright=False,
        labelcolor='black', color='grey',
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(frame_thickness)
    ax.grid(False)

    if show_legend:
        ax.legend(
            loc='best',
            fontsize=VISUALIZATION_STYLES['font_size'],
            frameon=True,
            fancybox=False,
            edgecolor='black',
            framealpha=1.0,
        )

    plt.tight_layout()
    return fig
