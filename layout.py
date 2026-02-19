"""
Layout and styling configuration for the NotchedBeam submission notebook.

This module provides a consistent color palette and font configuration.
Plotting styles and configuration have been moved to visualisation.py.
"""

import matplotlib.font_manager as fm
from typing import Dict, List, Tuple, Optional

# ============================================================================
# COLOR PALETTE
# ============================================================================

# Primary color palette (hex values)
COLORS = {
    'red': '#dc0100',
    'orange': '#fa8200',
    'gold': '#ffcd00',
    'indigo': '#000a51',
    'blue': '#2588bf',
    'lightblue': '#75d3f2'
}

# RGB equivalents for matplotlib
COLORS_RGB = {
    name: tuple(int(hex_color[i:i+2], 16)/255.0 for i in (1, 3, 5))
    for name, hex_color in COLORS.items()
}

# ============================================================================
# COLOR ACCESS CLASS
# ============================================================================

class ColorPalette:
    """
    Easy access to color palette with dot notation.
    
    Usage:
        lo = ColorPalette()
        color = lo.indigo  # Returns '#8319d7'
        rgb_color = lo.indigo_rgb  # Returns (0.51, 0.1, 0.84)
    """
    
    def __init__(self):
        # Add hex colors as attributes
        for name, hex_color in COLORS.items():
            setattr(self, name, hex_color)
        
        # Add RGB colors as attributes
        for name, rgb_color in COLORS_RGB.items():
            setattr(self, f"{name}_rgb", rgb_color)
    
    def get_color(self, name: str) -> str:
        """Get hex color by name."""
        return COLORS.get(name, '#000000')
    
    def get_rgb(self, name: str) -> Tuple[float, float, float]:
        """Get RGB color by name."""
        return COLORS_RGB.get(name, (0.0, 0.0, 0.0))
    
    def list_colors(self) -> List[str]:
        """List all available color names."""
        return list(COLORS.keys())
    
    def get_palette(self, names: Optional[List[str]] = None) -> List[str]:
        """Get a list of colors for plotting."""
        if names is None:
            return list(COLORS.values())
        return [COLORS.get(name, '#000000') for name in names]

# ============================================================================
# FONT CONFIGURATION
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

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def print_color_palette():
    """Print the color palette with names and hex values."""
    print("Color Palette:")
    print("=" * 50)
    for name, hex_color in COLORS.items():
        print(f"{name:12} : {hex_color}")

def plot_color_examples():
    """
    Plot example visualizations for each color in the palette.
    Creates two plots: one with all colors as lines, another with all colors as filled areas.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6))
    
    # Generate x data
    x = np.linspace(0, 10, 100)
    
    # Plot 1: All colors as lines
    for idx, (name, color) in enumerate(COLORS.items()):
        # Create different wave patterns for each color
        y = np.sin(x + idx * 0.5) + idx * 0.5
        ax1.plot(x, y, color=color, linewidth=3, label=name, alpha=0.8)
    
    ax1.set_xlabel('X', fontsize=14)
    ax1.set_ylabel('Y', fontsize=14)
    #ax1.set_title('All Colors - Lines', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=False, ncol=2)
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: All colors as filled areas
    for idx, (name, color) in enumerate(COLORS.items()):
        # Create different wave patterns for each color
        y_bottom = idx * 0.5
        y_top = np.sin(x + idx * 0.5) * 0.4 + idx * 0.5 + 0.5
        ax2.fill_between(x, y_bottom, y_top, color=color, alpha=0.7, label=name)
    
    ax2.set_xlabel('X', fontsize=14)
    ax2.set_ylabel('Y', fontsize=14)
    #ax2.set_title('All Colors - Filled Areas', fontsize=16, fontweight='bold', pad=20)
    ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=False, ncol=2)
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.suptitle('Color Palette Examples', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.close()
    
    return fig

# ============================================================================
# INSTANCE CREATION
# ============================================================================

# Create global instance for easy access
lo = ColorPalette()