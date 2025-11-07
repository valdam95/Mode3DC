"""
Load Signal Analyser Tool

Interactive tool for analyzing load cell signals from mechanical tests.

WHAT IT DOES:
- Opens load-displacement data from Parquet file
- Displays interactive plot with AFN dropdown selector
- Allows manual annotation of key features:
  * Offset region for baseline correction
  * Linear loading region for stiffness calculation
  * Peak force location
- Automatically saves annotations back to Parquet file

HOW TO USE:
1. Run from terminal: python load_signal_analyser.py <parquet_path>
2. Select AFN from dropdown to view that specimen's data
3. Press 'D' to enable/disable drawing mode
4. Press 'O' to mark offset region (2 left-clicks, right-click to finish)
5. Press 'L' to mark linear region (2 left-clicks, right-click to finish)
6. Press 'P' to mark peak point (1 left-click, right-click to finish)
7. Annotations are automatically saved to Parquet file

OUTPUT:
- Updates the input Parquet file with:
  * offset_region: [x1, x2] coordinates
  * P_off: average force in offset region
  * start_lin: (x, y) start of linear region
  * end_lin: (x, y) end of linear region
  * k: global stiffness (N/mm)
  * F_dot: global force rate (N/s), calculated from k and actuator speed v
  * P_c: (x, y) peak force coordinates

FEATURES:
- AFN dropdown selector
- Drawing mode toggle to prevent accidental clicks
- Visual feedback with colored markers and lines
- Find closest point functionality
- Automatic calculations (offset, stiffness)
- Auto-save after each annotation
- Display existing annotations if already present

Author: Generated for NotchedBeam project
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import numpy as np
import pandas as pd
import os
import sys
import argparse
import layout as lo

# Try to import visualisation module (optional)
try:
    import visualisation as vis
    # Setup visualization styling if available
    vis.setup_visualization_style()
except ImportError:
    # visualisation module not available - continue without it
    pass

def main():
    """Main function to run the load signal analyser."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Interactive load signal analysis tool')
    parser.add_argument('parquet_path', type=str, help='Path to Parquet file with load data')
    args = parser.parse_args()
    
    parquet_path = args.parquet_path
    
    # Check if file exists
    if not os.path.exists(parquet_path):
        print(f"Error: File not found: {parquet_path}")
        sys.exit(1)
    
    # Check if it's a Parquet file
    if not parquet_path.endswith('.parquet'):
        print(f"Error: File must be a .parquet file")
        sys.exit(1)
    
    print(f"Loading data from: {parquet_path}")
    
    # Check if analysis file exists and load from that instead
    base_name = os.path.basename(parquet_path)
    prefix = base_name.split('_')[0]
    analysis_path = os.path.join(os.path.dirname(parquet_path), f"{prefix}_load_signal_analysis.parquet")
    
    if os.path.exists(analysis_path):
        print(f"Found existing analysis file: {analysis_path}")
        try:
            df = pd.read_parquet(analysis_path, engine='fastparquet')
            print(f"Loaded analysis data: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"Error loading analysis file: {e}")
            sys.exit(1)
    else:
        print(f"No analysis file found, loading raw data: {parquet_path}")
        # Load Parquet file
        try:
            df = pd.read_parquet(parquet_path, engine='fastparquet')
            print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"Error loading Parquet file: {e}")
            sys.exit(1)
    
    # Check required columns
    required_columns = ['AFN', 'w_data', 'P_data']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        sys.exit(1)
    
    # Initialize analysis columns if they don't exist
    analysis_columns = ['offset_region', 'P_off', 'start_lin', 'end_lin', 'k', 'P_c', 'F_dot']
    for col in analysis_columns:
        if col not in df.columns:
            df[col] = None
            print(f"Initialized analysis column: {col}")
    
    
    # Get list of AFNs (filter out NA values)
    afn_list = sorted(df['AFN'].dropna().unique())
    
    if not afn_list:
        print("Error: No valid AFN values found in data")
        sys.exit(1)
    
    print(f"Found {len(afn_list)} specimens with AFNs: {afn_list[:5]}{'...' if len(afn_list) > 5 else ''}")
    print("\nStarting interactive load signal analyser...")
    print("Controls:")
    print("  D - Toggle drawing mode ON/OFF")
    print("  O - Start offset region selection")
    print("  L - Start linear region selection")
    print("  P - Start peak selection")
    print("  ESC - Exit all modes without saving")
    print("  ← → - Navigate between AFNs")
    print("  Left-click - Select point")
    print("  Right-click - Finish current selection")
    
    # Start with first AFN
    current_afn_index = 0
    
    # Create the interactive plot
    create_interactive_plot(df, afn_list, current_afn_index, parquet_path)

def create_interactive_plot(df, afn_list, afn_index, parquet_path):
    """
    Create interactive plot with AFN dropdown selector and mode indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with load data
    afn_list : list
        List of available AFNs
    afn_index : int
        Index of current AFN in afn_list
    parquet_path : str
        Path to Parquet file for saving
    """
    
    # State variables
    state = {
        'current_afn_index': afn_index,
        'drawing_mode': False,
        'selection_mode': None,  # None, 'O', 'L', or 'P'
        'df': df,
        'parquet_path': parquet_path
    }
    
    # Get current AFN
    current_afn = afn_list[state['current_afn_index']]
    
    # Get data for current AFN
    row = df[df['AFN'] == current_afn].iloc[0]
    w = row['w_data']  # distance
    P = row['P_data']  # force
    
    # Check if data is valid
    if w is None or P is None or len(w) == 0 or len(P) == 0:
        print(f"Error: No valid data for AFN {current_afn}")
        return
    
    # Create figure with extra space at bottom for UI elements
    fig = plt.figure(figsize=(12, 9.5), dpi=120)
    
    # Create main plot area (leave space at bottom for controls and left for y-axis label)
    ax = plt.axes([0.15, 0.23, 0.8, 0.72])
    
    # Explicitly set linear scale for both axes and lock it
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    
    # Plot the load-displacement curve
    line, = ax.plot(w, P, color=lo.COLORS['indigo'], alpha=1.0, linewidth=2.0, label='Load Signal')
    
    # Set labels and title
    ax.set_xlabel(r'Distance (mm) $\longrightarrow$', fontsize=20, labelpad=10)
    ax.set_ylabel(r'Force (N) $\longrightarrow$', fontsize=20, labelpad=10)
    ax.set_title(f'Load Signal Analysis - AFN {current_afn}', fontsize=20, pad=20)
    
    # Set tick label sizes and padding
    ax.tick_params(axis='both', which='major', labelsize=20, 
                   pad=10, direction='in', length=6)
    
    # Set custom tick formatting for y-axis with 3 decimal places
    from matplotlib.ticker import FuncFormatter
    def format_y_ticks(value, pos):
        return f'{value:.3f}'
    ax.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))
    
    # Set closer tick spacing for y-axis
    from matplotlib.ticker import MultipleLocator
    # Get current y-axis range to determine appropriate tick spacing
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    # Use approximately 8-10 ticks across the range
    tick_spacing = y_range / 8
    ax.yaxis.set_major_locator(MultipleLocator(tick_spacing))
    
    # Show all four spines (complete frame)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    
    # Remove grid
    ax.grid(False)
    
    # Add analysis values display in upper left corner
    analysis_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                           fontsize=20, verticalalignment='top')
    
    # Create UI control boxes at bottom
    # Row 1: Mode indicator boxes (more compact, single line)
    box_height = 0.05
    box_y = 0.08
    
    # Drawing mode indicator box
    box_d = plt.axes([0.14, box_y, 0.16, box_height])
    box_d.text(0.5, 0.5, "'D' Drawing", ha='center', va='center', fontsize=20)
    box_d.set_xlim(0, 1)
    box_d.set_ylim(0, 1)
    box_d.axis('off')
    box_d_bg = plt.Rectangle((0, 0), 1, 1, facecolor='lightgray', alpha=0.4, edgecolor='black', linewidth=2)
    box_d.add_patch(box_d_bg)
    
    # Offset mode indicator box
    box_o = plt.axes([0.32, box_y, 0.15, box_height])
    box_o.text(0.5, 0.5, "'O' Offset", ha='center', va='center', fontsize=20)
    box_o.set_xlim(0, 1)
    box_o.set_ylim(0, 1)
    box_o.axis('off')
    box_o_bg = plt.Rectangle((0, 0), 1, 1, facecolor='lightgray', alpha=0.4, edgecolor='black', linewidth=2)
    box_o.add_patch(box_o_bg)
    
    # Linear mode indicator box
    box_l = plt.axes([0.50, box_y, 0.15, box_height])
    box_l.text(0.5, 0.5, "'L' Linear", ha='center', va='center', fontsize=20)
    box_l.set_xlim(0, 1)
    box_l.set_ylim(0, 1)
    box_l.axis('off')
    box_l_bg = plt.Rectangle((0, 0), 1, 1, facecolor='lightgray', alpha=0.4, edgecolor='black', linewidth=2)
    box_l.add_patch(box_l_bg)
    
    # Peak mode indicator box
    box_p = plt.axes([0.68, box_y, 0.15, box_height])
    box_p.text(0.5, 0.5, "'P' Peak", ha='center', va='center', fontsize=20)
    box_p.set_xlim(0, 1)
    box_p.set_ylim(0, 1)
    box_p.axis('off')
    box_p_bg = plt.Rectangle((0, 0), 1, 1, facecolor='lightgray', alpha=0.4, edgecolor='black', linewidth=2)
    box_p.add_patch(box_p_bg)
    
    # Row 2: AFN selector and navigation
    
    # AFN input box
    afn_input_ax = plt.axes([0.35, 0.01, 0.1, 0.05])
    afn_textbox = TextBox(afn_input_ax, 'AFN:  ', initial=str(current_afn), 
                          textalignment='center')
    afn_textbox.label.set_fontsize(20)
    afn_textbox.text_disp.set_fontsize(20)
    
    # Navigation instructions
    nav_instr = plt.axes([0.49, 0.01, 0.21, 0.06])
    nav_instr.text(0.5, 0.5, 'or use ← → arrows', 
                   ha='center', va='center', fontsize=20)
    nav_instr.set_xlim(0, 1)
    nav_instr.set_ylim(0, 1)
    nav_instr.axis('off')
    
    # Store UI elements in state for updating
    state['ui_elements'] = {
        'fig': fig,
        'ax': ax,
        'line': line,
        'analysis_text': analysis_text,
        'box_d_bg': box_d_bg,
        'box_o_bg': box_o_bg,
        'box_l_bg': box_l_bg,
        'box_p_bg': box_p_bg,
        'afn_textbox': afn_textbox,
        'afn_list': afn_list,
        'w_data': w,
        'P_data': P
    }
    
    # Connect event handlers
    # Note: Our key_press handler is connected AFTER matplotlib's default handlers,
    # so we need to prevent matplotlib's 'l' key from toggling log scale
    def key_press_wrapper(event):
        # Block matplotlib's default 'l' key (log scale toggle)
        if event.key and event.key.lower() == 'l':
            # Force linear scale if 'l' was pressed
            ax.set_yscale('linear')
            fig.canvas.draw_idle()
        # Call our custom handler
        on_key_press(event, state)
    
    fig.canvas.mpl_connect('key_press_event', key_press_wrapper)
    fig.canvas.mpl_connect('button_press_event', lambda event: on_mouse_click(event, state))
    fig.canvas.mpl_connect('close_event', lambda event: save_temp_data_to_parquet(state))
    afn_textbox.on_submit(lambda text: on_afn_input(text, state))
    
    # Update UI colors based on current state
    update_ui_colors(state)
    
    # Update analysis values display initially
    update_analysis_display(state)
    
    # Plot existing offset data if available
    plot_existing_offset_data(state)
    
    # Plot existing linear data if available
    plot_existing_linear_data(state)
    
    # Plot existing peak data if available
    plot_existing_peak_data(state)
    
    plt.show()
    
    print(f"\nDisplaying AFN {current_afn} ({afn_index + 1}/{len(afn_list)})")

def on_key_press(event, state):
    """Handle keyboard events for mode selection and AFN navigation."""
    
    # Escape key - reset everything and reload Parquet
    if event.key == 'escape':
        print("Escape pressed - resetting all modes and reloading data...")
        
        # Reset all modes
        state['drawing_mode'] = False
        state['selection_mode'] = None
        
        # Clear any temporary selection data
        if 'offset_clicks' in state:
            state['offset_clicks'] = []
        if 'offset_lines' in state:
            for line in state['offset_lines']:
                line.remove()
            state['offset_lines'] = []
        if 'linear_clicks' in state:
            state['linear_clicks'] = []
        if 'linear_lines' in state:
            for line in state['linear_lines']:
                line.remove()
            state['linear_lines'] = []
        if 'peak_click' in state:
            state['peak_click'] = None
        if 'peak_line' in state and state['peak_line'] is not None:
            state['peak_line'].remove()
            state['peak_line'] = None
        
        # Reload Parquet file (try analysis file first, fallback to original)
        try:
            parquet_path = state['parquet_path']
            # Extract prefix from filename (e.g., ON4PB from ON4PB_raw.parquet)
            base_name = os.path.basename(parquet_path)
            prefix = base_name.split('_')[0]  # Get part before first underscore
            analysis_path = os.path.join(os.path.dirname(parquet_path), f"{prefix}_load_signal_analysis.parquet")
            
            if os.path.exists(analysis_path):
                df = pd.read_parquet(analysis_path, engine='fastparquet')
                print(f"Reloaded analysis data from: {analysis_path}")
            else:
                df = pd.read_parquet(parquet_path, engine='fastparquet')
                print(f"Reloaded original data from: {parquet_path}")
            
            state['df'] = df
        except Exception as e:
            print(f"Error reloading Parquet file: {e}")
        
        # Update UI and reload current AFN
        update_ui_colors(state)
        reload_afn(state)
        
        print("Reset complete - all modes OFF")
        return
    
    # Mode toggle keys
    if event.key == 'd' or event.key == 'D':
        state['drawing_mode'] = not state['drawing_mode']
        print(f"Drawing mode: {'ON' if state['drawing_mode'] else 'OFF'}")
        update_ui_colors(state)
        
    elif event.key == 'o' or event.key == 'O':
        if state['drawing_mode']:
            # Clear any existing offset data
            if hasattr(state, 'offset_clicks'):
                delattr(state, 'offset_clicks')
            if hasattr(state, 'offset_lines'):
                for line in state['offset_lines']:
                    line.remove()
                state['offset_lines'] = []
            state['selection_mode'] = 'O'
            print("Offset selection: ON - Click two x-coordinates, then right-click to finish")
            update_ui_colors(state)
        else:
            print("Drawing mode must be ON. Press 'D' first.")
    
    elif event.key == 'l' or event.key == 'L':
        if state['drawing_mode']:
            # Clear any existing linear data
            if 'linear_clicks' in state:
                state['linear_clicks'] = []
            if 'linear_lines' in state:
                for line in state['linear_lines']:
                    line.remove()
                state['linear_lines'] = []
            state['selection_mode'] = 'L'
            print("Linear selection: ON - Click two points for linear region, then right-click to finish")
            update_ui_colors(state)
        else:
            print("Drawing mode must be ON. Press 'D' first.")
    
    elif event.key == 'p' or event.key == 'P':
        if state['drawing_mode']:
            # Clear any existing peak data
            if 'peak_click' in state:
                state['peak_click'] = None
            if 'peak_line' in state and state['peak_line'] is not None:
                state['peak_line'].remove()
                state['peak_line'] = None
            state['selection_mode'] = 'P'
            print("Peak selection: ON - Click one point for peak value, then right-click to finish")
            update_ui_colors(state)
        else:
            print("Drawing mode must be ON. Press 'D' first.")
        
    # AFN navigation with arrow keys
    elif event.key == 'left':
        if state['current_afn_index'] > 0:
            state['current_afn_index'] -= 1
            reload_afn(state)
        else:
            print("Already at first AFN")
            
    elif event.key == 'right':
        if state['current_afn_index'] < len(state['ui_elements']['afn_list']) - 1:
            state['current_afn_index'] += 1
            reload_afn(state)
        else:
            print("Already at last AFN")

def on_mouse_click(event, state):
    """Handle mouse click events for point selection."""
    print(f"Mouse click: button={event.button}, drawing_mode={state['drawing_mode']}, selection_mode={state.get('selection_mode')}")
    
    # Only handle clicks if drawing mode is ON and we're in the main plot area
    if not state['drawing_mode'] or event.inaxes != state['ui_elements']['ax']:
        print("Click ignored: drawing mode OFF or not in plot area")
        return
    
    # Handle right clicks for finishing selections
    if event.button == 3:  # Right click
        print(f"Right click detected. selection_mode={state.get('selection_mode')}, has_offset_clicks={'offset_clicks' in state}")
        if 'offset_clicks' in state:
            print(f"offset_clicks length: {len(state['offset_clicks'])}")
        if state['selection_mode'] == 'O' and 'offset_clicks' in state and len(state['offset_clicks']) == 2:
            # Finish offset selection
            x_start, x_end = sorted(state['offset_clicks'])
            
            # Get current data
            w = state['ui_elements']['w_data']
            P = state['ui_elements']['P_data']
            
            # Find data points in this range and calculate average force
            mask = (np.array(w) >= x_start) & (np.array(w) <= x_end)
            if np.any(mask):
                P_off = np.mean(np.array(P)[mask])
                
                # Save directly to Parquet
                afn_list = state['ui_elements']['afn_list']
                current_afn = afn_list[state['current_afn_index']]
                
                # Update DataFrame
                df = state['df']
                row_idx = df[df['AFN'] == current_afn].index[0]
                df.at[row_idx, 'offset_region'] = [x_start, x_end]
                df.at[row_idx, 'P_off'] = P_off
                
                # Ensure P_off column has consistent data type (float64)
                df['P_off'] = pd.to_numeric(df['P_off'], errors='coerce')
                
                # Save to separate analysis Parquet file
                parquet_path = state['parquet_path']
                # Extract prefix from filename (e.g., ON4PB from ON4PB_raw.parquet)
                base_name = os.path.basename(parquet_path)
                prefix = base_name.split('_')[0]  # Get part before first underscore
                analysis_path = os.path.join(os.path.dirname(parquet_path), f"{prefix}_load_signal_analysis.parquet")
                df.to_parquet(analysis_path, engine='fastparquet', index=False)
                
                print(f"Offset selection completed and saved: P_off = {P_off:.2f} N")
                print(f"Saved offset_region: [{x_start:.2f}, {x_end:.2f}] mm")
                
                # Clear selection and visual feedback
                state['offset_clicks'] = []
                state['selection_mode'] = None
                update_ui_colors(state)
                
                # Clear temporary vertical lines
                if 'offset_lines' in state:
                    for line in state['offset_lines']:
                        line.remove()
                    state['offset_lines'] = []
                
                # Reload the current AFN to show updated data
                reload_afn(state)
                
            else:
                print("No data points found in selected range")
                state['offset_clicks'] = []
                state['selection_mode'] = None
                update_ui_colors(state)
        
        # Handle linear selection right-click
        elif state['selection_mode'] == 'L' and 'linear_clicks' in state and len(state['linear_clicks']) == 2:
            # Finish linear selection
            start_lin = state['linear_clicks'][0]
            end_lin = state['linear_clicks'][1]
            
            # Calculate global stiffness k (slope in N/mm)
            x1, y1 = start_lin
            x2, y2 = end_lin
            k = (y2 - y1) / (x2 - x1)  # N/mm
            
            # Save directly to Parquet
            afn_list = state['ui_elements']['afn_list']
            current_afn = afn_list[state['current_afn_index']]
            
            # Update DataFrame
            df = state['df']
            row_idx = df[df['AFN'] == current_afn].index[0]
            
            # Ensure columns exist before assigning values
            if 'start_lin' not in df.columns:
                df['start_lin'] = None
            if 'end_lin' not in df.columns:
                df['end_lin'] = None
            if 'k' not in df.columns:
                df['k'] = None
            
            df.at[row_idx, 'start_lin'] = [x1, y1]
            df.at[row_idx, 'end_lin'] = [x2, y2]
            df.at[row_idx, 'k'] = k
            
            # Calculate F_dot (global force rate) if v (actuator speed) is available
            # F_dot [N/s] = k [N/mm] * v [µm/s] * 1e-3 [mm/µm]
            if 'v' in df.columns and not pd.isna(df.at[row_idx, 'v']):
                v = df.at[row_idx, 'v']  # Actuator speed in µm/s
                F_dot = 1e-3 * k * v  # Convert µm/s to mm/s: 1e-3 mm/µm
                df.at[row_idx, 'F_dot'] = F_dot
                print(f"Calculated F_dot: {F_dot:.2f} N/s (k={k:.2f} N/mm, v={v:.2f} µm/s)")
            
            # Ensure k column has consistent data type (float64)
            df['k'] = pd.to_numeric(df['k'], errors='coerce')
            
            # Ensure F_dot column has consistent data type (float64)
            if 'F_dot' in df.columns:
                df['F_dot'] = pd.to_numeric(df['F_dot'], errors='coerce')
            
            # Save to separate analysis Parquet file
            parquet_path = state['parquet_path']
            # Extract prefix from filename (e.g., ON4PB from ON4PB_raw.parquet)
            base_name = os.path.basename(parquet_path)
            prefix = base_name.split('_')[0]  # Get part before first underscore
            analysis_path = os.path.join(os.path.dirname(parquet_path), f"{prefix}_load_signal_analysis.parquet")
            df.to_parquet(analysis_path, engine='fastparquet', index=False)
            
            print(f"Linear selection completed and saved: k = {k:.2f} N/mm")
            print(f"Saved start_lin: ({x1:.2f}, {y1:.2f})")
            print(f"Saved end_lin: ({x2:.2f}, {y2:.2f})")
            
            # Clear selection and visual feedback
            state['linear_clicks'] = []
            state['selection_mode'] = None
            update_ui_colors(state)
            
            # Clear temporary cross markers
            if 'linear_lines' in state:
                for line in state['linear_lines']:
                    line.remove()
                state['linear_lines'] = []
            
            # Reload the current AFN to show updated data
            reload_afn(state)
        
        # Handle peak selection right-click
        elif state['selection_mode'] == 'P' and 'peak_click' in state and state['peak_click'] is not None:
            # Finish peak selection
            P_c = state['peak_click']
            
            # Save directly to Parquet
            afn_list = state['ui_elements']['afn_list']
            current_afn = afn_list[state['current_afn_index']]
            
            # Update DataFrame
            df = state['df']
            row_idx = df[df['AFN'] == current_afn].index[0]
            
            # Ensure P_c column exists before assigning values
            if 'P_c' not in df.columns:
                df['P_c'] = None
            
            df.at[row_idx, 'P_c'] = [P_c[0], P_c[1]]
            
            # Note: P_c is a list, so no pd.to_numeric conversion needed
            
            # Save to separate analysis Parquet file
            parquet_path = state['parquet_path']
            # Extract prefix from filename (e.g., ON4PB from ON4PB_raw.parquet)
            base_name = os.path.basename(parquet_path)
            prefix = base_name.split('_')[0]  # Get part before first underscore
            analysis_path = os.path.join(os.path.dirname(parquet_path), f"{prefix}_load_signal_analysis.parquet")
            df.to_parquet(analysis_path, engine='fastparquet', index=False)
            
            print(f"Peak selection completed and saved: P_c = ({P_c[0]:.2f}, {P_c[1]:.2f})")
            
            # Clear selection and visual feedback
            state['peak_click'] = None
            state['selection_mode'] = None
            update_ui_colors(state)
            
            # Clear temporary circle marker
            if 'peak_line' in state and state['peak_line'] is not None:
                state['peak_line'].remove()
                state['peak_line'] = None
            
            # Update state['df'] to ensure reload_afn sees the new data
            state['df'] = df
            
            # Reload the current AFN to show updated data
            reload_afn(state)
        
        return
    
    # Only handle left clicks for point selection
    if event.button != 1:  # Left click
        return
    
    # Handle offset selection mode
    if state['selection_mode'] == 'O':
        # Check if we already have 2 clicks
        if 'offset_clicks' in state and len(state['offset_clicks']) >= 2:
            print("Already have 2 clicks. Right-click to finish selection.")
            return
            
        print("Processing offset click...")
        # Get current data
        w = state['ui_elements']['w_data']
        P = state['ui_elements']['P_data']
        
        if w is None or P is None or len(w) == 0 or len(P) == 0:
            print("No valid data found")
            return
        
        # Find closest data point by x-coordinate only
        x_distances = np.abs(np.array(w) - event.xdata)
        closest_idx = np.argmin(x_distances)
        closest_x = w[closest_idx]
        
        # Initialize offset clicks if not exists
        if 'offset_clicks' not in state:
            state['offset_clicks'] = []
            state['offset_lines'] = []
            print("Initialized offset clicks")
        
        state['offset_clicks'].append(closest_x)
        click_num = len(state['offset_clicks'])
        print(f"Click {click_num}: x = {closest_x:.2f} mm")
        
        # Draw vertical line at closest x-coordinate
        ax = state['ui_elements']['ax']
        line = ax.axvline(x=closest_x, color=lo.COLORS['orange'], linewidth=2, alpha=0.7)
        state['offset_lines'].append(line)
        
        # Force immediate redraw
        state['ui_elements']['fig'].canvas.draw()
        
        if click_num == 2:
            print("Two points selected. Right-click to finish offset selection.")
        else:
            print(f"Need {2 - click_num} more click(s).")
    
    # Handle linear selection mode
    elif state['selection_mode'] == 'L':
        # Check if we already have 2 clicks
        if 'linear_clicks' in state and len(state['linear_clicks']) >= 2:
            print("Already have 2 clicks. Right-click to finish selection.")
            return
            
        print("Processing linear click...")
        # Get current data
        w = state['ui_elements']['w_data']
        P = state['ui_elements']['P_data']
        
        if w is None or P is None or len(w) == 0 or len(P) == 0:
            print("No valid data found")
            return
        
        # Find closest data point by both x and y coordinates
        # Use weighted distance with more emphasis on x-coordinate for better accuracy
        x_distances = np.abs(np.array(w) - event.xdata)
        y_distances = np.abs(np.array(P) - event.ydata)
        # Weight x-distance more heavily (2x) since it's usually more important for linear selection
        total_distances = np.sqrt((2 * x_distances)**2 + y_distances**2)
        closest_idx = np.argmin(total_distances)
        closest_x = w[closest_idx]
        closest_y = P[closest_idx]
        
        # Initialize linear clicks if not exists
        if 'linear_clicks' not in state:
            state['linear_clicks'] = []
            state['linear_lines'] = []
            print("Initialized linear clicks")
        
        state['linear_clicks'].append((closest_x, closest_y))
        click_num = len(state['linear_clicks'])
        print(f"Click {click_num}: ({closest_x:.2f}, {closest_y:.2f})")
        
        # Draw cross marker at closest point
        ax = state['ui_elements']['ax']
        line = ax.plot(closest_x, closest_y, 'x', color=lo.COLORS['orange'], markersize=12, markeredgewidth=3, alpha=0.8)[0]
        state['linear_lines'].append(line)
        
        if click_num == 2:
            print("Two points selected. Right-click to finish linear selection.")
            # Draw preview line between the two points
            x1, y1 = state['linear_clicks'][0]
            x2, y2 = state['linear_clicks'][1]
            preview_line = ax.plot([x1, x2], [y1, y2], color=lo.COLORS['orange'], linewidth=2, alpha=0.5, linestyle='-')[0]
            state['linear_lines'].append(preview_line)
        
        # Force immediate redraw
        state['ui_elements']['fig'].canvas.draw()
        
        if click_num != 2:
            print(f"Need {2 - click_num} more click(s).")
    
    # Handle peak selection mode
    elif state['selection_mode'] == 'P':
        print("Processing peak click...")
        # Get current data
        w = state['ui_elements']['w_data']
        P = state['ui_elements']['P_data']
        
        if w is None or P is None or len(w) == 0 or len(P) == 0:
            print("No valid data found")
            return
        
        # Find closest data point by both x and y coordinates
        x_distances = np.abs(np.array(w) - event.xdata)
        y_distances = np.abs(np.array(P) - event.ydata)
        # Weight x-distance more heavily (2x) since it's usually more important for peak selection
        total_distances = np.sqrt((2 * x_distances)**2 + y_distances**2)
        closest_idx = np.argmin(total_distances)
        closest_x = w[closest_idx]
        closest_y = P[closest_idx]
        
        # Store peak click (update if already exists)
        state['peak_click'] = (closest_x, closest_y)
        print(f"Peak point: ({closest_x:.2f}, {closest_y:.2f})")
        
        # Remove previous circle marker if it exists
        if 'peak_line' in state and state['peak_line'] is not None:
            state['peak_line'].remove()
        
        # Draw new circle marker at closest point (hollow during selection)
        ax = state['ui_elements']['ax']
        line = ax.plot(closest_x, closest_y, 'o', color=lo.COLORS['orange'], markersize=10, 
                      markeredgewidth=2, alpha=0.8, markerfacecolor='none')[0]
        state['peak_line'] = line
        
        # Force immediate redraw
        state['ui_elements']['fig'].canvas.draw()
        
        print("Peak point updated. Right-click to finish peak selection.")
    
    else:
        print(f"Not in selection mode. Current selection_mode: {state.get('selection_mode')}")

def on_afn_input(text, state):
    """Handle AFN input from text box."""
    try:
        afn = int(text)
        afn_list = state['ui_elements']['afn_list']
        
        if afn in afn_list:
            state['current_afn_index'] = afn_list.index(afn)
            reload_afn(state)
        else:
            print(f"AFN {afn} not found in data. Available AFNs: {min(afn_list)} to {max(afn_list)}")
    except ValueError:
        print(f"Invalid AFN input: {text}. Please enter a number.")

def reload_afn(state):
    """Reload plot with new AFN."""
    afn_list = state['ui_elements']['afn_list']
    current_afn = afn_list[state['current_afn_index']]
    
    # Get data for new AFN
    row = state['df'][state['df']['AFN'] == current_afn].iloc[0]
    w = row['w_data']
    P = row['P_data']
    
    if w is None or P is None or len(w) == 0 or len(P) == 0:
        print(f"Error: No valid data for AFN {current_afn}")
        return
    
    # Update plot
    ui = state['ui_elements']
    ui['line'].set_xdata(w)
    ui['line'].set_ydata(P)
    ui['w_data'] = w
    ui['P_data'] = P
    
    # Update title
    ui['ax'].set_title(f'Load Signal Analysis - AFN {current_afn}', 
                       fontsize=20, pad=20)
    
    # AFN display is handled by the text box
    
    # Update text box
    ui['afn_textbox'].set_val(str(current_afn))
    
    # Set axis limits to show full data range
    w = state['ui_elements']['w_data']
    P = state['ui_elements']['P_data']
    
    if w is not None and P is not None and len(w) > 0 and len(P) > 0:
        # Calculate data ranges with small padding
        x_min, x_max = min(w), max(w)
        y_min, y_max = min(P), max(P)
        
        # Add 2% padding to the ranges
        x_padding = (x_max - x_min) * 0.02
        y_padding = (y_max - y_min) * 0.02
        
        # Set axis limits, ensuring positive values for y-axis
        ui['ax'].set_xlim(x_min - x_padding, x_max + x_padding)
        
        # Ensure y_min with padding is positive (at least 0.001 for safety)
        y_min_with_padding = max(0.001, y_min - y_padding)
        ui['ax'].set_ylim(y_min_with_padding, y_max + y_padding)
        
        # Update y-axis tick formatting for the new range
        from matplotlib.ticker import MaxNLocator
        ui['ax'].yaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
    
    # Update analysis values display
    update_analysis_display(state)
    
    # Plot existing offset data if available
    plot_existing_offset_data(state)
    
    # Plot existing linear data if available
    plot_existing_linear_data(state)
    
    # Plot existing peak data if available
    plot_existing_peak_data(state)
    
    # Redraw
    ui['fig'].canvas.draw_idle()
    
    print(f"Switched to AFN {current_afn} ({state['current_afn_index'] + 1}/{len(afn_list)})")

def plot_existing_offset_data(state):
    """Plot existing offset data (vertical lines and P_off horizontal line)."""
    try:
        df = state['df']
        afn_list = state['ui_elements']['afn_list']
        current_afn = afn_list[state['current_afn_index']]
        row = df[df['AFN'] == current_afn].iloc[0]
        
        ax = state['ui_elements']['ax']
        
        # Clear any existing offset visualization lines
        if 'offset_visualization_lines' in state:
            for line in state['offset_visualization_lines']:
                line.remove()
            state['offset_visualization_lines'] = []
        else:
            state['offset_visualization_lines'] = []
        
        # Plot offset region vertical lines and P_off
        offset_region = row['offset_region']
        p_off = row['P_off']
        
        if offset_region is not None and not (isinstance(offset_region, float) and pd.isna(offset_region)):
            if isinstance(offset_region, list) and len(offset_region) == 2:
                x_start, x_end = offset_region
                # Vertical lines for offset region
                line1 = ax.axvline(x=x_start, color=lo.COLORS['orange'], linewidth=2, alpha=0.8, linestyle='--')
                line2 = ax.axvline(x=x_end, color=lo.COLORS['orange'], linewidth=2, alpha=0.8, linestyle='--')
                state['offset_visualization_lines'].extend([line1, line2])
                
                # Plot P_off horizontal line spanning only the offset region
                if p_off is not None and not pd.isna(p_off):
                    # Horizontal line for P_off spanning only the offset region
                    line3 = ax.plot([x_start, x_end], [p_off, p_off], color=lo.COLORS['orange'], linewidth=2, alpha=0.8, linestyle='--')[0]
                    state['offset_visualization_lines'].append(line3)
                    
                    # Add text label for P_off value in the middle of offset region
                    x_middle = (x_start + x_end) / 2
                    text = ax.text(x_middle, p_off, f'P_off = {p_off:.2f} N', 
                                  color=lo.COLORS['orange'], fontsize=20,
                                  ha='center', va='bottom')
                    state['offset_visualization_lines'].append(text)
        
        # Redraw the plot
        state['ui_elements']['fig'].canvas.draw()
        
    except Exception as e:
        print(f"Error plotting existing offset data: {e}")

def plot_existing_linear_data(state):
    """Plot existing linear data (crosses and line with k value)."""
    try:
        df = state['df']
        afn_list = state['ui_elements']['afn_list']
        current_afn = afn_list[state['current_afn_index']]
        row = df[df['AFN'] == current_afn].iloc[0]
        
        ax = state['ui_elements']['ax']
        
        # Clear any existing linear visualization lines
        if 'linear_visualization_lines' in state:
            for line in state['linear_visualization_lines']:
                line.remove()
            state['linear_visualization_lines'] = []
        else:
            state['linear_visualization_lines'] = []
        
        # Plot linear data if available
        start_lin = row['start_lin']
        end_lin = row['end_lin']
        k = row['k']
        
        if (start_lin is not None and not (isinstance(start_lin, float) and pd.isna(start_lin)) and
            end_lin is not None and not (isinstance(end_lin, float) and pd.isna(end_lin)) and
            k is not None and not pd.isna(k)):
            
            if (isinstance(start_lin, list) and len(start_lin) == 2 and
                isinstance(end_lin, list) and len(end_lin) == 2):
                
                x1, y1 = start_lin
                x2, y2 = end_lin
                
                # Plot cross markers for start and end points
                cross1 = ax.plot(x1, y1, 'x', color=lo.COLORS['orange'], markersize=12, markeredgewidth=3, alpha=0.8)[0]
                cross2 = ax.plot(x2, y2, 'x', color=lo.COLORS['orange'], markersize=12, markeredgewidth=3, alpha=0.8)[0]
                state['linear_visualization_lines'].extend([cross1, cross2])
                
                # Plot line between the two points
                line = ax.plot([x1, x2], [y1, y2], color=lo.COLORS['orange'], linewidth=2, alpha=0.8, linestyle='--')[0]
                state['linear_visualization_lines'].append(line)
                
                # Add text label for k value in the middle of the line
                x_middle = (x1 + x2) / 2
                y_middle = (y1 + y2) / 2
                text = ax.text(x_middle, y_middle, f'k = {k:.2f} N/mm', 
                              color=lo.COLORS['orange'], fontsize=20,
                              ha='center', va='center',
                              bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8, edgecolor='none'))
                state['linear_visualization_lines'].append(text)
        
        # Redraw the plot
        state['ui_elements']['fig'].canvas.draw()
        
    except Exception as e:
        print(f"Error plotting existing linear data: {e}")

def plot_existing_peak_data(state):
    """Plot existing peak data (filled circle)."""
    try:
        df = state['df']
        afn_list = state['ui_elements']['afn_list']
        current_afn = afn_list[state['current_afn_index']]
        row = df[df['AFN'] == current_afn].iloc[0]
        
        ax = state['ui_elements']['ax']
        
        # Clear any existing peak visualization lines
        if 'peak_visualization_line' in state and state['peak_visualization_line'] is not None:
            if isinstance(state['peak_visualization_line'], list):
                for item in state['peak_visualization_line']:
                    if item is not None:
                        item.remove()
            else:
                state['peak_visualization_line'].remove()
            state['peak_visualization_line'] = None
        
        # Plot peak data if available
        P_c = row['P_c']
        
        if P_c is not None and not (isinstance(P_c, float) and pd.isna(P_c)):
            if isinstance(P_c, list) and len(P_c) == 2:
                x, y = P_c
                
                # Plot filled circle for peak point
                circle = ax.plot(x, y, 'o', color=lo.COLORS['orange'], markersize=12, 
                               markeredgewidth=2, alpha=0.8, markerfacecolor=lo.COLORS['orange'])[0]
                state['peak_visualization_line'] = circle
                
                # Add text label for peak value above and to the left of the circle
                text = ax.text(x-0.005, y, f'P_c = ({x:.2f}, {y:.2f})', 
                              color=lo.COLORS['orange'], fontsize=20,
                              ha='right', va='bottom',
                              bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.5, edgecolor='none'))
                state['peak_visualization_line'] = [circle, text]
        
        # Redraw the plot
        state['ui_elements']['fig'].canvas.draw()
        
    except Exception as e:
        print(f"Error plotting existing peak data: {e}")

def update_analysis_display(state):
    """Update the analysis values display in the upper left corner."""
    try:
        # Get current AFN data
        afn_list = state['ui_elements']['afn_list']
        current_afn = afn_list[state['current_afn_index']]
        df = state['df']
        row = df[df['AFN'] == current_afn].iloc[0]
        
        # Build display text with current file info
        parquet_path = state['parquet_path']
        base_name = os.path.basename(parquet_path)
        prefix = base_name.split('_')[0]
        analysis_path = os.path.join(os.path.dirname(parquet_path), f"{prefix}_load_signal_analysis.parquet")
        
        # Check if analysis file exists and use it for display
        if os.path.exists(analysis_path):
            current_file = os.path.basename(analysis_path)
        else:
            current_file = os.path.basename(parquet_path)
            
        lines = [f'Current Values']
        
        # Offset region
        offset_region = row['offset_region']
        try:
            if offset_region is not None and not (isinstance(offset_region, float) and pd.isna(offset_region)):
                if isinstance(offset_region, list) and len(offset_region) == 2:
                    lines.append(f"offset_region: [{offset_region[0]:.2f}, {offset_region[1]:.2f}] mm")
                else:
                    lines.append(f"offset_region: {offset_region}")
            else:
                lines.append("offset_region: --")
        except Exception as e:
            print(f"Error processing offset_region: {e}, value: {offset_region}, type: {type(offset_region)}")
            lines.append("offset_region: --")
        
        # P_off
        p_off = row['P_off']
        if p_off is not None and not pd.isna(p_off):
            lines.append(f"P_off: {p_off:.2f} N")
        else:
            lines.append("P_off: --")
        
        # start_lin
        start_lin = row['start_lin']
        try:
            if start_lin is not None and not (isinstance(start_lin, float) and pd.isna(start_lin)):
                if isinstance(start_lin, (list, tuple)) and len(start_lin) == 2:
                    lines.append(f"start_lin: ({start_lin[0]:.2f}, {start_lin[1]:.2f})")
                else:
                    lines.append(f"start_lin: {start_lin}")
            else:
                lines.append("start_lin: --")
        except Exception as e:
            print(f"Error processing start_lin: {e}, value: {start_lin}, type: {type(start_lin)}")
            lines.append("start_lin: --")
        
        # end_lin
        end_lin = row['end_lin']
        try:
            if end_lin is not None and not (isinstance(end_lin, float) and pd.isna(end_lin)):
                if isinstance(end_lin, (list, tuple)) and len(end_lin) == 2:
                    lines.append(f"end_lin: ({end_lin[0]:.2f}, {end_lin[1]:.2f})")
                else:
                    lines.append(f"end_lin: {end_lin}")
            else:
                lines.append("end_lin: --")
        except Exception as e:
            print(f"Error processing end_lin: {e}, value: {end_lin}, type: {type(end_lin)}")
            lines.append("end_lin: --")
        
        # k (global stiffness)
        k = row['k']
        if k is not None and not pd.isna(k):
            lines.append(f"k: {k:.2f} N/mm")
        else:
            lines.append("k: --")
        
        # F_dot (global force rate)
        F_dot = row['F_dot'] if 'F_dot' in row.index else None
        if F_dot is not None and not pd.isna(F_dot):
            lines.append(f"F_dot: {F_dot:.2f} N/s")
        else:
            lines.append("F_dot: --")
        
        # P_c (peak force)
        P_c = row['P_c']
        try:
            if P_c is not None and not (isinstance(P_c, float) and pd.isna(P_c)):
                if isinstance(P_c, (list, tuple)) and len(P_c) == 2:
                    lines.append(f"P_c: ({P_c[0]:.2f}, {P_c[1]:.2f})")
                else:
                    lines.append(f"P_c: {P_c}")
            else:
                lines.append("P_c: --")
        except Exception as e:
            print(f"Error processing P_c: {e}, value: {P_c}, type: {type(P_c)}")
            lines.append("P_c: --")
        
        # Update the display
        ui = state['ui_elements']
        ui['analysis_text'].set_text('\n'.join(lines))
        
        # Add white box background
        ui['analysis_text'].set_bbox(dict(boxstyle="round,pad=0.1", facecolor='white', alpha=1, edgecolor='none'))
        
    except Exception as e:
        print(f"Error updating analysis display: {e}")

def save_temp_data_to_parquet(state):
    """Save temporary storage data to Parquet file when window closes."""
    try:
        if hasattr(state, 'temp_storage') and state['temp_storage']:
            df = state['df']
            parquet_path = state['parquet_path']
            
            # Update DataFrame with temporary storage data
            for afn, data in state['temp_storage'].items():
                row_idx = df[df['AFN'] == afn].index[0]
                
                if 'offset_region' in data:
                    df.at[row_idx, 'offset_region'] = data['offset_region']
                if 'P_off' in data:
                    df.at[row_idx, 'P_off'] = data['P_off']
            
            # Save to Parquet
            df.to_parquet(parquet_path, engine='fastparquet', index=False)
            print(f"Saved {len(state['temp_storage'])} AFN(s) to Parquet file")
            
    except Exception as e:
        print(f"Error saving temporary data: {e}")

# Visual feedback functions will be added later

def update_ui_colors(state):
    """Update UI box colors based on current mode states."""
    ui = state['ui_elements']
    
    # Drawing mode box
    if state['drawing_mode']:
        ui['box_d_bg'].set_facecolor(lo.COLORS['gold'])
    else:
        ui['box_d_bg'].set_facecolor('lightgray')
    
    # Offset mode box
    if state.get('selection_mode') == 'O':
        ui['box_o_bg'].set_facecolor(lo.COLORS['gold'])
    else:
        ui['box_o_bg'].set_facecolor('lightgray')
    
    # Linear mode box
    if state.get('selection_mode') == 'L':
        ui['box_l_bg'].set_facecolor(lo.COLORS['gold'])
    else:
        ui['box_l_bg'].set_facecolor('lightgray')
    
    # Peak mode box
    if state.get('selection_mode') == 'P':
        ui['box_p_bg'].set_facecolor(lo.COLORS['gold'])
    else:
        ui['box_p_bg'].set_facecolor('lightgray')
    
    # Redraw
    ui['fig'].canvas.draw_idle()

if __name__ == "__main__":
    main()

