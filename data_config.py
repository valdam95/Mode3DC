"""
Internal data configuration for Submission Notebook.

Cross-platform server path management for team data access.
"""

import os
import platform
import pandas as pd
import numpy as np
import datetime
import glob
import re
import math
import subprocess

# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

# SMB Server connection
SMB_SERVER_URL = 'smb://wsl.ch'

# Cross-platform server paths (check your operating system and update the paths)
SERVER_PATHS = {
    'macos': '/Volumes/fe/lawprae/LBI/Projects/210_Weak_Layer_Mechanics/Valle/01_experiments/2025_Mode_III_displacement_controlled',
    'linux': '',
    'windows': ''
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_server_path():
    """Get server path for current platform."""
    system = platform.system().lower()
    if system == 'darwin':
        return SERVER_PATHS['macos']
    elif system == 'linux':
        return SERVER_PATHS['linux']
    elif system == 'windows':
        return SERVER_PATHS['windows']
    else:
        raise ValueError(f"Unsupported platform: {system}")

def check_server_availability():
    """Check if server path is accessible."""
    return os.path.exists(get_server_path())

def get_smb_info():
    """Get SMB server information."""
    return {
        'url': SMB_SERVER_URL,
        'server': 'ismd-server.synology.me',
        'share': 'Public',
        'current_path': get_server_path(),
        'platform': platform.system()
    }

def get_data_path(maindir, subdir=None, *, base_path=None, experiment_id=None):
    """
    Get path for specific data type and experiment.
    
    Args:
        maindir (str): Main directory ('01_raw_data', '02_processed_data')
        subdir (str, optional): Subdirectory within the data type
        base_path (str, optional): Base path to use instead of the default server path
        experiment_id (str, optional): Experiment identifier (currently unused)
    
    Returns:
        str: Full path to the data directory
    """
    if base_path is None:
        base_path = get_server_path()

    else:
        base_path = base_path
        
    path = os.path.join(base_path, maindir)

    if subdir:
        path = os.path.join(path, subdir)

    # Non-fatal feedback if the final path does not exist yet
    if not os.path.exists(path):
        print(f"Note: Path does not exist yet, please check the exact name of the folder: {path}")

    return path

# ============================================================================
# DATA HANDLING FUNCTIONS
# ============================================================================

def import_split_and_save_field_data_excel(filepath, data_start_row=5, save_as=None, data_type_conversion=True, sheet_name=None, force_overwrite=False, exclude_columns=None):
    """
    Import Excel file and merge into master dataset files.
    
    This function creates or merges data into the master dataset files:
    - {save_as}.parquet (data)
    - {save_as}_info.parquet (metadata)
    
    If files already exist and force_overwrite=False, provides feedback and returns existing data.
    If force_overwrite=True, merges new columns into existing files.
    
    Expected Excel structure (first 5 rows as metadata):
    Row 0: Abbreviations (column headers)
    Row 1: Units
    Row 2: Data_Type
    Row 3: Long Name
    Row 4: Description
    Row 4+: Actual data
    
    Parameters:
    -----------
    filepath : str
        Path to Excel file
    data_start_row : int, default 5
        Row where actual data begins (0-based)
    save_as : str, optional
        Custom name for saved files (without extension)
    data_type_conversion : bool, default True
        Apply automatic data type conversion based on Excel Data_Type row
    sheet_name : str or int, optional
        Name or index of Excel sheet to read
    force_overwrite : bool, default False
        If True, merges new data into existing files. If False, returns existing data if files exist.
    exclude_columns : list of str, optional
        List of column names (abbreviations) to exclude from import. These columns will be skipped
        and not added to the dataset. Example: ['force path fracture', 'force path preload']
        
    Returns:
    --------
    tuple
        (df_raw, df_raw_info) - Data and metadata DataFrames
    """
    
    try:
        print(f"Importing: {filepath}")
        if sheet_name is not None:
            print(f"Reading sheet: {sheet_name}")
        else:
            print("Reading first sheet (default)")
        
        # Get the directory where the Excel file is located
        metadata_dir = os.path.dirname(filepath)
        
        # Read the header rows (first 5 rows)
        header_df = pd.read_excel(filepath, nrows=5, header=None, sheet_name=sheet_name)
        
        # Read the actual data starting from row 6 (index 5)
        df_raw = pd.read_excel(filepath, skiprows=4, sheet_name=sheet_name)
        
        # Get abbreviations from first row and rename columns
        abbreviations = header_df.iloc[0].tolist()
        if len(abbreviations) >= len(df_raw.columns):
            # Strip whitespace from column names to avoid issues with trailing spaces
            clean_abbreviations = [str(abbr).strip() if pd.notna(abbr) else f"Column_{i}" for i, abbr in enumerate(abbreviations[:len(df_raw.columns)])]
            df_raw.columns = clean_abbreviations
        
        # Exclude specified columns if provided
        excluded_indices = []
        if exclude_columns:
            # Normalize excluded column names (strip whitespace, case-insensitive)
            exclude_set = {col.strip().lower() for col in exclude_columns}
            columns_to_drop = [col for col in df_raw.columns if col.strip().lower() in exclude_set]
            
            if columns_to_drop:
                print(f"Excluding columns: {columns_to_drop}")
                # Track which indices to exclude from metadata
                excluded_indices = [i for i, col in enumerate(df_raw.columns) if col in columns_to_drop]
                # Drop the columns from data
                df_raw = df_raw.drop(columns=columns_to_drop)
        
        # Filter abbreviations and other metadata to match remaining columns
        # Keep only indices that were not excluded
        all_indices = list(range(len(abbreviations)))
        kept_indices = [i for i in all_indices if i not in excluded_indices][:len(df_raw.columns)]
        
        abbreviations = [abbreviations[i] if i < len(abbreviations) else None for i in kept_indices]
        
        # Get data types from third row (index 2) for remaining columns
        data_types_full = header_df.iloc[2].tolist()
        data_types = [data_types_full[i] if i < len(data_types_full) else None for i in kept_indices[:len(df_raw.columns)]]
        
        # Apply automatic data type conversion if requested
        if data_type_conversion:
            print("Applying data type conversion...")
            converted_columns = []
            
            for col, data_type in zip(df_raw.columns, data_types):
                if pd.isna(data_type):
                    continue
                    
                data_type = str(data_type).lower().strip()
                
                try:
                    if 'datetime' in data_type:
                        df_raw[col] = pd.to_datetime(df_raw[col], errors='coerce')
                        converted_columns.append(f"{col} (datetime)")
                    elif 'time' in data_type:
                        # Convert Excel time to HH:MM:SS string
                        
                        def convert_time(time_val):
                            if pd.isna(time_val):
                                return None
                            # Handle datetime.time objects (pandas reads Excel time format as this)
                            if isinstance(time_val, datetime.time):
                                return time_val.strftime("%H:%M:%S")
                            # Handle string format (already converted)
                            if isinstance(time_val, str) and ':' in time_val:
                                return time_val.strip()
                            # Handle Excel serial time format (fraction of a day)
                            if isinstance(time_val, (int, float)):
                                total_seconds = float(time_val) * 24 * 3600
                                hours = int(total_seconds // 3600)
                                minutes = int((total_seconds % 3600) // 60)
                                seconds = int(total_seconds % 60)
                                return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                            return None
                        
                        df_raw[col] = df_raw[col].apply(convert_time)
                        converted_columns.append(f"{col} (time)")
                    elif 'integer' in data_type or 'int' in data_type:
                        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').astype('Int64')
                        converted_columns.append(f"{col} (integer)")
                    elif 'float' in data_type:
                        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
                        converted_columns.append(f"{col} (float)")
                    elif 'boolean' in data_type or 'bool' in data_type:
                        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').astype('boolean')
                        converted_columns.append(f"{col} (boolean)")
                    elif 'string' in data_type or 'str' in data_type:
                        df_raw[col] = df_raw[col].astype(str)
                        df_raw[col] = df_raw[col].replace('nan', pd.NA)
                        converted_columns.append(f"{col} (string)")
                        
                except Exception as e:
                    print(f"Warning: Could not convert column '{col}' to '{data_type}': {e}")
                    continue
            
            if converted_columns:
                print(f"Converted {len(converted_columns)} columns: {', '.join(converted_columns)}")
        else:
            print("Using pandas default data type inference")
        
        # Create metadata DataFrame using filtered metadata
        units_full = header_df.iloc[1].tolist()
        long_names_full = header_df.iloc[3].tolist()
        descriptions_full = header_df.iloc[4].tolist()
        
        df_raw_info = pd.DataFrame({
            'Abreviation': abbreviations[:len(df_raw.columns)],
            'Units': [units_full[i] for i in kept_indices if i < len(units_full)][:len(df_raw.columns)],
            'Data_Type': data_types,
            'Long Name': [long_names_full[i] for i in kept_indices if i < len(long_names_full)][:len(df_raw.columns)],
            'Description': [descriptions_full[i] for i in kept_indices if i < len(descriptions_full)][:len(df_raw.columns)]
        })
        
        # Check for and remove duplicate columns in new data BEFORE merging
        duplicate_data_columns = df_raw.columns[df_raw.columns.duplicated()].tolist()
        if duplicate_data_columns:
            print(f"WARNING: Found duplicate columns in Excel file: {duplicate_data_columns}")
            print(f"Removing duplicate columns, keeping first occurrence")
            # Keep only first occurrence of each column
            df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]
            
            # Also remove duplicate metadata entries to keep them in sync
            df_raw_info = df_raw_info[~df_raw_info['Abreviation'].duplicated()].reset_index(drop=True)
            # Ensure metadata matches the remaining columns
            df_raw_info = df_raw_info[df_raw_info['Abreviation'].isin(df_raw.columns)].reset_index(drop=True)
        
        # Check for duplicate metadata entries (in case abbreviations match but columns don't)
        duplicate_metadata = df_raw_info[df_raw_info['Abreviation'].duplicated(keep=False)]
        if len(duplicate_metadata) > 0:
            print(f"WARNING: Found duplicate metadata entries: {duplicate_metadata['Abreviation'].unique().tolist()}")
            print(f"Removing duplicate metadata entries, keeping first occurrence")
            df_raw_info = df_raw_info.drop_duplicates(subset=['Abreviation'], keep='first').reset_index(drop=True)
            # Ensure metadata matches the remaining columns
            df_raw_info = df_raw_info[df_raw_info['Abreviation'].isin(df_raw.columns)].reset_index(drop=True)
        
        print(f"Successfully imported:")
        print(f"  df_raw shape: {df_raw.shape}")
        print(f"  df_raw_info shape: {df_raw_info.shape}")
        if len(df_raw_info) != len(df_raw.columns):
            print(f"  WARNING: Metadata rows ({len(df_raw_info)}) != Data columns ({len(df_raw.columns)})")
        else:
            print(f"  Last column in data: {df_raw.columns[-1]}")
            print(f"  Last abbreviation in metadata: {df_raw_info['Abreviation'].iloc[-1]}")
        
        # Handle file operations with force_overwrite logic
        if save_as:
            data_filename = os.path.join(metadata_dir, f"{save_as}.parquet")
            metadata_filename = os.path.join(metadata_dir, f"{save_as}_info.parquet")
        else:
            data_filename = os.path.join(metadata_dir, "df_raw.parquet")
            metadata_filename = os.path.join(metadata_dir, "df_raw_info.parquet")
        
        # Check if files already exist
        data_exists = os.path.exists(data_filename)
        metadata_exists = os.path.exists(metadata_filename)
        
        if data_exists and metadata_exists:
            if not force_overwrite:
                print(f"Master dataset files already exist:")
                print(f"  - Data: {data_filename}")
                print(f"  - Metadata: {metadata_filename}")
                print(f"Use force_overwrite=True to merge new data into existing files")
                
                # Return existing data
                existing_data = pd.read_parquet(data_filename, engine='fastparquet')
                existing_info = pd.read_parquet(metadata_filename, engine='fastparquet')
                print(f"Returning existing data: {existing_data.shape[0]} rows, {existing_data.shape[1]} columns")
                return existing_data, existing_info
            else:
                # Force overwrite: merge new data into existing files
                print(f"Force overwrite enabled - merging into existing files")
                
                if data_exists and metadata_exists:
                    # Load existing data
                    existing_data = pd.read_parquet(data_filename, engine='fastparquet')
                    existing_info = pd.read_parquet(metadata_filename, engine='fastparquet')
                    
                    # Strip whitespace from abbreviations in existing_info to ensure consistent comparisons
                    existing_info['Abreviation'] = existing_info['Abreviation'].str.strip()
                    
                    print(f"Existing data: {existing_data.shape[0]} rows, {existing_data.shape[1]} columns")
                    print(f"New data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
                    
                    # Remove excluded columns from existing data if specified
                    if exclude_columns:
                        exclude_set = {col.strip().lower() for col in exclude_columns}
                        existing_cols_to_drop = [col for col in existing_data.columns if col.strip().lower() in exclude_set]
                        
                        if existing_cols_to_drop:
                            print(f"Removing excluded columns from existing data: {existing_cols_to_drop}")
                            existing_data = existing_data.drop(columns=existing_cols_to_drop)
                            # Also remove from metadata
                            existing_info = existing_info[~existing_info['Abreviation'].str.strip().str.lower().isin(exclude_set)]
                    
                    # Check for and remove duplicate data columns
                    duplicate_data_columns = existing_data.columns[existing_data.columns.duplicated()].tolist()
                    if duplicate_data_columns:
                        print(f"Found duplicate data columns: {duplicate_data_columns}")
                        existing_data = existing_data.loc[:, ~existing_data.columns.duplicated()]
                        print(f"Removed duplicate data columns, keeping first occurrence")
                    
                    # Check for and remove duplicate metadata columns
                    duplicate_metadata_columns = existing_info[existing_info.duplicated(subset=['Abreviation'], keep=False)]['Abreviation'].unique()
                    
                    if len(duplicate_metadata_columns) > 0:
                        print(f"Found duplicate metadata columns: {list(duplicate_metadata_columns)}")
                        existing_info = existing_info.drop_duplicates(subset=['Abreviation'], keep='first')
                        print(f"Removed duplicate metadata entries, keeping first occurrence")
                    
                    # Merge new data (rows + columns) into existing data
                    merged_info = existing_info.copy()

                    new_columns = [col for col in df_raw.columns if col not in existing_data.columns]

                    if 'AFN' in existing_data.columns and 'AFN' in df_raw.columns:
                        existing_for_merge = existing_data.copy()
                        new_for_merge = df_raw.copy()

                        existing_for_merge['AFN'] = pd.to_numeric(existing_for_merge['AFN'], errors='coerce')
                        new_for_merge['AFN'] = pd.to_numeric(new_for_merge['AFN'], errors='coerce')

                        existing_afn_set = set(existing_for_merge['AFN'].dropna().astype('Int64').tolist())
                        new_afn_series = pd.to_numeric(new_for_merge['AFN'].dropna(), errors='coerce').dropna()
                        new_afn_set = set(new_afn_series.astype('Int64').tolist())
                        added_afns = sorted(list(new_afn_set - existing_afn_set))
                        if added_afns:
                            preview = ", ".join(str(afn) for afn in added_afns[:10])
                            suffix = "..." if len(added_afns) > 10 else ""
                            print(f"Adding {len(added_afns)} new AFNs: {preview}{suffix}")

                        existing_non_na = existing_for_merge[existing_for_merge['AFN'].notna()].copy()
                        existing_na = existing_for_merge[existing_for_merge['AFN'].isna()].copy()
                        new_non_na = new_for_merge[new_for_merge['AFN'].notna()].copy()
                        new_na = new_for_merge[new_for_merge['AFN'].isna()].copy()

                        if not existing_non_na.empty:
                            existing_non_na['AFN'] = existing_non_na['AFN'].astype('Int64')
                        if not new_non_na.empty:
                            new_non_na['AFN'] = new_non_na['AFN'].astype('Int64')

                        existing_non_na = existing_non_na.drop_duplicates(subset=['AFN'], keep='first')
                        new_non_na = new_non_na.drop_duplicates(subset=['AFN'], keep='last')

                        existing_indexed = existing_non_na.set_index('AFN')
                        new_indexed = new_non_na.set_index('AFN')

                        merged_non_na = new_indexed.combine_first(existing_indexed)
                        merged_non_na = merged_non_na.sort_index().reset_index()

                        column_order = existing_data.columns.tolist()
                        for col in df_raw.columns:
                            if col not in column_order:
                                column_order.append(col)
                        if 'AFN' in column_order:
                            column_order.insert(0, column_order.pop(column_order.index('AFN')))

                        merged_non_na = merged_non_na.reindex(columns=column_order)

                        if not new_na.empty:
                            na_rows = new_na
                        elif not existing_na.empty:
                            na_rows = existing_na
                        else:
                            na_rows = pd.DataFrame(columns=column_order)
                        na_rows = na_rows.reindex(columns=column_order)

                        merged_data = pd.concat([merged_non_na, na_rows], ignore_index=True, sort=False)
                        merged_data = merged_data.reindex(columns=column_order)

                        if 'AFN' in merged_data.columns:
                            merged_data['AFN'] = pd.to_numeric(merged_data['AFN'], errors='coerce')
                            merged_data['AFN'] = merged_data['AFN'].astype('Int64')
                    else:
                        print("Warning: 'AFN' column missing - falling back to column-wise merge without row expansion")
                        merged_data = existing_data.copy()
                        for col in df_raw.columns:
                            merged_data[col] = df_raw[col]
                        column_order = merged_data.columns.tolist()
                        if 'AFN' in column_order:
                            column_order.insert(0, column_order.pop(column_order.index('AFN')))
                        merged_data = merged_data.reindex(columns=column_order)

                    merged_info['Abreviation'] = merged_info['Abreviation'].astype(str).str.strip()

                    for _, meta_row in df_raw_info.iterrows():
                        abbr = meta_row['Abreviation']
                        if pd.isna(abbr):
                            continue
                        abbr_str = str(abbr).strip()
                        existing_metadata = merged_info[merged_info['Abreviation'] == abbr_str]
                        if existing_metadata.empty:
                            new_meta = pd.DataFrame({
                                'Abreviation': [abbr_str],
                                'Units': [meta_row['Units']],
                                'Data_Type': [meta_row['Data_Type']],
                                'Long Name': [meta_row['Long Name']],
                                'Description': [meta_row['Description']]
                            })
                            merged_info = pd.concat([merged_info, new_meta], ignore_index=True)
                        else:
                            idx = existing_metadata.index[0]
                            merged_info.at[idx, 'Units'] = meta_row['Units']
                            merged_info.at[idx, 'Data_Type'] = meta_row['Data_Type']
                            merged_info.at[idx, 'Long Name'] = meta_row['Long Name']
                            merged_info.at[idx, 'Description'] = meta_row['Description']

                    column_names_str = [str(col).strip() for col in merged_data.columns]
                    merged_info = merged_info[merged_info['Abreviation'].isin(column_names_str)].reset_index(drop=True)

                    if new_columns:
                        print(f"Added new columns: {new_columns}")

                    df_raw = merged_data
                    df_raw_info = merged_info
                else:
                    print("Creating new master dataset files")
        
        # Sort info parquet by alphabetical order of abbreviations before saving
        df_raw_info = df_raw_info.sort_values('Abreviation', ascending=True).reset_index(drop=True)
        
        # Final safety check: ensure no duplicate columns before saving (parquet doesn't allow duplicates)
        if df_raw.columns.duplicated().any():
            print(f"ERROR: Duplicate columns detected before saving: {df_raw.columns[df_raw.columns.duplicated()].tolist()}")
            print(f"Removing duplicate columns as final safety measure")
            df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]
            # Sync metadata
            df_raw_info = df_raw_info[df_raw_info['Abreviation'].isin(df_raw.columns)].reset_index(drop=True)
        
        # Final safety check: ensure no duplicate metadata before saving
        if df_raw_info['Abreviation'].duplicated().any():
            print(f"ERROR: Duplicate metadata detected before saving: {df_raw_info[df_raw_info['Abreviation'].duplicated()]['Abreviation'].tolist()}")
            print(f"Removing duplicate metadata entries as final safety measure")
            df_raw_info = df_raw_info.drop_duplicates(subset=['Abreviation'], keep='first').reset_index(drop=True)
        
        # Save files
        df_raw.to_parquet(data_filename, index=False, engine='fastparquet')
        df_raw_info.to_parquet(metadata_filename, index=False, engine='fastparquet')
        
        # Verify files were created
        if os.path.exists(data_filename):
            file_size = os.path.getsize(data_filename)
            print(f"Saved successfully:")
            print(f"  - Data: {data_filename} ({file_size} bytes)")
        else:
            print(f"ERROR: Failed to create {data_filename}")
            
        if os.path.exists(metadata_filename):
            file_size = os.path.getsize(metadata_filename)
            print(f"  - Metadata: {metadata_filename} ({file_size} bytes)")
        else:
            print(f"ERROR: Failed to create {metadata_filename}")
        
        return df_raw, df_raw_info
        
    except Exception as e:
        print(f"Error importing: {e}")
        return None, None


def import_parquet(file_path: str) -> pd.DataFrame:
    """
    Import single Parquet file.
    
    Parameters:
    -----------
    file_path : str
        Path to the Parquet file
        
    Returns:
    --------
    pd.DataFrame
        Imported DataFrame
    """
    try:
        df = pd.read_parquet(file_path, engine='fastparquet')
        print(f"Successfully imported Parquet: {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error importing Parquet {file_path}: {e}")
        return pd.DataFrame()


def import_csv(file_path: str) -> pd.DataFrame:
    """
    Simple function to import a CSV file with a single header row as a DataFrame.
    Automatically converts time columns to float (seconds).
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file to import
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the CSV data with time columns as float
        
    Example:
    --------
    df_crack_strikes = helper.import_csv(dc.get_data_path(ON4PB_path, 'crack_paths') + '/crack_strikes.csv')
    """
    try:
        df = pd.read_csv(file_path)
        
        # Time columns are already in HH:MM:SS string format - no conversion needed
        time_columns = ['exp_start', 'exp_end', 'start_preload', 'end_preload']
        found_time_cols = [col for col in time_columns if col in df.columns]
        if found_time_cols:
            print(f"Time columns found (HH:MM:SS format): {found_time_cols}")
        
        # Convert string representations of lists back to actual lists
        for col in df.columns:
            if df[col].dtype == 'object':  # String columns
                # Check if this column contains list-like data (starts with [ and ends with ])
                sample_values = df[col].dropna().head(5)
                if len(sample_values) > 0:
                    # Check if any value looks like a list
                    if any(str(val).startswith('[') and str(val).endswith(']') for val in sample_values):
                        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else x)
        
        print(f"Successfully imported CSV: {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error importing CSV {file_path}: {e}")
        return pd.DataFrame()

def fix_vs3pb_distance_scaling(raw_force_data_path, afn_range=(275, 352), dry_run=True):
    """
    Permanently fix distance scaling issue in VS3PB CSV files.
    
    For AFNs 275-352, the distance values need to be multiplied by 10 due to
    encoder impulse configuration issue (33270 impulses/mm instead of 1210).
    
    Background:
    -----------
    The old encoder had 33270 impulses per mm at 3327 samples per mm (200Hz).
    The true values should be 1210 encoder impulses per mm with 1210 samples per mm (2kHz).
    Therefore distance values for these experiments need to be multiplied by 10.
    
    Parameters:
    -----------
    raw_force_data_path : str
        Directory containing MSCU{AFN:04d}_samples.csv files
    afn_range : tuple, default (275, 352)
        Range of AFNs to fix (inclusive: both start and end are included)
    dry_run : bool, default True
        If True, only shows what would be changed without modifying files
        
    Returns:
    --------
    dict
        Summary of processed files with keys: 'processed', 'not_found', 'errors'
        
    Example:
    --------
    # Check what will be changed
    results = fix_vs3pb_distance_scaling(raw_force_data_path_VS3PB, dry_run=True)
    
    # Apply the changes
    results = fix_vs3pb_distance_scaling(raw_force_data_path_VS3PB, dry_run=False)
    """
    
    afn_start, afn_end = afn_range
    results = {
        'processed': [],
        'not_found': [],
        'errors': []
    }
    
    print(f"{'DRY RUN: ' if dry_run else ''}Processing AFNs {afn_start} to {afn_end} (inclusive)")
    print(f"Directory: {raw_force_data_path}\n")
    
    for afn in range(afn_start, afn_end + 1):
        filename = f"MSCU{afn:04d}_samples.csv"
        filepath = os.path.join(raw_force_data_path, filename)
        
        if not os.path.exists(filepath):
            results['not_found'].append(afn)
            continue
            
        try:
            # Read CSV
            df = pd.read_csv(filepath, comment='#')
            
            # Check if distance column exists
            if 'distance [mm]' not in df.columns:
                print(f"WARNING: AFN {afn} - 'distance [mm]' column not found")
                results['errors'].append(afn)
                continue
            
            # Check current distance range
            old_min = df['distance [mm]'].min()
            old_max = df['distance [mm]'].max()
            
            # Apply correction
            df['distance [mm]'] *= 10
            
            new_min = df['distance [mm]'].min()
            new_max = df['distance [mm]'].max()
            
            if dry_run:
                print(f"AFN {afn}: Would scale distance from [{old_min:.3f}, {old_max:.3f}] to [{new_min:.3f}, {new_max:.3f}]")
            else:
                # Save back to CSV
                df.to_csv(filepath, index=False)
                print(f"AFN {afn}: Scaled distance from [{old_min:.3f}, {old_max:.3f}] to [{new_min:.3f}, {new_max:.3f}] ✓")
                
            results['processed'].append(afn)
            
        except Exception as e:
            print(f"ERROR: AFN {afn} - {e}")
            results['errors'].append(afn)
    
    # Print summary
    print(f"\n{'DRY RUN ' if dry_run else ''}SUMMARY:")
    print(f"  Processed: {len(results['processed'])} files")
    print(f"  Not found: {len(results['not_found'])} files")
    print(f"  Errors: {len(results['errors'])} files")
    
    if results['not_found']:
        print(f"\n  Missing AFNs: {results['not_found'][:10]}{'...' if len(results['not_found']) > 10 else ''}")
    
    if dry_run:
        print(f"\n  To apply changes, run with dry_run=False")
    else:
        print(f"\n  ✓ Changes saved to CSV files")
    
    return results

def merge_load_data(raw_data_path, raw_info_data_path, load_data_dir, force_overwrite=False):
    """
    Merge loading cell data into master dataset files.
    
    This function adds load data columns (w_data, P_data) to the master dataset files:
    - {base_name}.parquet (data)
    - {base_name}_info.parquet (metadata)
    
    If force_overwrite=False and columns already exist, provides feedback and returns existing data.
    If force_overwrite=True, updates existing load data columns.
    
    Parameters:
    -----------
    raw_data_path : str
        Filepath to master Parquet data file (must contain AFN column)
    raw_info_data_path : str
        Filepath to master Parquet metadata file
    load_data_dir : str
        Directory path containing loading data CSV files (MSCU{AFN}_samples.csv)
    force_overwrite : bool, default False
        If True, updates existing load data columns. If False, returns existing data if columns exist.
        
    Returns:
    --------
    tuple
        (df_merged, df_info_merged) - Merged data and metadata DataFrames
    """
    
    # Load master dataset files
    try:
        df_raw = pd.read_parquet(raw_data_path, engine='fastparquet')
        print(f"Loaded master data from: {raw_data_path}")
    except Exception as e:
        print(f"Error loading master data {raw_data_path}: {e}")
        return None, None
    
    try:
        df_raw_info = pd.read_parquet(raw_info_data_path, engine='fastparquet')
        print(f"Loaded master metadata from: {raw_info_data_path}")
    except Exception as e:
        print(f"Error loading master metadata {raw_info_data_path}: {e}")
        return None, None
    
    # Check if AFN column exists
    if 'AFN' not in df_raw.columns:
        print("Error: AFN column not found in master data")
        return None, None
    
    # Check if load data columns already exist
    load_columns = ['w_data', 'P_data']
    existing_columns = [col for col in load_columns if col in df_raw.columns]
    
    if existing_columns and not force_overwrite:
        print(f"Load data columns already exist: {existing_columns}")
        print("Use force_overwrite=True to update existing load data")
        return df_raw, df_raw_info
    
    if existing_columns and force_overwrite:
        print(f"Force overwrite enabled - updating existing load data columns: {existing_columns}")
    elif not existing_columns:
        print("Adding new load data columns: w_data, P_data")
    
    # Create copies to avoid modifying originals
    df_merged = df_raw.copy()
    
    # Define new columns info for loading data
    new_columns_info = {
        'w_data': {
            'units': 'mm',
            'data_type': 'List',
            'long_name': 'distance',
            'description': 'distance of the linear actuator'
        },
        'P_data': {
            'units': 'N', 
            'data_type': 'List',
            'long_name': 'force',
            'description': 'force signal of the load cell'
        }
    }
    
    # Initialize new columns as lists
    df_merged['w_data'] = None  # distance [mm] as list
    df_merged['P_data'] = None  # force [N] as list
    
    # Track statistics
    found_count = 0
    not_found_count = 0
    
    # Loop through each AFN and look for corresponding loading data file
    for idx, row in df_merged.iterrows():
        afn = row['AFN']
        
        # Skip rows with NaN AFN values
        if pd.isna(afn):
            print(f"Skipping row {idx}: AFN is NaN")
            not_found_count += 1
            continue
            
        # Convert AFN to integer and format with zero-padding to match filename format
        try:
            afn_int = int(float(afn))
        except (ValueError, TypeError) as e:
            print(f"Skipping row {idx}: Cannot convert AFN '{afn}' to integer: {e}")
            not_found_count += 1
            continue
            
        # Format AFN with zero-padding (e.g., 153 -> 0153)
        load_filename = f"MSCU{afn_int:04d}_samples.csv"
        load_filepath = os.path.join(load_data_dir, load_filename)
        
        try:
            if os.path.exists(load_filepath):
                # Load loading data (skip comment lines starting with #)
                load_df = pd.read_csv(load_filepath, comment='#')
                
                # Check if required columns exist
                if 'distance [mm]' in load_df.columns and 'force [N]' in load_df.columns:
                    # Convert to lists (all entries for this AFN)
                    distance_list = load_df['distance [mm]'].tolist()
                    force_list = load_df['force [N]'].tolist()
                    
                    # Add to merged DataFrame
                    df_merged.at[idx, 'w_data'] = distance_list
                    df_merged.at[idx, 'P_data'] = force_list
                    
                    found_count += 1
                    print(f"Found loading data for AFN {afn_int}: {len(distance_list)} measurements")
                else:
                    print(f"Warning: Required columns not found in {load_filename}")
                    not_found_count += 1
            else:
                print(f"Loading data file not found: {load_filename}")
                not_found_count += 1
                
        except Exception as e:
            print(f"Error processing loading data file {load_filename}: {e}")
            not_found_count += 1
    
    # Print summary
    print(f"\nLoading data merge summary:")
    print(f"  - Total AFNs processed: {len(df_merged)}")
    print(f"  - Loading data found: {found_count}")
    print(f"  - Loading data not found: {not_found_count}")
    
    # Extend metadata with new columns (check for existing entries)
    df_info_merged = df_raw_info.copy()
    
    # First, check for and remove duplicate columns if force_overwrite is True
    if force_overwrite:
        # Check for duplicate data columns
        duplicate_data_columns = df_merged.columns[df_merged.columns.duplicated()].tolist()
        if duplicate_data_columns:
            print(f"Found duplicate data columns: {duplicate_data_columns}")
            df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]
            print(f"Removed duplicate data columns, keeping first occurrence")
        
        # Check for duplicate metadata columns
        duplicate_metadata_columns = df_info_merged[df_info_merged.duplicated(subset=['Abreviation'], keep=False)]['Abreviation'].unique()
        
        if len(duplicate_metadata_columns) > 0:
            print(f"Found duplicate metadata columns: {list(duplicate_metadata_columns)}")
            
            # Remove duplicates, keeping only the first occurrence
            df_info_merged = df_info_merged.drop_duplicates(subset=['Abreviation'], keep='first')
            print(f"Removed duplicate metadata entries, keeping first occurrence")
    
    for col_name, col_info in new_columns_info.items():
        # Check if metadata row already exists
        existing_row = df_info_merged[df_info_merged['Abreviation'] == col_name]
        
        if existing_row.empty:
            # Add new metadata row
            new_row = pd.DataFrame({
                'Abreviation': [col_name],
                'Units': [col_info['units']],
                'Data_Type': [col_info['data_type']],
                'Long Name': [col_info['long_name']],
                'Description': [col_info['description']]
            })
            df_info_merged = pd.concat([df_info_merged, new_row], ignore_index=True)
            print(f"Added metadata for new column: {col_name}")
        else:
            # Update existing metadata row
            idx = existing_row.index[0]
            df_info_merged.at[idx, 'Units'] = col_info['units']
            df_info_merged.at[idx, 'Data_Type'] = col_info['data_type']
            df_info_merged.at[idx, 'Long Name'] = col_info['long_name']
            df_info_merged.at[idx, 'Description'] = col_info['description']
            print(f"Updated existing metadata for column: {col_name}")
    
    # Save to master dataset files
    data_filename = raw_data_path
    info_filename = raw_info_data_path
    
    # Convert w_data and P_data columns to object dtype to properly store lists
    df_merged['w_data'] = df_merged['w_data'].astype(object)
    df_merged['P_data'] = df_merged['P_data'].astype(object)
    
    # Sort info parquet by alphabetical order of abbreviations before saving
    df_info_merged = df_info_merged.sort_values('Abreviation', ascending=True).reset_index(drop=True)
    
    # Save as Parquet files
    df_merged.to_parquet(data_filename, index=False, engine='fastparquet')
    df_info_merged.to_parquet(info_filename, index=False, engine='fastparquet')
    
    # Verify files were created
    if os.path.exists(data_filename):
        file_size = os.path.getsize(data_filename)
        print(f"Updated master dataset:")
        print(f"  - Data: {data_filename} ({file_size} bytes)")
    else:
        print(f"ERROR: Failed to update {data_filename}")
        
    if os.path.exists(info_filename):
        file_size = os.path.getsize(info_filename)
        print(f"  - Metadata: {info_filename} ({file_size} bytes)")
    else:
        print(f"ERROR: Failed to update {info_filename}")
    
    return df_merged, df_info_merged

def merge_load_analyser_data(raw_data_path, raw_info_data_path, force_overwrite=False):
    """
    Merge load signal analyser results into master dataset files.
    
    This function adds analysis columns to the master dataset files:
    - {base_name}.parquet (data)
    - {base_name}_info.parquet (metadata)
    
    The analysis file is automatically detected as {prefix}_load_signal_analysis.parquet
    where prefix is extracted from the raw_data_path.
    
    If force_overwrite=False and columns already exist, provides feedback and returns existing data.
    If force_overwrite=True, updates existing analysis columns.
    
    Parameters:
    -----------
    raw_data_path : str
        Filepath to master Parquet data file (e.g., ON4PB_raw.parquet)
    raw_info_data_path : str
        Filepath to master Parquet metadata file (e.g., ON4PB_raw_info.parquet)
    force_overwrite : bool, default False
        If True, updates existing analysis columns. If False, returns existing data if columns exist.
        
    Returns:
    --------
    tuple
        (df_merged, df_info_merged) - Merged data and metadata DataFrames
    """
    
    # Load master dataset files
    try:
        df_raw = pd.read_parquet(raw_data_path, engine='fastparquet')
        print(f"Loaded master data from: {raw_data_path}")
    except Exception as e:
        print(f"Error loading master data {raw_data_path}: {e}")
        return None, None
    
    try:
        df_raw_info = pd.read_parquet(raw_info_data_path, engine='fastparquet')
        print(f"Loaded master metadata from: {raw_info_data_path}")
    except Exception as e:
        print(f"Error loading master metadata {raw_info_data_path}: {e}")
        return None, None
    
    # Auto-detect analysis file path
    base_name = os.path.basename(raw_data_path)
    prefix = base_name.split('_')[0]  # Get part before first underscore
    analysis_data_path = os.path.join(os.path.dirname(raw_data_path), f"{prefix}_load_signal_analysis.parquet")
    
    # Check if analysis file exists
    if not os.path.exists(analysis_data_path):
        print(f"Analysis file not found: {analysis_data_path}")
        print("Run load_signal_analyser.py first to create analysis data")
        return df_raw, df_raw_info
    
    # Load analysis data
    try:
        df_analysis = pd.read_parquet(analysis_data_path, engine='fastparquet')
        print(f"Loaded analysis data from: {analysis_data_path}")
    except Exception as e:
        print(f"Error loading analysis data {analysis_data_path}: {e}")
        return df_raw, df_raw_info
    
    # Check if AFN column exists in both datasets
    if 'AFN' not in df_raw.columns or 'AFN' not in df_analysis.columns:
        print("Error: AFN column not found in master data or analysis data")
        return df_raw, df_raw_info
    
    # Define analysis columns info
    analysis_columns_info = {
        'offset_region': {
            'units': 'mm',
            'data_type': 'List',
            'long_name': 'x region of offset calculation',
            'description': 'the x region where the offset is calculated as list (x1,x2) before the load sensor hits the sample. X2 also describes the time just before the load cell hits the sample.'
        },
        'P_off': {
            'units': 'N',
            'data_type': 'Float',
            'long_name': 'offset of the load cell',
            'description': 'offset which was calculated on the signal during a free run before the load sensor hits the sample'
        },
        'start_lin': {
            'units': 'None',
            'data_type': 'Coordinates as Tuple',
            'long_name': 'start of linear signal before failure',
            'description': 'starting point where the global stiffness is calculated after valley but behind noise'
        },
        'end_lin': {
            'units': 'None',
            'data_type': 'Coordinates as Tuple',
            'long_name': 'end of the linear rise before failure',
            'description': 'end of the linear rise before failure or plastical cap or stagnating behaviour to calculate global stiffness'
        },
        'k': {
            'units': 'N/mm',
            'data_type': 'Float',
            'long_name': 'global stiffness',
            'description': 'global stiffness of the bending sample calculated in the linear rise of load signal before failure'
        },
        'P_c': {
            'units': 'N',
            'data_type': 'Coordinates as Tuple',
            'long_name': 'critical fracture Force',
            'description': 'critical force at the time of the crack onset'
        },
        'F_dot': {
            'units': 'N/s',
            'data_type': 'Float',
            'long_name': 'global force rate',
            'description': 'The global force rate quantifies how quickly the applied force changes over time. It is obtained as the product of the global stiffness k and the shear speed (F_dot = 1e-3 * k * shear_speed, converting µm/s to mm/s)'
        }
    }
    
    # Check if analysis columns already exist
    analysis_columns = list(analysis_columns_info.keys())
    existing_columns = [col for col in analysis_columns if col in df_raw.columns]
    
    if existing_columns and not force_overwrite:
        print(f"Analysis columns already exist: {existing_columns}")
        print("Use force_overwrite=True to update existing analysis data")
        return df_raw, df_raw_info
    
    if existing_columns and force_overwrite:
        print(f"Force overwrite enabled - updating existing analysis columns: {existing_columns}")
    elif not existing_columns:
        print("Adding new analysis columns: " + ", ".join(analysis_columns))
    
    # Create copies to avoid modifying originals
    df_merged = df_raw.copy()
    
    # Check for and remove duplicate columns if force_overwrite is True
    if force_overwrite:
        # Check for duplicate data columns
        duplicate_data_columns = df_merged.columns[df_merged.columns.duplicated()].tolist()
        if duplicate_data_columns:
            print(f"Found duplicate data columns: {duplicate_data_columns}")
            df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]
            print(f"Removed duplicate data columns, keeping first occurrence")
    
    # Initialize analysis columns if they don't exist
    # Set dtype to 'object' to allow storing lists, tuples, and other Python objects
    for col_name in analysis_columns:
        if col_name not in df_merged.columns:
            df_merged[col_name] = None
            # Convert to object dtype to allow lists/tuples
            df_merged[col_name] = df_merged[col_name].astype(object)
            print(f"Initialized analysis column: {col_name}")
        else:
            # Ensure existing columns are also object dtype to accept lists/tuples
            df_merged[col_name] = df_merged[col_name].astype(object)
    
    # Merge analysis data by AFN
    merged_count = 0
    for _, analysis_row in df_analysis.iterrows():
        afn = analysis_row['AFN']
        
        # Find matching row in master data
        master_mask = df_merged['AFN'] == afn
        if master_mask.any():
            master_idx = df_merged[master_mask].index[0]
            
            # Update analysis columns
            for col_name in analysis_columns:
                if col_name in analysis_row:
                    value = analysis_row[col_name]
                    # Handle different data types for null checking
                    if isinstance(value, (list, tuple)):
                        # For list/tuple values, check if not empty
                        if value is not None and len(value) > 0:
                            df_merged.at[master_idx, col_name] = value
                    elif pd.notna(value):
                        # For scalar values, use standard null check
                        df_merged.at[master_idx, col_name] = value
            
            merged_count += 1
            print(f"Merged analysis data for AFN {afn}")
        else:
            print(f"Warning: AFN {afn} not found in master data")
    
    # Calculate F_dot (global force rate) for all merged rows
    # F_dot [N/s] = k [N/mm] * shear_speed [µm/s] * 1e-3 [mm/µm]
    print(f"\nCalculating F_dot (global force rate)...")
    f_dot_calculated = 0
    f_dot_missing_data = 0
    
    if 'k' in df_merged.columns and 'shear speed' in df_merged.columns:
        for idx, row in df_merged.iterrows():
            k = row['k'] if 'k' in row.index else None
            shear_speed = row['shear speed'] if 'shear speed' in row.index else None
            
            # Check if both k and shear_speed are available and valid
            if pd.notna(k) and pd.notna(shear_speed):
                try:
                    k_val = float(k)
                    shear_speed_val = float(shear_speed)
                    F_dot = 1e-3 * k_val * shear_speed_val  # Convert µm/s to mm/s: 1e-3 mm/µm
                    df_merged.at[idx, 'F_dot'] = F_dot
                    f_dot_calculated += 1
                except (ValueError, TypeError) as e:
                    f_dot_missing_data += 1
            else:
                f_dot_missing_data += 1
    else:
        missing_cols = []
        if 'k' not in df_merged.columns:
            missing_cols.append('k')
        if 'shear speed' not in df_merged.columns:
            missing_cols.append('shear speed')
        print(f"Warning: Cannot calculate F_dot - missing columns: {', '.join(missing_cols)}")
    
    if f_dot_calculated > 0:
        print(f"  - Calculated F_dot for {f_dot_calculated} AFNs")
    if f_dot_missing_data > 0:
        print(f"  - Skipped {f_dot_missing_data} AFNs (missing k or shear speed)")
    
    # Ensure F_dot column has consistent data type (float64)
    if 'F_dot' in df_merged.columns:
        df_merged['F_dot'] = pd.to_numeric(df_merged['F_dot'], errors='coerce')
    
    print(f"\nAnalysis data merge summary:")
    print(f"  - Total AFNs in analysis data: {len(df_analysis)}")
    print(f"  - Successfully merged: {merged_count}")
    
    # Extend metadata with analysis columns (check for existing entries)
    df_info_merged = df_raw_info.copy()
    
    # First, check for and remove duplicate columns if force_overwrite is True
    if force_overwrite:
        # Check for duplicate metadata columns
        duplicate_metadata_columns = df_info_merged[df_info_merged.duplicated(subset=['Abreviation'], keep=False)]['Abreviation'].unique()
        
        if len(duplicate_metadata_columns) > 0:
            print(f"Found duplicate metadata columns: {list(duplicate_metadata_columns)}")
            
            # Remove duplicates, keeping only the first occurrence
            df_info_merged = df_info_merged.drop_duplicates(subset=['Abreviation'], keep='first')
            print(f"Removed duplicate metadata entries, keeping first occurrence")
    
    for col_name, col_info in analysis_columns_info.items():
        # Check if metadata row already exists
        existing_row = df_info_merged[df_info_merged['Abreviation'] == col_name]
        
        if existing_row.empty:
            # Add new metadata row
            new_row = pd.DataFrame({
                'Abreviation': [col_name],
                'Units': [col_info['units']],
                'Data_Type': [col_info['data_type']],
                'Long Name': [col_info['long_name']],
                'Description': [col_info['description']]
            })
            df_info_merged = pd.concat([df_info_merged, new_row], ignore_index=True)
            print(f"Added metadata for analysis column: {col_name}")
        else:
            # Always update existing metadata row (even if force_overwrite=False)
            # This ensures descriptions stay synchronized with data_config.py
            idx = existing_row.index[0]
            df_info_merged.at[idx, 'Units'] = col_info['units']
            df_info_merged.at[idx, 'Data_Type'] = col_info['data_type']
            df_info_merged.at[idx, 'Long Name'] = col_info['long_name']
            df_info_merged.at[idx, 'Description'] = col_info['description']
            print(f"Updated metadata for analysis column: {col_name}")
    
    # Save to master dataset files
    data_filename = raw_data_path
    info_filename = raw_info_data_path
    
    # Sort info parquet by alphabetical order of abbreviations before saving
    df_info_merged = df_info_merged.sort_values('Abreviation', ascending=True).reset_index(drop=True)
    
    # Save as Parquet files
    df_merged.to_parquet(data_filename, index=False, engine='fastparquet')
    df_info_merged.to_parquet(info_filename, index=False, engine='fastparquet')
    
    # Verify files were created
    if os.path.exists(data_filename):
        file_size = os.path.getsize(data_filename)
        print(f"Updated master dataset:")
        print(f"  - Data: {data_filename} ({file_size} bytes)")
    else:
        print(f"ERROR: Failed to update {data_filename}")
        
    if os.path.exists(info_filename):
        file_size = os.path.getsize(info_filename)
        print(f"  - Metadata: {info_filename} ({file_size} bytes)")
    else:
        print(f"ERROR: Failed to update {info_filename}")
    
    return df_merged, df_info_merged

def add_columns_to_data(raw_data_path, raw_info_data_path, new_columns_info, appendix="", save=True):
    """
    Add new columns to data and metadata DataFrames and save as Parquet files.
    
    Parameters:
    -----------
    raw_data_path : str
        Filepath to Parquet data file
    raw_info_data_path : str
        Filepath to Parquet metadata file
    new_columns_info : dict
        Dictionary with new column information. Format:
        {
            'column_name': {
                'units': str,
                'data_type': str,
                'long_name': str,
                'description': str
            },
            ...
        }
    appendix : str, optional
        Appendix for saved files (default: "")
    save : bool, default True
        Whether to save extended data as Parquet files
        
    Returns:
    --------
    tuple
        (df_extended, df_info_extended) - Extended data and metadata DataFrames
        
    Example:
    --------
    >>> new_cols = {
    ...     'P_off': {
    ...         'units': 'N',
    ...         'data_type': 'Float',
    ...         'long_name': 'offset of the load cell',
    ...         'description': 'offset calculated from signal before contact'
    ...     }
    ... }
    >>> df, df_info = add_columns_to_dataframe(
    ...     'data.parquet', 'data_info.parquet', new_cols, appendix='_man'
    ... )
    """
    
    # Load data from Parquet
    try:
        df_raw = pd.read_parquet(raw_data_path, engine='fastparquet')
        print(f"Loaded data from: {raw_data_path}")
    except Exception as e:
        print(f"Error loading data {raw_data_path}: {e}")
        return None, None
    
    # Load metadata from Parquet
    try:
        df_raw_info = pd.read_parquet(raw_info_data_path, engine='fastparquet')
        print(f"Loaded metadata from: {raw_info_data_path}")
    except Exception as e:
        print(f"Error loading metadata {raw_info_data_path}: {e}")
        return None, None
    
    # Create copies
    df_extended = df_raw.copy()
    df_info_extended = df_raw_info.copy()
    
    # Check for existing columns and warn about potential data loss
    existing_columns = []
    new_columns = []
    
    for col_name in new_columns_info.keys():
        if col_name not in df_extended.columns:
            df_extended[col_name] = None
            new_columns.append(col_name)
            print(f"Added column: {col_name}")
        else:
            existing_columns.append(col_name)
            print(f"WARNING: Column '{col_name}' already exists - skipping initialization to preserve existing data")
    
    # Enhanced warning for data safety
    if existing_columns:
        print(f"\nDATA SAFETY WARNING:")
        print(f"   The following columns already exist and will NOT be overwritten:")
        for col in existing_columns:
            print(f"   - {col}")
        print(f"   If you need to update these columns, consider using merge_load_and_analysis_data() instead.")
        print(f"   This prevents accidental loss of analysis results from load_signal_analyser.py")
    
    if new_columns:
        print(f"\nSuccessfully added new columns: {new_columns}")
    
    # Add or update metadata rows
    for col_name, col_info in new_columns_info.items():
        # Check if metadata row already exists (handle whitespace in abbreviations)
        existing_row_idx = df_info_extended[df_info_extended['Abreviation'].str.strip() == col_name.strip()].index
        
        if len(existing_row_idx) > 0:
            # Update existing metadata row
            idx = existing_row_idx[0]
            df_info_extended.at[idx, 'Units'] = col_info['units']
            df_info_extended.at[idx, 'Data_Type'] = col_info['data_type']
            df_info_extended.at[idx, 'Long Name'] = col_info['long_name']
            df_info_extended.at[idx, 'Description'] = col_info['description']
            print(f"Updated metadata for: {col_name}")
        else:
            # Add new metadata row
            new_row = pd.DataFrame({
                'Abreviation': [col_name],
                'Units': [col_info['units']],
                'Data_Type': [col_info['data_type']],
                'Long Name': [col_info['long_name']],
                'Description': [col_info['description']]
            })
            df_info_extended = pd.concat([df_info_extended, new_row], ignore_index=True)
            print(f"Added metadata for: {col_name}")
    
    # Save as Parquet files if requested
    if save:
        # Determine save directory and filenames
        save_dir = os.path.dirname(raw_data_path)
        base_name_raw = os.path.splitext(os.path.basename(raw_data_path))[0]
        base_name_info = os.path.splitext(os.path.basename(raw_info_data_path))[0]
        data_filename = os.path.join(save_dir, f"{base_name_raw}{appendix}.parquet")
        info_filename = os.path.join(save_dir, f"{base_name_info}{appendix}.parquet")
        
        # Check if files exist before saving
        if os.path.exists(data_filename):
            print(f"Warning: Overwriting existing file: {data_filename}")
        if os.path.exists(info_filename):
            print(f"Warning: Overwriting existing file: {info_filename}")
        
        # Sort info parquet by alphabetical order of abbreviations before saving
        df_info_extended = df_info_extended.sort_values('Abreviation', ascending=True).reset_index(drop=True)
        
        # Save as Parquet files
        df_extended.to_parquet(data_filename, index=False, engine='fastparquet')
        df_info_extended.to_parquet(info_filename, index=False, engine='fastparquet')
        
        # Verify files were created
        if os.path.exists(data_filename):
            file_size = os.path.getsize(data_filename)
            print(f"Saved successfully:")
            print(f"  - Data: {data_filename} ({file_size} bytes)")
        else:
            print(f"ERROR: Failed to create {data_filename}")
            
        if os.path.exists(info_filename):
            file_size = os.path.getsize(info_filename)
            print(f"  - Metadata: {info_filename} ({file_size} bytes)")
        else:
            print(f"ERROR: Failed to create {info_filename}")
    
    print(f"\nSummary: Added {len(new_columns_info)} new columns")
    
    return df_extended, df_info_extended


def apply_rotation_correction(frames_folder_path: str, master_parquet_path: str, max_workers: int = 4, force_overwrite: bool = False):
    """
    Apply rotation correction to all images in AFN_undistorted_frames folders.
    
    This function searches for all folders matching the pattern "*_undistorted_frames"
    (e.g., "154_undistorted_frames"), extracts the AFN number (e.g., 154), reads the
    rotation correction value from the master parquet file, and uses ffmpeg to rotate
    all images. Rotated images are saved in new folders with "*_undistorted_rotcor_frames"
    suffix (e.g., "154_undistorted_rotcor_frames") with the same filenames (e.g., 001.png).
    
    IMPORTANT: This function must be run AFTER distortion correction, as distortion
    correction parameters are orientation-specific and must be applied to unrotated images.
    
    Parameters:
    -----------
    frames_folder_path : str
        Path to folder containing AFN_undistorted_frames folders
        (e.g., "/path/to/video_fracture_and_preload")
    master_parquet_path : str
        Path to master parquet file containing 'rotation correction' column (in radians)
        (e.g., "/path/to/metadata/ON4PB_raw.parquet")
    max_workers : int, optional
        Maximum number of parallel workers for image processing (default: 4)
    force_overwrite : bool, optional
        Whether to reprocess existing images (default: False)
        If False, checks if output folder exists and has same number of images as input
        - If complete: skips processing
        - If incomplete: continues from where it stopped
        
    Notes:
    ------
    - Only processes folders matching "*_undistorted_frames" pattern
    - Extracts AFN from folder name (e.g., "154_undistorted_frames" -> AFN 154)
    - Looks up rotation correction for each AFN in master parquet
    - Creates new folders: "{AFN}_undistorted_rotcor_frames"
    - Rotates images using ffmpeg maintaining original quality and format
    - Preserves original filenames (e.g., 001.png, 002.png, etc.)
    - Rotation correction is in radians (positive = anticlockwise/counterclockwise, negative = clockwise)
    - FFmpeg rotate filter: positive angles rotate clockwise, opposite of user convention
    - Uses parallel processing for faster image rotation
    - Note: Stair-stepping/aliasing in black areas is normal for rotated images (resolution preserved)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    import shutil
    # Load master parquet
    try:
        df_master = pd.read_parquet(master_parquet_path, engine='fastparquet')
        print(f"Loaded master parquet: {master_parquet_path}")
    except Exception as e:
        print(f"Error loading master parquet: {e}")
        return
    
    # Check if 'rotation correction' column exists
    if 'rotation correction' not in df_master.columns:
        print(f"Error: 'rotation correction' column not found in master parquet")
        return
    
    # Find all folders matching "*_undistorted_frames" pattern
    # Only matches exact pattern: number_undistorted_frames (e.g., "154_undistorted_frames")
    pattern = os.path.join(frames_folder_path, "*_undistorted_frames")
    undistorted_folders = glob.glob(pattern)
    
    if not undistorted_folders:
        print(f"No folders matching *_undistorted_frames pattern found in {frames_folder_path}")
        return
    
    print(f"Found {len(undistorted_folders)} folders to process")
    
    # Process each folder
    for folder in sorted(undistorted_folders):
        folder_name = os.path.basename(folder)
        # Extract AFN number from folder name (e.g., "154_undistorted_frames" -> "154")
        # Pattern ensures exact match: number followed by "_undistorted_frames"
        afn_match = re.match(r'^(\d+)_undistorted_frames$', folder_name)
        if not afn_match:
            print(f"Warning: Could not extract AFN from folder name: {folder_name}")
            continue
        
        afn_str = afn_match.group(1)
        try:
            afn = int(afn_str)
        except ValueError:
            print(f"Warning: Invalid AFN number: {afn_str}")
            continue
        
        # Get rotation correction for this AFN from master parquet
        afn_row = df_master[df_master['AFN'] == afn]
        if afn_row.empty:
            print(f"Warning: AFN {afn} not found in master parquet, skipping")
            continue
        
        rotation_rad = afn_row['rotation correction'].iloc[0]
        
        # Check if rotation correction is valid
        if pd.isna(rotation_rad):
            print(f"Skipping AFN {afn}: rotation correction is NaN")
            continue
        
        rotation_rad = float(rotation_rad)
        
        # Create output folder: "{AFN}_undistorted_rotcor_frames" (e.g., "154_undistorted_rotcor_frames")
        output_folder = os.path.join(frames_folder_path, f"{afn}_undistorted_rotcor_frames")
        
        # Find all image files in folder (common formats)
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder, ext)))
        
        if not image_files:
            print(f"Warning: No images found in {folder}")
            continue
        
        image_files.sort()  # Process in order (e.g., 001.png, 002.png, ...)
        num_images = len(image_files)
        
        # Check if rotation is exactly 0 - if so, just copy the folder
        if rotation_rad == 0.0:
            print(f"\nProcessing AFN {afn}: rotation = 0 rad (no rotation needed)")
            print(f"Output folder: {output_folder}")
            
            # Check if output folder exists and compare image counts
            if not force_overwrite and os.path.exists(output_folder):
                output_images = []
                for ext in image_extensions:
                    output_images.extend(glob.glob(os.path.join(output_folder, ext)))
                output_num = len(output_images)
                
                if output_num == num_images and output_num > 0:
                    print(f"  ✓ Already processed ({output_num}/{num_images} images) - skipping")
                    continue
                elif output_num > 0:
                    print(f"  ⚠ Incomplete processing detected ({output_num}/{num_images} images) - copying all")
                else:
                    print(f"  ⚠ Empty output folder detected - copying all")
            
            # Copy entire folder (rotation is 0, so no processing needed)
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)  # Remove existing folder
            
            shutil.copytree(folder, output_folder)
            print(f"  ✓ Copied {num_images} images (no rotation applied)")
            continue
        
        # Rotation is non-zero - apply rotation correction
        rotation_deg = math.degrees(rotation_rad)
        print(f"\nProcessing AFN {afn}: rotation = {rotation_rad:.4f} rad ({rotation_deg:.2f}°)")
        print(f"Output folder: {output_folder}")
        
        # Check if output folder exists and compare image counts (for force_overwrite logic)
        if not force_overwrite and os.path.exists(output_folder):
            output_images = []
            for ext in image_extensions:
                output_images.extend(glob.glob(os.path.join(output_folder, ext)))
            output_num = len(output_images)
            
            if output_num == num_images and output_num > 0:
                print(f"  ✓ Already processed ({output_num}/{num_images} images) - skipping")
                continue
            elif output_num > 0:
                print(f"  ⚠ Incomplete processing detected ({output_num}/{num_images} images) - reprocessing")
            else:
                print(f"  ⚠ Empty output folder detected - reprocessing")
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Get list of already processed images if continuing from incomplete processing
        existing_images = set()
        if not force_overwrite and os.path.exists(output_folder):
            for ext in image_extensions:
                existing_images.update([os.path.basename(f) for f in glob.glob(os.path.join(output_folder, ext))])
        
        # Filter out already processed images if continuing
        if existing_images:
            image_files = [f for f in image_files if os.path.basename(f) not in existing_images]
            print(f"  Continuing: {len(image_files)} images remaining (skipping {len(existing_images)} already processed)")
        
        if not image_files:
            print(f"  ✓ All images already processed")
            continue
        
        print(f"Found {len(image_files)} images to rotate")
        
        # Define worker function for parallel processing
        def rotate_image_worker(args):
            """Worker function to rotate a single image with ffmpeg."""
            img_path, output_path, rotation_rad = args
            img_filename = os.path.basename(img_path)
            
            try:
                # Determine input format for quality preservation
                img_ext_lower = os.path.splitext(img_filename)[1].lower()
                
                # Build ffmpeg command with quality preservation
                # rotate filter: ow=iw:oh=ih keeps same dimensions (black corners will appear)
                # fillcolor=black sets black background for corners (default, but explicit for clarity)
                # FFmpeg rotate filter: positive = clockwise, opposite of user convention
                # Stair-stepping in black areas is normal visual artifact (resolution preserved)
                rotate_filter = f'rotate={-rotation_rad}:ow=iw:oh=ih:fillcolor=black'
                
                if img_ext_lower == '.png':
                    # PNG: Use PNG codec for lossless quality
                    cmd = [
                        'ffmpeg',
                        '-i', img_path,
                        '-vf', rotate_filter,  # Rotate with same dimensions (black corners)
                        '-c:v', 'png',  # PNG codec for lossless compression
                        '-pix_fmt', 'rgb24',  # Preserve RGB color format
                        '-y',  # Overwrite output file if exists
                        output_path
                    ]
                else:
                    # JPEG: Use high quality settings
                    cmd = [
                        'ffmpeg',
                        '-i', img_path,
                        '-vf', rotate_filter,  # Rotate with same dimensions (black corners)
                        '-q:v', '1',  # Highest quality (1-31, lower is better)
                        '-pix_fmt', 'rgb24',
                        '-y',
                        output_path
                    ]
                
                # Run ffmpeg (suppress output unless error)
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                if result.returncode != 0:
                    return False, f"Error rotating {img_filename}: {result.stderr[:100]}"
                else:
                    return True, img_filename
                    
            except Exception as e:
                return False, f"Error processing {img_filename}: {str(e)[:100]}"
        
        # Prepare arguments for parallel processing
        worker_args = [(img_path, os.path.join(output_folder, os.path.basename(img_path)), rotation_rad) 
                      for img_path in image_files]
        
        # Process images in parallel
        successful_images = 0
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_args = {executor.submit(rotate_image_worker, args): args for args in worker_args}
            
            # Process completed tasks
            for future in as_completed(future_to_args):
                success, message = future.result()
                if success:
                    successful_images += 1
                    if successful_images % 50 == 0:  # Progress update every 50 images
                        elapsed = time.time() - start_time
                        rate = successful_images / elapsed if elapsed > 0 else 0
                        print(f"    Processed {successful_images}/{len(image_files)} images ({rate:.1f} img/s)")
                else:
                    print(f"  ✗ {message}")
        
        elapsed = time.time() - start_time
        rate = successful_images / elapsed if elapsed > 0 else 0
        print(f"Completed AFN {afn}: {successful_images}/{len(image_files)} images rotated in {elapsed:.1f}s ({rate:.1f} img/s)")
    
    print("\nRotation correction complete!")


# ============================================================================
# GENERAL STATISTICS DISPLAY
# ============================================================================

def display_gen_stats(parquet_path_masters, parquet_path_info):
    """
    Display general statistics of the master dataset.
    
    Currently shows mean and standard deviation for slab density measurements
    (rho_1 … rho_4, combined slab, and rho_sub).
    
    Parameters
    ----------
    parquet_path_masters : str
        Path to the master data Parquet file (e.g. M3DC_raw.parquet)
        containing at least the columns rho_1, rho_2, rho_3, rho_4, and rho_sub.
    parquet_path_info : str
        Path to the master metadata Parquet file (e.g. M3DC_raw_info.parquet)
        with columns 'Abreviation' and 'Units'.
    """

    # --- load data -------------------------------------------------------
    try:
        masters = pd.read_parquet(parquet_path_masters, engine='fastparquet')
    except Exception as e:
        print(f"Error loading master data: {e}")
        return
    try:
        masters_info = pd.read_parquet(parquet_path_info, engine='fastparquet')
    except Exception as e:
        print(f"Error loading master info: {e}")
        return

    # --- helper: look up unit from masters_info --------------------------
    def _get_unit(col_name):
        row = masters_info.loc[masters_info['Abreviation'] == col_name, 'Units']
        if not row.empty:
            return row.values[0]
        return '–'

    # --- individual rho columns ------------------------------------------
    rho_cols = ['rho_1', 'rho_2', 'rho_3', 'rho_4']
    unit = _get_unit('rho_1')          # same unit for all rho columns

    header = "General Dataset Statistics"
    sep = "=" * 60

    print(f"\n{sep}")
    print(f"  {header}")
    print(f"{sep}\n")

    # Section: individual slab densities
    print(f"  Slab densities (100 ml cutter)  [{unit}]")
    print(f"  {'-' * 46}")

    all_rho_values = []
    for col in rho_cols:
        if col in masters.columns:
            vals = masters[col].dropna()
            n = len(vals)
            mean = vals.mean()
            std = vals.std()
            all_rho_values.append(vals)
            print(f"    {col:<8}  mean = {mean:7.1f}  ±  {std:5.1f}   (n={n})")
        else:
            print(f"    {col:<8}  — column not found")

    # Section: all slab densities combined
    if all_rho_values:
        combined = pd.concat(all_rho_values, ignore_index=True)
        n_all = len(combined)
        mean_all = combined.mean()
        std_all = combined.std()
        print(f"    {'all':<8}  mean = {mean_all:7.1f}  ±  {std_all:5.1f}   (n={n_all})")

    print()

    # Section: substratum density
    sub_col = 'rho_sub'
    unit_sub = _get_unit(sub_col)
    print(f"  Substratum density  [{unit_sub}]")
    print(f"  {'-' * 46}")
    if sub_col in masters.columns:
        vals_sub = masters[sub_col].dropna()
        n_sub = len(vals_sub)
        if n_sub > 0:
            mean_sub = vals_sub.mean()
            std_sub = vals_sub.std()
            print(f"    {sub_col:<8}  mean = {mean_sub:7.1f}  ±  {std_sub:5.1f}   (n={n_sub})")
        else:
            print(f"    {sub_col:<8}  — no valid measurements")
    else:
        print(f"    {sub_col:<8}  — column not found")

    print()

    # Section: slope inclination
    phi_col = 'phi'
    unit_phi = _get_unit(phi_col)
    print(f"  Slope inclination  [{unit_phi}]")
    print(f"  {'-' * 46}")
    if phi_col in masters.columns:
        vals_phi = masters[phi_col].dropna().abs()
        n_phi = len(vals_phi)
        if n_phi > 0:
            mean_phi = vals_phi.mean()
            std_phi = vals_phi.std()
            print(f"    {phi_col:<8}  mean = {mean_phi:7.1f}  ±  {std_phi:5.1f}   (n={n_phi})")
        else:
            print(f"    {phi_col:<8}  — no valid measurements")
    else:
        print(f"    {phi_col:<8}  — column not found")

    print()

    # Section: temperatures
    unit_T = _get_unit('T_s1')
    print(f"  Temperatures  [{unit_T}]")
    print(f"  {'-' * 46}")

    # Slab temperature: average of T_s1 and T_s2 per experiment
    t_slab_cols = ['T_s1', 'T_s2']
    if all(c in masters.columns for c in t_slab_cols):
        T_slab = masters[t_slab_cols].mean(axis=1).dropna()
        n_slab_T = len(T_slab)
        if n_slab_T > 0:
            mean_slab_T = T_slab.mean()
            std_slab_T = T_slab.std()
            print(f"    {'T_slab':<16}  mean = {mean_slab_T:7.1f}  ±  {std_slab_T:5.1f}   (n={n_slab_T})  [avg of T_s1, T_s2]")
        else:
            print(f"    {'T_slab':<16}  — no valid measurements")
    else:
        missing = [c for c in t_slab_cols if c not in masters.columns]
        print(f"    {'T_slab':<16}  — column(s) not found: {missing}")

    # Weak layer temperature: average of T_s2 and T_s3 per experiment
    t_wl_cols = ['T_s2', 'T_s3']
    if all(c in masters.columns for c in t_wl_cols):
        T_wl = masters[t_wl_cols].mean(axis=1).dropna()
        n_wl_T = len(T_wl)
        if n_wl_T > 0:
            mean_wl_T = T_wl.mean()
            std_wl_T = T_wl.std()
            print(f"    {'T_wl':<16}  mean = {mean_wl_T:7.1f}  ±  {std_wl_T:5.1f}   (n={n_wl_T})  [avg of T_s2, T_s3]")
        else:
            print(f"    {'T_wl':<16}  — no valid measurements")
    else:
        missing = [c for c in t_wl_cols if c not in masters.columns]
        print(f"    {'T_wl':<16}  — column(s) not found: {missing}")

    print()

    # Section: mean slab height
    hs_col = 'h_s'
    unit_hs = _get_unit(hs_col)
    print(f"  Slab height  [{unit_hs}]")
    print(f"  {'-' * 46}")
    if hs_col in masters.columns:
        vals_hs = masters[hs_col].dropna()
        n_hs = len(vals_hs)
        if n_hs > 0:
            mean_hs = vals_hs.mean()
            std_hs = vals_hs.std()
            print(f"    {hs_col:<8}  mean = {mean_hs:7.1f}  ±  {std_hs:5.1f}   (n={n_hs})")
        else:
            print(f"    {hs_col:<8}  — no valid measurements")
    else:
        print(f"    {hs_col:<8}  — column not found")

    print()

    # Section: sample length
    L_col = 'L'
    unit_L = _get_unit(L_col)
    print(f"  Sample length  [{unit_L}]")
    print(f"  {'-' * 46}")
    if L_col in masters.columns:
        vals_L = masters[L_col].dropna()
        n_L = len(vals_L)
        if n_L > 0:
            mean_L = vals_L.mean()
            std_L = vals_L.std()
            print(f"    {L_col:<8}  mean = {mean_L:7.1f}  ±  {std_L:5.1f}   (n={n_L})")
        else:
            print(f"    {L_col:<8}  — no valid measurements")
    else:
        print(f"    {L_col:<8}  — column not found")

    print()

    # Section: experiment dates
    date_col = 'date'
    print(f"  Experiment dates")
    print(f"  {'-' * 46}")
    if date_col in masters.columns:
        dates = pd.to_datetime(masters[date_col], errors='coerce').dropna()
        date_counts = dates.dt.date.value_counts().sort_index()
        for d, count in date_counts.items():
            print(f"    {d}   n = {count}")
        print(f"    {'—' * 30}")
        print(f"    {'total':<12}   n = {date_counts.sum()}")
    else:
        print(f"    {date_col:<8}  — column not found")

    print(f"\n{sep}\n")
