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
import ast
import tempfile

# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

# SMB Server connection
SMB_SERVER_URL =  'smb://ismd-server.synology.me/Public'

# Cross-platform server paths (check your operating system and update the paths)
SERVER_PATHS = {
    'macos': '/Volumes/Public/04 phds/Adam/02_experiments/04_mode_III_DC',
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

def calculate_err_weac(
    parquet_path_masters,
    parquet_path_info,
    force_overwrite=False,
    rho_wl=138.3,
    h_wl=4.5,
    E_wl=1.0,
    nu_wl=0.15,
    h_wl_err=1.0,
    h_s_err=2.5,
    a_err=5.0,
    l_dw_err=5.0,
    L_err=5.0,
    rho_err=1.0,
    E_wl_err=0.25,
    nu_wl_err=0.05,
    pc_rel_err=0.01,
    phi_err=1.0,
    theta_err=1.0,
    total_weights_rel_err=0.01,
):
    """
    Calculate WEAC differential fracture energies and their propagated uncertainties.

    This function computes Gc, G1c, G2c and G3c using WEAC for each valid row in the
    master parquet. It also computes uncertainty columns
    (Gc_uncertainty, G1c_uncertainty, G2c_uncertainty, G3c_uncertainty)
    via first-order error propagation with independent input uncertainties.
    Because parquet cannot store uncertainty objects, nominal values and one-sigma
    uncertainties are stored in separate float columns.

    Parameters
    ----------
    parquet_path_masters : str
        Path to the master data parquet (e.g. M3DC_raw.parquet).
    parquet_path_info : str
        Path to the master metadata parquet (e.g. M3DC_raw_info.parquet).
    force_overwrite : bool, default False
        If True, recompute and overwrite Gc/G1c/G2c/G3c for all rows with valid
        inputs. If False, only rows with missing value/uncertainty columns are computed.
    rho_wl : float, default 138.3
        Weak-layer density in kg/m^3.
    h_wl : float, default 4.5
        Weak-layer thickness in mm.
    E_wl : float, default 1.0
        Weak-layer Young's modulus in MPa.
    nu_wl : float, default 0.15
        Weak-layer Poisson ratio (-).
    h_wl_err : float, default 1.0
        Absolute uncertainty of weak-layer thickness (mm).
    h_s_err : float, default 2.5
        Absolute uncertainty of slab height h_s (mm).
    a_err : float, default 5.0
        Absolute uncertainty of crack length a (mm).
    l_dw_err : float, default 2.5
        Absolute uncertainty of l_dw (mm).
    L_err : float, default 2.5
        Absolute uncertainty of total slab length L (mm).
    rho_err : float, default 1.0
        Absolute uncertainty of slab densities rho_1 to rho_4 (kg/m^3).
    E_wl_err : float, default 0.25
        Absolute uncertainty of weak-layer Young's modulus (MPa).
    nu_wl_err : float, default 0.05
        Absolute uncertainty of weak-layer Poisson ratio (-).
    pc_rel_err : float, default 0.01
        Relative uncertainty of P_c[1] (1% = 0.01).
    phi_err : float, default 1.0
        Absolute uncertainty of inclination angle phi (degrees).
    theta_err : float, default 1.0
        Absolute uncertainty of in-plane angle theta (degrees). The nominal
        value is kept at theta=0 in ScenarioConfig.
    total_weights_rel_err : float, default 0.01
        Relative uncertainty of total weights used for surface-load calculation
        (1% = 0.01).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Updated (master_data, master_info) DataFrames.
    """
    try:
        from weac.components import Layer, Config, ScenarioConfig, ModelInput, WeakLayer, Segment
        from weac.core.system_model import SystemModel
        from weac.analysis.analyzer import Analyzer
        from uncertainties import ufloat
    except Exception as e:
        print(f"Error importing WEAC/uncertainties modules: {e}")
        return None, None

    try:
        df_masters = pd.read_parquet(parquet_path_masters, engine='fastparquet')
        print(f"Loaded master data from: {parquet_path_masters}")
    except Exception as e:
        print(f"Error loading master data {parquet_path_masters}: {e}")
        return None, None

    try:
        df_masters_info = pd.read_parquet(parquet_path_info, engine='fastparquet')
        print(f"Loaded master metadata from: {parquet_path_info}")
    except Exception as e:
        print(f"Error loading master metadata {parquet_path_info}: {e}")
        return None, None

    required_columns = [
        'phi', 'h_s', 'rho_1', 'rho_2', 'rho_3', 'rho_4',
        'P_c', 'a', 'L', 'weight number', 'l_dw'
    ]
    missing_required = [col for col in required_columns if col not in df_masters.columns]
    if missing_required:
        print(f"Error: Missing required columns in master data: {missing_required}")
        return df_masters, df_masters_info

    def _atomic_write_parquet(df_to_write, target_path):
        """Write parquet to a temp file in the same directory and atomically replace target."""
        target_dir = os.path.dirname(target_path) or "."
        os.makedirs(target_dir, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".parquet", dir=target_dir)
        os.close(fd)
        try:
            df_to_write.to_parquet(temp_path, index=False, engine='fastparquet')
            os.replace(temp_path, target_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def _parse_list_like(value):
        """Convert list-like strings to Python lists while keeping non-list scalars unchanged."""
        if isinstance(value, (list, tuple, np.ndarray)):
            return list(value)
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, (list, tuple, np.ndarray)):
                    return list(parsed)
            except Exception:
                return None
        return None

    def _ensure_positive(value, minimum=1e-9):
        return max(float(value), minimum)

    def _clip_nu(value):
        return float(np.clip(value, 1e-6, 0.499))

    def _extract_pc_force(value):
        """Return fracture force from P_c[1], or NaN if unavailable."""
        parsed = _parse_list_like(value)
        if parsed is None or len(parsed) < 2:
            return np.nan
        try:
            return float(parsed[1])
        except Exception:
            return np.nan

    def _afn_as_int(value):
        """Return AFN as int if possible, else None."""
        try:
            if pd.isna(value):
                return None
            return int(float(value))
        except Exception:
            return None

    def _compute_load_vector(experiment, pc_force, h_s_value, rho_values, phi_value, include_loading_head_mass=True):
        # Logic kept equivalent to notebook equations.
        phi = float(phi_value)
        l_lh = 50.0  # loading-head length [mm]

        g = 9810.0  # mm/s^2
        m_lh = 0.5404 * 1e-3  # loading-head mass [tonne]
        if not include_loading_head_mass:
            m_lh = 0.0
        h_s_val = float(h_s_value)
        rho_vals = np.array(rho_values, dtype=float)

        m_rs = h_s_val * 290 * 50 * np.mean(rho_vals) * 1e-12  # remaining snow mass [tonne]
        l_rs_z = np.sum(rho_vals * np.linspace(-3 / 8, 3 / 8, 4) * h_s_val) / np.sum(rho_vals)

        F_x = -(g * (m_lh + m_rs) * np.sin(np.deg2rad(phi)))
        F_z = g * (m_lh + m_rs) * np.cos(np.deg2rad(phi))
        F_y = -pc_force
        M_x = 0.0
        M_y = (
            -m_rs * g * np.sin(np.deg2rad(phi)) * l_rs_z
            + m_rs * g * np.cos(np.deg2rad(phi)) * l_lh / 2
            + m_lh * g * np.cos(np.deg2rad(phi)) * l_lh
        )
        M_z = pc_force * l_lh

        return np.array([F_x, F_y, F_z, M_x, M_y, M_z], dtype=float)

    def _setup_weac_and_calculate_err(experiment, pc_force, h_s_value, rho_values, h_wl_value, E_wl_value, nu_wl_value, l_dw_value, L_value, phi_value, a_value, theta_value, include_loading_head_mass=True):
        layers = [
            Layer(rho=float(rho_values[0]), h=float(h_s_value) / 4),
            Layer(rho=float(rho_values[1]), h=float(h_s_value) / 4),
            Layer(rho=float(rho_values[2]), h=float(h_s_value) / 4),
            Layer(rho=float(rho_values[3]), h=float(h_s_value) / 4),
        ]

        weak_layer = WeakLayer(
            rho=float(rho_wl),
            h=float(h_wl_value),
            E=float(E_wl_value),
            nu=float(nu_wl_value),
            constitutive_model='PlaneStrain'
        )

        phi = float(phi_value)
        weight_number = experiment['weight number']
        a = float(a_value)
        l_total = float(L_value)

        if pd.notna(weight_number) and phi >= 0:
            l_loading_head = 50.0
            l_dw = float(l_dw_value)
            l_load = float(weight_number) * 50.0
            l1 = l_dw - l_loading_head

            if a < (l_load + l_dw):
                l2 = a - l_dw
                l3 = l_load + l_dw - a
                l4 = l_total - (l_load + l_dw)
                if l1 > 0:
                    lengths = [l1, l2, l3, l4]
                    is_bedded = [False, False, True, True]
                    is_loaded = [False, True, True, False]
                else:
                    lengths = [l2, l3, l4]
                    is_bedded = [False, True, True]
                    is_loaded = [True, True, False]
            else:
                l2 = l_load
                l3 = a - l_load - l_dw
                l4 = l_total - a
                if l1 > 0:
                    lengths = [l1, l2, l3, l4]
                    is_bedded = [False, False, False, True]
                    is_loaded = [False, True, False, False]
                else:
                    lengths = [l2, l3, l4]
                    is_bedded = [False, False, True]
                    is_loaded = [True, False, False]

        elif pd.notna(weight_number) and phi < 0:
            l_dw = float(l_dw_value)
            l_load = float(weight_number) * 50.0
            l1 = a
            l2 = l_total - a - l_dw - l_load
            l3 = l_load
            l4 = l_dw
            lengths = [l1, l2, l3, l4]
            is_bedded = [False, True, True, True]
            is_loaded = [False, False, True, False]
        else:
            lengths = [a, l_total - a]
            is_bedded = [False, True]
            is_loaded = [False, False]

        segments = [
            Segment(length=l, has_foundation=b, is_loaded=ld, m=0)
            for l, b, ld in zip(lengths, is_bedded, is_loaded)
        ]

        total_weights = experiment['total weights'] if 'total weights' in experiment.index else np.nan
        # Protect against invalid or zero loaded length in the denominator.
        if pd.notna(weight_number) and pd.notna(total_weights) and float(weight_number) > 0:
            surface_load = float(total_weights) * 1e-3 * 9810 / (float(weight_number) * 50 * 290)
        else:
            surface_load = 0.0

        scenario = ScenarioConfig(
            system_type='-pst',
            phi=phi,
            theta=float(theta_value),
            b=290,
            surface_load=surface_load,
            load_vector_left=_compute_load_vector(
                experiment,
                pc_force,
                h_s_value,
                rho_values,
                phi,
                include_loading_head_mass=include_loading_head_mass,
            ),
            load_vector_right=np.array([0, 0, 0, 0, 0, 0], dtype=float),
        )
        config = Config(touchdown=False, backend='generalized')
        model_input = ModelInput(
            layers=layers,
            weak_layer=weak_layer,
            segments=segments,
            scenario_config=scenario
        )
        system = SystemModel(model_input=model_input, config=config)
        analyzer = Analyzer(system_model=system)
        gdif = analyzer.differential_ERR(unit='J/m^2')

        return np.array(gdif, dtype=float)

    df_updated = df_masters.copy()
    df_info_updated = df_masters_info.copy()

    g_columns = ['Gc', 'G1c', 'G2c', 'G3c']
    g_unc_columns = [f"{col}_uncertainty" for col in g_columns]
    surface_load_col = 'surface_load'
    surface_load_unc_col = 'surface_load_uncertainty'

    for col in g_columns:
        if col not in df_updated.columns:
            df_updated[col] = np.nan
            print(f"Initialized missing column: {col}")
    for col in g_unc_columns:
        if col not in df_updated.columns:
            df_updated[col] = np.nan
            print(f"Initialized missing column: {col}")

    if surface_load_col not in df_updated.columns:
        df_updated[surface_load_col] = np.nan
        print(f"Initialized missing column: {surface_load_col}")
    if surface_load_unc_col not in df_updated.columns:
        df_updated[surface_load_unc_col] = np.nan
        print(f"Initialized missing column: {surface_load_unc_col}")

    # Compute and persist slope-normal surface load p_w and its propagated uncertainty.
    # p_w = m*g / (n_w * 50 * 290), with g=9.81 m/s^2 -> N/mm^2.
    total_weights_series = pd.to_numeric(df_updated['total weights'], errors='coerce')
    weight_number_series = pd.to_numeric(df_updated['weight number'], errors='coerce')
    valid_surface = total_weights_series.notna() & weight_number_series.notna() & (weight_number_series > 0)

    surface_load_values = pd.Series(0.0, index=df_updated.index, dtype='float64')
    surface_load_values.loc[valid_surface] = (
        total_weights_series.loc[valid_surface] * 9.81
        / (weight_number_series.loc[valid_surface] * 50.0 * 290.0)
    )

    sigma_m = total_weights_series.abs() * abs(float(total_weights_rel_err))
    sigma_n = 0.0
    surface_load_unc_values = pd.Series(0.0, index=df_updated.index, dtype='float64')
    if (sigma_m > 0.0).any() or sigma_n > 0.0:
        dp_dm = pd.Series(0.0, index=df_updated.index, dtype='float64')
        dp_dn = pd.Series(0.0, index=df_updated.index, dtype='float64')
        dp_dm.loc[valid_surface] = 9.81 / (weight_number_series.loc[valid_surface] * 50.0 * 290.0)
        dp_dn.loc[valid_surface] = (
            -total_weights_series.loc[valid_surface] * 9.81
            / ((weight_number_series.loc[valid_surface] ** 2) * 50.0 * 290.0)
        )
        surface_load_unc_values.loc[valid_surface] = np.sqrt(
            (dp_dm.loc[valid_surface] * sigma_m.loc[valid_surface]) ** 2 +
            (dp_dn.loc[valid_surface] * sigma_n) ** 2
        )

    df_updated[surface_load_col] = surface_load_values.to_numpy(dtype=float)
    df_updated[surface_load_unc_col] = surface_load_unc_values.to_numpy(dtype=float)

    pst_id_col = 'AFM' if 'AFM' in df_updated.columns else ('AFN' if 'AFN' in df_updated.columns else None)
    pst_afn_set = {1, 2, 3, 4, 5, 6, 7}
    if pst_id_col is not None:
        pst_mask = df_updated[pst_id_col].apply(_afn_as_int).isin(pst_afn_set)
    else:
        pst_mask = pd.Series(False, index=df_updated.index)

    # For PST ids (1..7): if P_c is missing/non-numeric, enforce explicit zero in stored data.
    pst_pc_zeroed = 0
    if 'P_c' in df_updated.columns and pst_mask.any():
        for idx in df_updated.index[pst_mask]:
            value = df_updated.at[idx, 'P_c']
            pc_force_val = _extract_pc_force(value)
            if pd.isna(pc_force_val):
                already_zero_scalar = False
                try:
                    scalar_val = float(value)
                    already_zero_scalar = bool(np.isfinite(scalar_val) and abs(scalar_val) < 1e-12)
                except Exception:
                    already_zero_scalar = False
                if not already_zero_scalar:
                    df_updated.at[idx, 'P_c'] = [0.0, 0.0]
                    pst_pc_zeroed += 1

    if force_overwrite:
        target_mask = pd.Series(True, index=df_updated.index)
        print("force_overwrite=True: recomputing G-values and uncertainties for all rows with valid inputs.")
    else:
        # Always update AFN 1..7 as PST rows (P_c forced to 0), even if values already exist.
        target_mask = df_updated[g_columns + g_unc_columns].isna().any(axis=1) | pst_mask
        print("force_overwrite=False: computing only rows with missing G-values/uncertainties.")
    if pst_mask.any():
        print(
            f"PST override active for {pst_id_col} in {sorted(pst_afn_set)}: "
            "calculating with P_c[1]=0 (same PST setup, without out-of-plane load from fracture force)."
        )

    computed = 0
    skipped_invalid_pc = 0
    skipped_no_target = 0
    failed = 0
    pst_computed = 0

    for idx, experiment in df_updated.iterrows():
        if not bool(target_mask.loc[idx]):
            skipped_no_target += 1
            continue

        afn_int = _afn_as_int(experiment[pst_id_col]) if pst_id_col in experiment.index else None
        is_pst_afn = afn_int in pst_afn_set

        if is_pst_afn:
            pc_force = 0.0
        else:
            pc_force = _extract_pc_force(experiment['P_c']) if 'P_c' in experiment.index else np.nan
            if pd.isna(pc_force):
                skipped_invalid_pc += 1
                continue

        try:
            # Build uncertain scalar inputs (independent uncertainties).
            pc_u = ufloat(float(pc_force), abs(float(pc_force)) * abs(float(pc_rel_err)))
            h_s_u = ufloat(float(experiment['h_s']), abs(float(h_s_err)))
            rho_1_u = ufloat(float(experiment['rho_1']), abs(float(rho_err)))
            rho_2_u = ufloat(float(experiment['rho_2']), abs(float(rho_err)))
            rho_3_u = ufloat(float(experiment['rho_3']), abs(float(rho_err)))
            rho_4_u = ufloat(float(experiment['rho_4']), abs(float(rho_err)))
            h_wl_u = ufloat(float(h_wl), abs(float(h_wl_err)))
            E_wl_u = ufloat(float(E_wl), abs(float(E_wl_err)))
            nu_wl_u = ufloat(float(nu_wl), abs(float(nu_wl_err)))
            phi_u = ufloat(float(experiment['phi']), abs(float(phi_err)))
            theta_u = ufloat(0.0, abs(float(theta_err)))
            a_u = ufloat(float(experiment['a']), abs(float(a_err)))

            l_dw_nominal = float(experiment['l_dw']) if pd.notna(experiment['l_dw']) else 0.0
            L_nominal = float(experiment['L']) if pd.notna(experiment['L']) else 10000.0
            l_dw_u = ufloat(l_dw_nominal, abs(float(l_dw_err)) if pd.notna(experiment['l_dw']) else 0.0)
            L_u = ufloat(L_nominal, abs(float(L_err)) if pd.notna(experiment['L']) else 0.0)

            base_values = {
                'pc_force': float(pc_u.nominal_value),
                'h_s': _ensure_positive(h_s_u.nominal_value),
                'rho_1': float(rho_1_u.nominal_value),
                'rho_2': float(rho_2_u.nominal_value),
                'rho_3': float(rho_3_u.nominal_value),
                'rho_4': float(rho_4_u.nominal_value),
                'h_wl': _ensure_positive(h_wl_u.nominal_value),
                'E_wl': _ensure_positive(E_wl_u.nominal_value),
                'nu_wl': _clip_nu(nu_wl_u.nominal_value),
                'phi': float(phi_u.nominal_value),
                'theta': float(theta_u.nominal_value),
                'a': _ensure_positive(a_u.nominal_value),
                'l_dw': _ensure_positive(l_dw_u.nominal_value),
                'L': _ensure_positive(L_u.nominal_value),
            }

            def _solve_with(values):
                rho_vals = [values['rho_1'], values['rho_2'], values['rho_3'], values['rho_4']]
                return _setup_weac_and_calculate_err(
                    experiment=experiment,
                    pc_force=values['pc_force'],
                    h_s_value=values['h_s'],
                    rho_values=rho_vals,
                    h_wl_value=values['h_wl'],
                    E_wl_value=values['E_wl'],
                    nu_wl_value=values['nu_wl'],
                    phi_value=values['phi'],
                    a_value=values['a'],
                    theta_value=values['theta'],
                    l_dw_value=values['l_dw'],
                    L_value=values['L'],
                    include_loading_head_mass=not is_pst_afn,
                )

            gdif_nom = _solve_with(base_values)
            if len(gdif_nom) < 4:
                raise ValueError("WEAC returned fewer than 4 fracture-energy values.")

            # First-order propagation with independent inputs:
            # sigma_G^2 = sum_i ((dG/dx_i) * sigma_x_i)^2
            uncertainty_specs = {
                'pc_force': float(pc_u.std_dev),
                'h_s': float(h_s_u.std_dev),
                'rho_1': float(rho_1_u.std_dev),
                'rho_2': float(rho_2_u.std_dev),
                'rho_3': float(rho_3_u.std_dev),
                'rho_4': float(rho_4_u.std_dev),
                'h_wl': float(h_wl_u.std_dev),
                'E_wl': float(E_wl_u.std_dev),
                'nu_wl': float(nu_wl_u.std_dev),
                'phi': float(phi_u.std_dev),
                'theta': float(theta_u.std_dev),
                'a': float(a_u.std_dev),
                'l_dw': float(l_dw_u.std_dev),
                'L': float(L_u.std_dev),
            }

            variance = np.zeros(4, dtype=float)
            for var_name, sigma in uncertainty_specs.items():
                if sigma <= 0:
                    continue

                center = float(base_values[var_name])
                plus_values = dict(base_values)
                minus_values = dict(base_values)
                plus_values[var_name] = center + sigma
                minus_values[var_name] = center - sigma

                if var_name in ('h_s', 'h_wl', 'E_wl', 'a', 'l_dw', 'L'):
                    plus_values[var_name] = _ensure_positive(plus_values[var_name])
                    minus_values[var_name] = _ensure_positive(minus_values[var_name])
                elif var_name == 'nu_wl':
                    plus_values[var_name] = _clip_nu(plus_values[var_name])
                    minus_values[var_name] = _clip_nu(minus_values[var_name])

                if plus_values[var_name] == minus_values[var_name]:
                    continue

                g_plus = None
                g_minus = None
                try:
                    g_plus = _solve_with(plus_values)
                except Exception:
                    pass
                try:
                    g_minus = _solve_with(minus_values)
                except Exception:
                    pass

                if g_plus is not None and g_minus is not None:
                    deriv = (g_plus[:4] - g_minus[:4]) / (plus_values[var_name] - minus_values[var_name])
                elif g_plus is not None:
                    step = plus_values[var_name] - center
                    if step == 0:
                        continue
                    deriv = (g_plus[:4] - gdif_nom[:4]) / step
                elif g_minus is not None:
                    step = center - minus_values[var_name]
                    if step == 0:
                        continue
                    deriv = (gdif_nom[:4] - g_minus[:4]) / step
                else:
                    continue

                variance += (deriv * sigma) ** 2

            gdif_unc = np.sqrt(variance)

            df_updated.loc[idx, g_columns] = gdif_nom[:4]
            df_updated.loc[idx, g_unc_columns] = gdif_unc[:4]
            computed += 1
            if is_pst_afn:
                pst_computed += 1
        except Exception as e:
            afn_val = experiment['AFN'] if 'AFN' in experiment.index else idx
            print(f"Warning: Could not compute WEAC ERR for AFN/index {afn_val}: {e}")
            failed += 1

    metadata_rows = {
        'Gc': {
            'units': 'J/m^2',
            'data_type': 'Float',
            'long_name': 'critical differential fracture energy',
            'description': 'Differential fracture energy at crack initiation from WEAC (Mode III).'
        },
        'G1c': {
            'units': 'J/m^2',
            'data_type': 'Float',
            'long_name': 'mode I differential fracture energy',
            'description': 'Mode I component of differential fracture energy at crack initiation from WEAC.'
        },
        'G2c': {
            'units': 'J/m^2',
            'data_type': 'Float',
            'long_name': 'mode II differential fracture energy',
            'description': 'Mode II component of differential fracture energy at crack initiation from WEAC.'
        },
        'G3c': {
            'units': 'J/m^2',
            'data_type': 'Float',
            'long_name': 'mode III differential fracture energy',
            'description': 'Mode III component of differential fracture energy at crack initiation from WEAC.'
        },
        'Gc_uncertainty': {
            'units': 'J/m^2',
            'data_type': 'Float',
            'long_name': 'uncertainty of critical differential fracture energy',
            'description': 'Propagated one-sigma uncertainty of Gc from independent input uncertainties.'
        },
        'G1c_uncertainty': {
            'units': 'J/m^2',
            'data_type': 'Float',
            'long_name': 'uncertainty of mode I differential fracture energy',
            'description': 'Propagated one-sigma uncertainty of G1c from independent input uncertainties.'
        },
        'G2c_uncertainty': {
            'units': 'J/m^2',
            'data_type': 'Float',
            'long_name': 'uncertainty of mode II differential fracture energy',
            'description': 'Propagated one-sigma uncertainty of G2c from independent input uncertainties.'
        },
        'G3c_uncertainty': {
            'units': 'J/m^2',
            'data_type': 'Float',
            'long_name': 'uncertainty of mode III differential fracture energy',
            'description': 'Propagated one-sigma uncertainty of G3c from independent input uncertainties.'
        },
        'surface_load': {
            'units': 'N/mm^2',
            'data_type': 'Float',
            'long_name': 'slope-normal surface load',
            'description': 'Surface load used in WEAC scenario configuration: m*g/(n_w*50*290).'
        },
        'surface_load_uncertainty': {
            'units': 'N/mm^2',
            'data_type': 'Float',
            'long_name': 'uncertainty of slope-normal surface load',
            'description': 'Propagated one-sigma uncertainty of surface_load (hardcoded as zero; no input uncertainty for total weights and weight number).'
        }
    }

    if force_overwrite and 'Abreviation' in df_info_updated.columns:
        duplicate_metadata_columns = df_info_updated[
            df_info_updated.duplicated(subset=['Abreviation'], keep=False)
        ]['Abreviation'].unique()
        if len(duplicate_metadata_columns) > 0:
            print(f"Found duplicate metadata abbreviations: {list(duplicate_metadata_columns)}")
            df_info_updated = df_info_updated.drop_duplicates(subset=['Abreviation'], keep='first')
            print("Removed duplicate metadata entries (kept first occurrence).")

    for col_name, col_info in metadata_rows.items():
        existing_row = df_info_updated[df_info_updated['Abreviation'] == col_name]
        if existing_row.empty:
            new_row = pd.DataFrame({
                'Abreviation': [col_name],
                'Units': [col_info['units']],
                'Data_Type': [col_info['data_type']],
                'Long Name': [col_info['long_name']],
                'Description': [col_info['description']]
            })
            df_info_updated = pd.concat([df_info_updated, new_row], ignore_index=True)
            print(f"Added metadata row for {col_name}")
        else:
            row_idx = existing_row.index[0]
            df_info_updated.at[row_idx, 'Units'] = col_info['units']
            df_info_updated.at[row_idx, 'Data_Type'] = col_info['data_type']
            df_info_updated.at[row_idx, 'Long Name'] = col_info['long_name']
            df_info_updated.at[row_idx, 'Description'] = col_info['description']
            print(f"Updated metadata row for {col_name}")

    df_info_updated = df_info_updated.sort_values('Abreviation', ascending=True).reset_index(drop=True)

    try:
        _atomic_write_parquet(df_updated, parquet_path_masters)
        _atomic_write_parquet(df_info_updated, parquet_path_info)
        print("Saved updated master parquet files using atomic write.")
    except Exception as e:
        print(f"Error while writing parquet files atomically: {e}")
        return None, None

    print("\nWEAC ERR calculation summary:")
    print(f"  - Rows computed (values + uncertainties): {computed}")
    print(f"  - Rows computed with PST override (AFN 1..7, P_c[1]=0): {pst_computed}")
    print(f"  - PST rows with invalid P_c normalized to [0.0, 0.0]: {pst_pc_zeroed}")
    print("  - Updated columns: surface_load, surface_load_uncertainty (N/mm^2)")
    print(f"  - Rows skipped (no target update required): {skipped_no_target}")
    print(f"  - Rows skipped (invalid/missing P_c[1]): {skipped_invalid_pc}")
    print(f"  - Rows failed during WEAC solve: {failed}")
    print(f"  - Propagated angle uncertainty: phi ±{float(phi_err):g} deg")
    print(f"  - Surface-load uncertainty: total weights relative error ±{100.0 * abs(float(total_weights_rel_err)):g}%")

    return df_updated, df_info_updated


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

def explore_parquet_data(afn, column_name, raw_data_path, raw_info_data_path):
    """
    Display metadata and value for a specific AFN and column from parquet files.
    
    Parameters:
    -----------
    raw_data_path : str
        Filepath to master Parquet data file (e.g., 'ON4PB_raw.parquet')
    raw_info_data_path : str
        Filepath to master Parquet metadata file (e.g., 'ON4PB_raw_info.parquet')
    afn : int
        AFN (Automated File Number) to display
    column_name : str
        Column name to display
        
    Returns:
    --------
    None
        Prints formatted output with metadata and value
        
    Example:
    --------
    >>> explore_parquet_data('ON4PB_raw.parquet', 'ON4PB_raw_info.parquet', 1001, 'sample_geometry_FEM')
    """
    # Load parquet files
    try:
        df_data = pd.read_parquet(raw_data_path, engine='fastparquet')
    except Exception as e:
        print(f"Error loading data file {raw_data_path}: {e}")
        return None
    
    try:
        df_info = pd.read_parquet(raw_info_data_path, engine='fastparquet')
    except Exception as e:
        print(f"Error loading metadata file {raw_info_data_path}: {e}")
        return None
    
    # Get available columns
    available_columns = sorted(df_data.columns.tolist())
    
    # Get data row for selected AFN
    afn_row = df_data[df_data['AFN'] == afn]
    if afn_row.empty:
        print(f"Error: AFN {afn} not found in data")
        return None
    
    afn_data = afn_row.iloc[0]
    
    # Check if column exists
    if column_name not in df_data.columns:
        print(f"Error: Column '{column_name}' not found in data")
        print()
        print("Available columns:")
        print("-" * 80)
        for col in available_columns:
            print(f"  {col}")
        return None
    
    # Get value
    value = afn_data[column_name]
    
    # Get metadata from info parquet
    info_row = df_info[df_info['Abreviation'] == column_name]
    
    # Format value for display
    def format_value(val):
        """Format a value for display, handling various data types."""
        # Check for None first (before pd.isna which can fail on arrays)
        if val is None:
            return "None"
        
        # Handle lists and arrays first (before pd.isna which can fail on arrays)
        if isinstance(val, (list, tuple, np.ndarray)):
            if len(val) == 0:
                return "[] (empty)"
            
            # Check for NaN in arrays
            if isinstance(val, np.ndarray):
                if np.any(pd.isna(val)):
                    # Has some NaN values
                    val = val.tolist()
                else:
                    val = val.tolist()
            elif isinstance(val, tuple):
                val = list(val)
            
            # Format the full list
            formatted = str(val)
            return f"{formatted}\n  (Type: {type(val).__name__}, Length: {len(val)})"
        
        # For scalar values, check for NaN
        try:
            if pd.isna(val):
                return "NaN (missing data)"
        except (TypeError, ValueError):
            # pd.isna might fail for some types, continue
            pass
        
        # Handle strings
        if isinstance(val, str):
            if len(val) == 0:
                return '"" (empty string)'
            return f'"{val}"\n  (Type: str, Length: {len(val)})'
        
        # Handle numeric types
        if isinstance(val, (int, float, np.integer, np.floating)):
            return f"{val}\n  (Type: {type(val).__name__})"
        
        # Handle boolean
        if isinstance(val, bool):
            return f"{val}\n  (Type: bool)"
        
        # Default: convert to string
        return f"{val}\n  (Type: {type(val).__name__})"
    
    # Print formatted output
    print("=" * 80)
    print(f"AFN: {afn} | Column: {column_name}")
    print("=" * 80)
    print()
    
    # Display metadata if available
    if not info_row.empty:
        info_data = info_row.iloc[0]
        print("METADATA:")
        print("-" * 80)
        
        long_name = info_data.get('Long Name', 'N/A')
        units = info_data.get('Units', 'N/A')
        data_type = info_data.get('Data_Type', 'N/A')
        description = info_data.get('Description', 'N/A')
        
        print(f"  Long Name: {long_name}")
        print(f"  Units: {units}")
        print(f"  Data Type: {data_type}")
        print(f"  Description:")
        # Indent description lines
        for line in str(description).split('\n'):
            print(f"    {line}")
        print()
    else:
        print("METADATA:")
        print("-" * 80)
        print("  No metadata found for this column")
        print()
    
    # Display value
    print("VALUE:")
    print("-" * 80)
    formatted_value = format_value(value)
    for line in formatted_value.split('\n'):
        print(f"  {line}")
    print()
    print("=" * 80)
    print()
    
    # Always print available columns at the end
    print("AVAILABLE COLUMNS:")
    print("-" * 80)
    for col in available_columns:
        print(f"  {col}")
    print()
    print("=" * 80)
    
    return None


def opt_GI_GII_ODR(
        df, dim=1, gc0=.6, gc0_I=None, gc0_III=None, exp=2, var='B',
        indi=False, ifixb=[1, 1, 0, 0],
        print_results=True, verbose=False,
        same_exponent=False,
        n_bounds=(0.1, 10.0),
        m_bounds=(0.1, 10.0),
        gc_bounds=(1e-3, 10.0)):
    """
    Orthogonal distance regression for GI/GIII interaction law.

    This function is adapted from regression.py (Mode I/II version) with
    Mode II terms replaced by Mode III terms.
    """
    from uncertainties import unumpy
    from collections import defaultdict
    from itertools import product
    from scipy.odr import RealData, Model, ODR
    from scipy.stats import distributions

    # Accept DataFrame directly or parquet path.
    if isinstance(df, pd.DataFrame):
        df_local = df.copy()
    elif isinstance(df, (str, os.PathLike)):
        try:
            df_local = pd.read_parquet(df, engine='fastparquet')
        except Exception:
            df_local = pd.read_parquet(df)
    else:
        raise TypeError("opt_GI_GII_ODR: 'df' must be a pandas DataFrame or a parquet file path.")

    # Harmonize common project column names (G1c/G3c -> GIc/GIIIc).
    if 'GIc' not in df_local.columns and 'G1c' in df_local.columns:
        df_local['GIc'] = df_local['G1c']
    if 'GIIIc' not in df_local.columns and 'G3c' in df_local.columns:
        df_local['GIIIc'] = df_local['G3c']

    required_cols = ['GIc', 'GIIIc']
    missing_cols = [col for col in required_cols if col not in df_local.columns]
    if missing_cols:
        raise ValueError(f"opt_GI_GII_ODR: Missing required columns: {missing_cols}")

    def residual(beta, x, var='B', bounds=False):
        GIc, GIIIc, n, m = beta
        Gi, Giii = x
        eps = 1e-12
        GIc_s = max(float(GIc), eps)
        GIIIc_s = max(float(GIIIc), eps)
        n_s = max(float(n), eps)
        m_s = max(float(m), eps)
        if same_exponent:
            m_s = n_s
        Gi_s = np.clip(np.asarray(Gi, dtype=float), eps, None)
        Giii_s = np.clip(np.asarray(Giii, dtype=float), eps, None)

        if bounds:
            n_lo, n_hi = n_bounds
            m_lo, m_hi = m_bounds
            gc_lo, gc_hi = gc_bounds
            if not (gc_lo <= GIc_s <= gc_hi and gc_lo <= GIIIc_s <= gc_hi and n_lo <= n_s <= n_hi and m_lo <= m_s <= m_hi):
                return np.full_like(Gi_s, 1e3, dtype=float)

        with np.errstate(invalid='ignore'):
            if var == 'A':
                res = ((Gi_s / GIc_s) ** n_s + (Giii_s / GIIIc_s) ** m_s) ** (2 / (n_s + m_s)) - 1
            elif var == 'B':
                res = ((Gi_s / GIc_s) ** n_s + (Giii_s / GIIIc_s) ** m_s) - 1
            elif var == 'C':
                res = ((Gi_s / GIc_s) ** (1 / n_s) + (Giii_s / GIIIc_s) ** (1 / m_s)) - 1
            elif var == 'BK':
                denom = np.clip(Gi_s + Giii_s, eps, None)
                ratio = np.clip(Giii_s / denom, eps, 1.0)
                res = (GIc_s + (GIIIc_s - GIc_s) * ratio ** m_s) / denom - 1
            else:
                raise NotImplementedError(f'Criterion type {var} not implemented.')
        return res

    def param_jacobian(beta, x, var='B', *args):
        GIc, GIIIc, n, m = beta
        Gi, Giii = x
        eps = 1e-12
        GIc = max(float(GIc), eps)
        GIIIc = max(float(GIIIc), eps)
        n = max(float(n), eps)
        m = max(float(m), eps)
        Gi = np.clip(np.asarray(Gi, dtype=float), eps, None)
        Giii = np.clip(np.asarray(Giii, dtype=float), eps, None)

        with np.errstate(invalid='ignore'):
            if var == 'A':
                dGIc = -(2 * Gi * (Gi / GIc) ** (-1 + n) * ((Gi / GIc) ** n + (Giii / GIIIc) ** m) ** (-1 + 2 / (m + n)) * n) / (GIc ** 2 * (m + n))
                dGIIIc = -((2 * Giii * ((Gi / GIc) ** n + (Giii / GIIIc) ** m) ** (-1 + 2 / (m + n)) * (Giii / GIIIc) ** (-1 + m) * m) / (GIIIc ** 2 * (m + n)))
                dn = (((Gi / GIc) ** n + (Giii / GIIIc) ** m) ** (-1 + 2 / (m + n)) * (2 * (Gi / GIc) ** n * (m + n) * np.log(Gi / GIc) - 2 * ((Gi / GIc) ** n + (Giii / GIIIc) ** m) * np.log((Gi / GIc) ** n + (Giii / GIIIc) ** m))) / (m + n) ** 2
                dm = (((Gi / GIc) ** n + (Giii / GIIIc) ** m) ** (-1 + 2 / (m + n)) * (-2 * ((Gi / GIc) ** n + (Giii / GIIIc) ** m) * np.log((Gi / GIc) ** n + (Giii / GIIIc) ** m) + 2 * (Giii / GIIIc) ** m * (m + n) * np.log(Giii / GIIIc))) / (m + n) ** 2
            elif var == 'B':
                dGIc = -(((Gi / GIc) ** n * n) / GIc)
                dGIIIc = -(((Giii / GIIIc) ** m * m) / GIIIc)
                dn = (Gi / GIc) ** n * np.log(Gi / GIc)
                dm = (Giii / GIIIc) ** m * np.log(Giii / GIIIc)
            elif var == 'C':
                dGIc = -((Gi / GIc) ** (1 / n)) / (GIc * n)
                dGIIIc = -((Giii / GIIIc) ** (1 / m)) / (GIIIc * m)
                dn = -((Gi / GIc) ** (1 / n) * np.log(Gi / GIc)) / (n ** 2)
                dm = -((Giii / GIIIc) ** (1 / m) * np.log(Giii / GIIIc)) / (m ** 2)
            elif var == 'BK':
                dGIc = (1 - (Giii / (Gi + Giii)) ** m) / (Gi + Giii)
                dGIIIc = ((Giii / (Gi + Giii)) ** m) / (Gi + Giii)
                dn = np.zeros_like(Gi)
                dm = ((Giii / (Gi + Giii)) ** m * (GIIIc - GIc) * np.log(Giii / (Gi + Giii))) / (Gi + Giii)
            else:
                raise NotImplementedError(f'Criterion type {var} not implemented.')

        return np.row_stack([dGIc, dGIIIc, dn, dm])

    def value_jacobian(beta, x, var='B', *args):
        GIc, GIIIc, n, m = beta
        Gi, Giii = x
        eps = 1e-12
        GIc = max(float(GIc), eps)
        GIIIc = max(float(GIIIc), eps)
        n = max(float(n), eps)
        m = max(float(m), eps)
        Gi = np.clip(np.asarray(Gi, dtype=float), eps, None)
        Giii = np.clip(np.asarray(Giii, dtype=float), eps, None)

        if var == 'A':
            dGi = (2 * (Gi / GIc) ** n * ((Gi / GIc) ** n + (Giii / GIIIc) ** m) ** (-1 + 2 / (m + n)) * n) / (Gi * (m + n))
            dGiii = (2 * ((Gi / GIc) ** n + (Giii / GIIIc) ** m) ** (-1 + 2 / (m + n)) * (Giii / GIIIc) ** m * m) / (Giii * (m + n))
        elif var == 'B':
            dGi = ((Gi / GIc) ** n * n) / Gi
            dGiii = ((Giii / GIIIc) ** m * m) / Giii
        elif var == 'C':
            dGi = ((Gi / GIc) ** (1 / n)) / (n * Gi)
            dGiii = ((Giii / GIIIc) ** (1 / m)) / (m * Giii)
        elif var == 'BK':
            dGi = (-GIc + (1 + m) * (GIc - GIIIc) * (Giii / (Gi + Giii)) ** m) / (Gi + Giii) ** 2
            dGiii = (-GIc * Giii + (Giii - m * Gi) * (GIc - GIIIc) * (Giii / (Gi + Giii)) ** m) / (Giii * (Gi + Giii) ** 2)
        else:
            raise NotImplementedError(f'Criterion type {var} not implemented.')

        return np.row_stack([dGi, dGiii])

    def assemble_data(df_local, dim_local=1, min_std=1e-9):
        exp_local = np.row_stack(df_local[['GIc', 'GIIIc']].apply(unumpy.nominal_values).values.T).astype(float)
        std_local = np.row_stack(df_local[['GIc', 'GIIIc']].apply(unumpy.std_devs).values.T).astype(float)

        # Remove invalid rows and floor zero uncertainties to avoid ODR divide-by-zero.
        valid = (
            np.isfinite(exp_local[0]) & np.isfinite(exp_local[1]) &
            np.isfinite(std_local[0]) & np.isfinite(std_local[1])
        )
        exp_local = exp_local[:, valid]
        std_local = std_local[:, valid]
        std_local = np.clip(std_local, min_std, None)

        ndof_local = exp_local.shape[1] - 4
        return RealData(exp_local, y=dim_local, sx=std_local), ndof_local

    def get_initial_guesses(gc0_I_local=0.7, gc0_III_local=0.7, exp=2, indi=False, var='B', verbose=False):
        if isinstance(exp, tuple) and len(exp) == 2:
            n0 = [exp[0]]
            m0 = [exp[1]]
        elif isinstance(exp, (list, np.ndarray)):
            n0 = m0 = exp
        else:
            if var == 'C':
                n_points = max(int(exp), 6)
                n0 = m0 = np.geomspace(0.1, 2.0, n_points)
            else:
                n0 = m0 = 1 + np.arange(exp)

        if verbose:
            print('Running the following initial guesses for the exponents (n, m):')
            print(n0)
            print()

        if indi:
            return list(product([gc0_I_local], [gc0_III_local], n0, m0))
        return np.column_stack([
            np.full(len(n0), gc0_I_local),
            np.full(len(n0), gc0_III_local),
            n0,
            n0
        ])

    def run_regression(
            data, model, beta0,
            sstol=1e-12, partol=1e-12,
            maxit=1000, ndigit=12,
            ifixb=[1, 1, 0, 0],
            fit_type=1, deriv=3,
            init=0, iteration=0, final=0):
        odr = ODR(
            data,
            model,
            beta0=beta0,
            sstol=sstol,
            partol=partol,
            maxit=maxit,
            ndigit=ndigit,
            ifixb=ifixb,
        )
        odr.set_job(fit_type=fit_type, deriv=deriv)
        odr.set_iprint(init=init, iter=iteration, final=final)
        return odr.run()

    def calc_fit_statistics(final, ndof):
        fit = defaultdict()
        fit['params'] = final.beta
        fit['stddev'] = final.sd_beta
        fit['reduced_chi_squared'] = final.res_var
        fit['chi_squared'] = ndof * fit['reduced_chi_squared']
        fit['p_value'] = distributions.chi2.sf(fit['chi_squared'], ndof)
        fit['R_squared'] = 1 - fit['chi_squared'] / (ndof + fit['chi_squared'])
        fit['final'] = final
        return fit

    def results(fit):
        GIc, GIIIc, n, m = fit['params']
        chi2 = fit['reduced_chi_squared']
        pval = fit['p_value']
        r2 = fit['R_squared']
        header = 'Variable      Value    Description'.upper()
        rule = '---'.join(['-' * s for s in [8, 5, 50]])
        print(header)
        print(rule)
        print(f"GIc        {GIc:8.3f}    Mode I fracture toughness")
        print(f"GIIIc      {GIIIc:8.3f}    Mode III fracture toughness")
        print(f"n          {n:8.3f}    Interaction-law exponent")
        print(f"m          {m:8.3f}    Interaction-law exponent")
        if str(fit.get('var', var)).upper() == 'C':
            inv_n = np.inf if abs(float(n)) < 1e-15 else 1.0 / float(n)
            inv_m = np.inf if abs(float(m)) < 1e-15 else 1.0 / float(m)
            print(f"1/n        {inv_n:8.3f}    Effective exponent in criterion C")
            print(f"1/m        {inv_m:8.3f}    Effective exponent in criterion C")
        print(rule)
        print(f"chi2       {chi2:8.3f}    Reduced chi^2 per DOF (goodness of fit)")
        print(f"p-value    {pval:8.1e}    p-value (statistically significant if below 0.05)")
        print(f"R2         {r2:8.3f}    R-squared (not valid for nonlinear regression)")
        print()

    data, ndof = assemble_data(df_local, dim)
    model = Model(
        fcn=residual,
        fjacb=param_jacobian,
        fjacd=value_jacobian,
        implicit=True,
        extra_args=(var, True)
    )
    gc0_I_eff = float(gc0) if gc0_I is None else float(gc0_I)
    gc0_III_eff = float(gc0) if gc0_III is None else float(gc0_III)
    guess = get_initial_guesses(
        gc0_I_local=gc0_I_eff,
        gc0_III_local=gc0_III_eff,
        exp=exp,
        indi=indi,
        var=var,
        verbose=verbose
    )
    ifixb_eff = list(ifixb)
    if same_exponent and len(ifixb_eff) >= 4:
        # Tie n and m by fixing m and evaluating residual with m := n.
        ifixb_eff[3] = 0
        guess = [np.array([g[0], g[1], g[2], g[2]], dtype=float) for g in guess]
    runs_all = []
    for g in guess:
        try:
            runs_all.append(run_regression(
                data,
                model,
                g,
                ifixb=ifixb_eff,
                # Numerical derivatives are more robust for constrained/tied exponents.
                deriv=0 if same_exponent else 3,
            ))
        except Exception:
            continue
    runs = [r for r in runs_all if np.isfinite(getattr(r, 'sum_square', np.nan))]
    if not runs:
        raise RuntimeError("ODR did not converge for any initial guess.")
    final = runs[np.argmin([run.sum_square for run in runs])]
    fit = calc_fit_statistics(final, ndof)
    if same_exponent:
        fit['params'] = np.array(fit['params'], dtype=float)
        fit['params'][3] = fit['params'][2]
        fit['stddev'] = np.array(fit['stddev'], dtype=float)
        # m is tied to n; report same uncertainty.
        fit['stddev'][3] = fit['stddev'][2]
    fit['var'] = var
    if print_results:
        results(fit)
    return fit


def optimize_mode_I_III_interaction(
    df_or_path,
    auto_plateau_exponent=True,
    use_plateau_cap_for_geo=True,
    plateau_p_min=1,
    plateau_p_max=30,
    plateau_cap_step=0.1,
    plateau_rel_tol=0.002,
    plateau_abs_tol=0.1,
    plateau_consecutive=5,
    fit_resolution=1000,
    print_results=True,
):
    """Optimize Mode I/III interaction and return all values/stats as dict."""
    if isinstance(df_or_path, pd.DataFrame):
        df = df_or_path.copy()
    else:
        try:
            df = pd.read_parquet(df_or_path, engine='fastparquet')
            if print_results:
                print(f"Loaded data from Parquet: {df_or_path}")
        except Exception as e:
            if print_results:
                print(f"Error loading parquet: {e}")
            return None

    for col in ['G1c', 'G3c']:
        if col not in df.columns:
            if print_results:
                print(f"Error: Missing required columns: {col}")
            return None

    work = pd.DataFrame(index=df.index)
    work['G1c'] = pd.to_numeric(df['G1c'], errors='coerce')
    work['G3c'] = pd.to_numeric(df['G3c'], errors='coerce')
    work['G1c_unc'] = pd.to_numeric(df['G1c_uncertainty'], errors='coerce').fillna(0.0).clip(lower=0.0) if 'G1c_uncertainty' in df.columns else 0.0
    work['G3c_unc'] = pd.to_numeric(df['G3c_uncertainty'], errors='coerce').fillna(0.0).clip(lower=0.0) if 'G3c_uncertainty' in df.columns else 0.0
    if 'AFN_num' in df.columns:
        work['AFN_num'] = pd.to_numeric(df['AFN_num'], errors='coerce')
    elif 'AFN' in df.columns:
        work['AFN_num'] = pd.to_numeric(df['AFN'], errors='coerce')
    else:
        work['AFN_num'] = np.nan
    work = work.dropna(subset=['G1c', 'G3c'])
    if work.empty:
        if print_results:
            print("No valid rows for optimization.")
        return None

    eps = 1e-12
    gi = np.clip(work['G1c'].to_numpy(dtype=float), eps, None)
    giii = np.clip(work['G3c'].to_numpy(dtype=float), eps, None)
    sx = np.clip(work['G1c_unc'].to_numpy(dtype=float), 1e-9, None)
    sy = np.clip(work['G3c_unc'].to_numpy(dtype=float), 1e-9, None)
    afn = work['AFN_num'].to_numpy(dtype=float)

    afn_round = np.rint(afn)
    pst_mask = np.isfinite(afn_round) & np.isin(afn_round.astype(int), np.arange(1, 8))
    non_pst_mask = ~pst_mask
    gi_seed = gi[pst_mask] if np.any(pst_mask) else gi
    giii_seed = giii[non_pst_mask] if np.any(non_pst_mask) else giii

    def _mean_sem(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return np.nan, np.nan, 0
        mu = float(np.mean(vals))
        sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        return mu, sem, int(vals.size)

    gic_ref, gic_ref_unc, n_gic_ref = _mean_sem(gi_seed)
    giiic_ref, giiic_ref_unc, n_giiic_ref = _mean_sem(giii_seed)
    if not np.isfinite(gic_ref) or gic_ref <= 0:
        gic_ref = float(np.nanmedian(gi))
    if not np.isfinite(giiic_ref) or giiic_ref <= 0:
        giiic_ref = float(np.nanmedian(giii))
    g13 = gi + giii
    max_ratio_iii = float(np.nanmax(np.divide(giii, np.maximum(g13, eps)))) if g13.size else np.nan
    n_samples = max(int(fit_resolution), 50)

    def _quality(red):
        if not np.isfinite(red):
            return "unknown"
        if red < 0.5:
            return "possibly overestimated uncertainties"
        if red <= 2.0:
            return "good"
        if red <= 5.0:
            return "acceptable"
        return "poor"

    def _orth_stats(n_eval, m_eval, gic_eval, giiic_eval, curve_samples):
        n_eval = max(float(n_eval), 1e-9)
        m_eval = max(float(m_eval), 1e-9)
        gic_eval = max(float(gic_eval), eps)
        giiic_eval = max(float(giiic_eval), eps)
        implicit = (gi / gic_eval) ** n_eval + (giii / giiic_eval) ** m_eval - 1.0
        t = np.linspace(0.0, 1.0, max(int(curve_samples), 200))
        t_dense = np.unique(np.concatenate([t, t ** 2, 1.0 - (1.0 - t) ** 2]))
        x_curve = np.clip(gic_eval * t_dense, 0.0, gic_eval)
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            term = 1.0 - (x_curve / gic_eval) ** n_eval
            y_curve = np.full_like(x_curve, np.nan, dtype=float)
            msk = term >= 0.0
            y_curve[msk] = giiic_eval * np.power(term[msk], 1.0 / m_eval)
        mskc = np.isfinite(x_curve) & np.isfinite(y_curve)
        x_curve = x_curve[mskc]
        y_curve = y_curve[mskc]
        if x_curve.size < 2:
            x_curve = np.array([0.0, gic_eval], dtype=float)
            y_curve = np.array([giiic_eval, 0.0], dtype=float)

        dx = gi[:, None] - x_curve[None, :]
        dy = giii[:, None] - y_curve[None, :]
        d2 = dx ** 2 + dy ** 2
        idx = np.argmin(d2, axis=1)
        orth_abs = np.sqrt(np.maximum(d2[np.arange(gi.size), idx], 0.0))
        orth = np.sign(implicit) * orth_abs
        gi_p = np.clip(x_curve[idx], eps, None)
        giii_p = np.clip(y_curve[idx], eps, None)
        dfx = n_eval * np.power(gi_p / gic_eval, n_eval - 1.0) / gic_eval
        dfy = m_eval * np.power(giii_p / giiic_eval, m_eval - 1.0) / giiic_eval
        gnorm = np.sqrt(dfx ** 2 + dfy ** 2)
        gnorm = np.clip(np.nan_to_num(gnorm, nan=1e-12, posinf=1e12, neginf=1e-12), 1e-12, None)
        sorth = np.sqrt((dfx * sx) ** 2 + (dfy * sy) ** 2) / gnorm
        sorth = np.clip(np.nan_to_num(sorth, nan=1e-9, posinf=1e9, neginf=1e-9), 1e-9, None)
        chi2 = float(np.sum((orth / sorth) ** 2))
        dof = max(int(gi.size) - 2, 1)
        return {
            'chi2': chi2,
            'red_chi2': float(chi2 / dof),
            'rmse_orth': float(np.sqrt(np.mean(orth ** 2))),
            'x_curve': x_curve,
            'y_curve': y_curve,
        }

    forced_fits = []

    geo_fit = {'n': np.nan, 'm': np.nan, 'giiic': np.nan, 'chi2': np.nan, 'red_chi2': np.nan, 'rmse_orth': np.nan,
               'quality': 'unknown', 'method': 'not-run', 'starts_total': 0, 'starts_success': 0, 'x_curve': np.array([]), 'y_curve': np.array([])}
    plateau_fit = {'enabled': bool(auto_plateau_exponent), 'p_selected': np.nan, 'p_plateau_start': np.nan,
                   'giiic_selected': np.nan, 'chi2_selected': np.nan, 'red_chi2_selected': np.nan, 'rmse_selected': np.nan, 'rows': []}

    # Plateau scan with step control.
    if bool(auto_plateau_exponent):
        p_lo = max(float(plateau_p_min), 1.0)
        p_hi = max(float(plateau_p_max), p_lo)
        p_step = max(float(plateau_cap_step), 1e-6)
        p_vals = np.arange(p_lo, p_hi + 0.5 * p_step, p_step, dtype=float)
        giiic_upper = float(max(3.0 * giiic_ref, 1.5 * np.nanmax(giii), eps))
        try:
            from scipy.optimize import minimize_scalar
            has_scalar = True
        except Exception:
            has_scalar = False
        chi2_vals = []
        for p_scan in p_vals:
            def _obj_g3(g3):
                return float(_orth_stats(p_scan, p_scan, gic_ref, float(g3), max(350, int(n_samples // 2)))['chi2'])
            g3_opt = float(giiic_ref)
            if has_scalar:
                try:
                    r = minimize_scalar(_obj_g3, bounds=(eps, giiic_upper), method='bounded', options={'maxiter': 200})
                    if r.success and np.isfinite(r.x):
                        g3_opt = float(r.x)
                except Exception:
                    pass
            st = _orth_stats(p_scan, p_scan, gic_ref, g3_opt, max(n_samples, 1200))
            chi2_vals.append(float(st['chi2']))
            plateau_fit['rows'].append({'p': float(p_scan), 'giiic': float(g3_opt), 'chi2': float(st['chi2']),
                                        'red_chi2': float(st['red_chi2']), 'rmse': float(st['rmse_orth'])})
        if len(chi2_vals) >= 2:
            d_abs = [chi2_vals[i - 1] - chi2_vals[i] for i in range(1, len(chi2_vals))]
            d_rel = [(d_abs[i] / max(abs(chi2_vals[i]), 1e-12)) for i in range(len(d_abs))]
            flags = [(d_abs[i] <= float(plateau_abs_tol)) and (d_rel[i] <= float(plateau_rel_tol)) for i in range(len(d_abs))]
            run, idx = 0, None
            k = max(int(plateau_consecutive), 1)
            for i, f in enumerate(flags):
                run = run + 1 if f else 0
                if run >= k:
                    idx = i - k + 2
                    break
            if idx is not None:
                p_start = float(p_vals[idx])
                p_sel = float(max(p_start - p_step, p_lo))
            else:
                p_start = np.nan
                p_sel = float(p_vals[int(np.argmin(chi2_vals))])
            row_sel = next((r for r in plateau_fit['rows'] if np.isclose(float(r['p']), float(p_sel), atol=1e-12)), None)
            if row_sel is not None:
                plateau_fit.update({
                    'p_selected': float(p_sel),
                    'p_plateau_start': float(p_start) if np.isfinite(p_start) else np.nan,
                    'giiic_selected': float(row_sel['giiic']),
                    'chi2_selected': float(row_sel['chi2']),
                    'red_chi2_selected': float(row_sel['red_chi2']),
                    'rmse_selected': float(row_sel['rmse']),
                })

    # Free red-fit with n=m=p and free GIIIc.
    try:
        from scipy.optimize import minimize
        seed_axis = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 20.0, 35.0, 50.0], dtype=float)
        giii_scales = np.array([0.6, 0.8, 1.0, 1.2, 1.5], dtype=float)
        seeds = [(float(a), float(max(giiic_ref * s, eps))) for a in seed_axis for s in giii_scales]
        giiic_upper = float(max(3.0 * giiic_ref, 1.5 * np.nanmax(giii), eps))

        def _obj_geo(x):
            p_try, g3_try = float(x[0]), float(x[1])
            if p_try <= 0.05 or g3_try <= eps or (not np.isfinite(p_try + g3_try)):
                return 1e18
            return float(_orth_stats(p_try, p_try, gic_ref, g3_try, max(350, int(n_samples // 2)))['chi2'])

        runs = []
        for p0, g0 in seeds:
            geo_fit['starts_total'] += 1
            try:
                r = minimize(_obj_geo, x0=np.array([p0, g0]), method='L-BFGS-B', bounds=[(0.05, 50.0), (eps, giiic_upper)])
                if r.success and np.all(np.isfinite(r.x)) and np.isfinite(r.fun):
                    geo_fit['starts_success'] += 1
                    runs.append((float(r.fun), float(r.x[0]), float(r.x[1])))
            except Exception:
                continue
        if runs:
            runs.sort(key=lambda t: t[0])
            _, p_geo, g3_geo = runs[0]
            st = _orth_stats(p_geo, p_geo, gic_ref, g3_geo, max(n_samples, 1200))
            geo_fit.update({'n': p_geo, 'm': p_geo, 'giiic': g3_geo, 'chi2': st['chi2'], 'red_chi2': st['red_chi2'],
                            'rmse_orth': st['rmse_orth'], 'quality': _quality(st['red_chi2']),
                            'method': 'multistart nearest-arc orthogonal (n=m, free GIIIc; L-BFGS-B)'})

        if bool(use_plateau_cap_for_geo):
            p_lo = max(float(plateau_p_min), 1.0)
            p_hi = max(float(plateau_p_max), p_lo)
            p_step = max(float(plateau_cap_step), 1e-6)
            nm_caps = np.arange(p_lo, p_hi + 0.5 * p_step, p_step, dtype=float)
            best_overall, prev, run, used_cap = None, None, 0, np.nan
            for cap in nm_caps:
                seed_p = min(max(float(geo_fit['n']) if np.isfinite(geo_fit['n']) else 2.0, 0.05), cap)
                seed_g = min(max(float(geo_fit['giiic']) if np.isfinite(geo_fit['giiic']) else giiic_ref, eps), giiic_upper)
                try:
                    r = minimize(_obj_geo, x0=np.array([seed_p, seed_g]), method='L-BFGS-B',
                                 bounds=[(0.05, cap), (eps, giiic_upper)])
                    if not (r.success and np.all(np.isfinite(r.x)) and np.isfinite(r.fun)):
                        continue
                except Exception:
                    continue
                best_cap = (float(r.fun), float(r.x[0]), float(r.x[1]))
                best_overall, used_cap = best_cap, float(cap)
                if prev is not None:
                    d_abs = float(prev - best_cap[0])
                    d_rel = float(d_abs / max(abs(prev), 1e-12))
                    run = run + 1 if ((d_abs <= float(plateau_abs_tol)) and (d_rel <= float(plateau_rel_tol))) else 0
                    if run >= max(int(plateau_consecutive), 1):
                        break
                prev = float(best_cap[0])
            if best_overall is not None:
                _, p_geo, g3_geo = best_overall
                st = _orth_stats(p_geo, p_geo, gic_ref, g3_geo, max(n_samples, 1200))
                geo_fit.update({'n': p_geo, 'm': p_geo, 'giiic': g3_geo, 'chi2': st['chi2'], 'red_chi2': st['red_chi2'],
                                'rmse_orth': st['rmse_orth'], 'quality': _quality(st['red_chi2']),
                                'method': f"multistart nearest-arc orthogonal (n=m, free GIIIc; adaptive-plateau-stop at cap={used_cap:.2f})"})
    except Exception:
        pass

    if np.isfinite(geo_fit['n']) and geo_fit['n'] > 0 and np.isfinite(geo_fit['m']) and geo_fit['m'] > 0 and np.isfinite(geo_fit['giiic']) and geo_fit['giiic'] > 0:
        x_geo = np.linspace(eps, max(float(gic_ref), eps), n_samples)
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            term = 1.0 - (x_geo / max(float(gic_ref), eps)) ** float(geo_fit['n'])
            y_geo = np.full_like(x_geo, np.nan, dtype=float)
            vm = term >= 0.0
            y_geo[vm] = float(geo_fit['giiic']) * (term[vm] ** (1.0 / max(float(geo_fit['m']), eps)))
        vm = np.isfinite(x_geo) & np.isfinite(y_geo) & (y_geo >= 0.0)
        geo_fit['x_curve'] = x_geo[vm]
        geo_fit['y_curve'] = y_geo[vm]

    result = {
        'N': int(gi.size),
        'N_fit': int(gi.size),
        'N_PST': int(np.sum(pst_mask)),
        'N_nonPST': int(np.sum(non_pst_mask)),
        'max_ratio_iii': float(max_ratio_iii),
        'gic_ref': float(gic_ref),
        'gic_ref_unc': float(gic_ref_unc),
        'n_gic_ref': int(n_gic_ref),
        'giiic_ref': float(giiic_ref),
        'giiic_ref_unc': float(giiic_ref_unc),
        'n_giiic_ref': int(n_giiic_ref),
        'forced_fits': forced_fits,
        'geo_fit': geo_fit,
        'plateau_fit': plateau_fit,
        'settings': {
            'auto_plateau_exponent': bool(auto_plateau_exponent),
            'use_plateau_cap_for_geo': bool(use_plateau_cap_for_geo),
            'plateau_p_min': float(plateau_p_min),
            'plateau_p_max': float(plateau_p_max),
            'plateau_cap_step': float(plateau_cap_step),
            'plateau_rel_tol': float(plateau_rel_tol),
            'plateau_abs_tol': float(plateau_abs_tol),
            'plateau_consecutive': int(plateau_consecutive),
        },
    }

    if print_results:
        print("")
        print("=" * 72)
        print("optimize_mode_I_III_interaction - report")
        print("-" * 72)
        print(f"  data points        : N={result['N']}")
        print(f"  optimization points: N_fit={result['N_fit']} (PST included)")
        print("  AFN split          : PST=AFN 1..7 for GIc, non-PST for GIIIc")
        print(f"  subset counts      : N_PST={result['N_PST']}, N_nonPST={result['N_nonPST']}")
        print(f"  GIc reference      : {result['gic_ref']:.4f} +/- {result['gic_ref_unc']:.4f} J/m^2 (mean +/- SEM, n={result['n_gic_ref']})")
        print(f"  GIIIc reference    : {result['giiic_ref']:.4f} +/- {result['giiic_ref_unc']:.4f} J/m^2 (mean +/- SEM, n={result['n_giiic_ref']})")
        print(f"  max GIII/(GI+GIII) : {result['max_ratio_iii']:.4f}")
        print("-" * 72)
        gf = result['geo_fit']
        print("  geometric best fit (n=m, free GIIIc)")
        print(f"    method            : {gf['method']}")
        print(f"    starts total/succ : {gf['starts_total']} / {gf['starts_success']}")
        print(f"    n                 : {gf['n']:.4f}")
        print(f"    m                 : {gf['m']:.4f}")
        print(f"    GIIIc             : {gf['giiic']:.4f} J/m^2 (free)")
        print(f"    chi^2={gf['chi2']:.4g}, reduced chi^2={gf['red_chi2']:.4g}, RMSE_orth={gf['rmse_orth']:.4g}, quality={gf['quality']}")
        if result['plateau_fit']['enabled']:
            pf = result['plateau_fit']
            s = result['settings']
            print("-" * 72)
            print("  plateau scan (n=m, weighted nearest-arc chi^2)")
            print(f"    scan range         : p={s['plateau_p_min']:.2f}..{s['plateau_p_max']:.2f} (step={s['plateau_cap_step']:.2f})")
            print(f"    plateau criteria   : d_abs<={s['plateau_abs_tol']:.4g}, d_rel<={s['plateau_rel_tol']:.4g}, consecutive={s['plateau_consecutive']}")
            if np.isfinite(pf['p_plateau_start']):
                print(f"    plateau starts at  : p={pf['p_plateau_start']:.2f}")
            else:
                print("    plateau starts at  : not detected (used minimum chi^2)")
            print(f"    selected exponent  : p={pf['p_selected']:.2f}")
            print(f"    selected GIIIc     : {pf['giiic_selected']:.4f} J/m^2")
            print(f"    chi^2={pf['chi2_selected']:.4g}, reduced chi^2={pf['red_chi2_selected']:.4g}, RMSE_orth={pf['rmse_selected']:.4g}")
        print("=" * 72)
        print("")

    return result

