# Mode3DC

Pipeline for processing mode III displacement-controlled snow fracture experiments. From raw data to publication-ready mixed-mode (I + III) fracture toughness results, including analytical modeling, visualization, and digital image analysis of weak-layer out of plane anti-crack behavior.

## Setup

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Mode3DC
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment:**
   
   On macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```
   
   On Windows:
   ```bash
   .venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
Mode3DC/
├── data_config.py          # Data configuration and import functions
├── layout.py               # Color palette and layout configuration
├── load_signal_analyser.py # Interactive load signal analysis tool
├── M3DC_Notebook.ipynb     # Main analysis notebook
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Usage

### Running the Notebook

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open `M3DC_Notebook.ipynb` in your browser

### Using the Load Signal Analyser

The interactive load signal analyser can be run from the command line:

```bash
python load_signal_analyser.py <path_to_parquet_file>
```

For example:
```bash
python load_signal_analyser.py "/path/to/data/ON4PB_raw.parquet"
```

## Configuration

### Server Path Configuration

The project uses cross-platform server path management. Update the server paths in `data_config.py` if needed:

```python
SERVER_PATHS = {
    'macos': '/Volumes/Public/04 phds/Adam/02_experiments/02_NotchedBeam',
    'linux': '/mnt/smb_public/04 phds/Adam/02_experiments/02_NotchedBeam',
    'windows': 'Z:/04 phds/Adam/02_experiments/02_NotchedBeam'
}
```

## Dependencies

Key dependencies:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **openpyxl**: Excel file reading/writing
- **fastparquet**: Parquet file support
- **jupyter**: Notebook environment

See `requirements.txt` for the complete list with version specifications.

## Notes

- The project uses Parquet format for efficient data storage
- Color palette is defined in `layout.py` and should be used consistently across visualizations
- Data paths are configured for cross-platform server access

## Troubleshooting

### Import Errors

If you encounter import errors for `visualisation` module, this is expected if the module is not included in the repository. The `load_signal_analyser.py` will work without it, but some visualization styling may not be applied. The import is optional and wrapped in a try/except block.

### Parquet File Issues

If you encounter issues reading Parquet files, ensure `fastparquet` is installed:
```bash
pip install fastparquet
```

### Excel File Issues

If you encounter issues reading Excel files, ensure `openpyxl` is installed:
```bash
pip install openpyxl
```
