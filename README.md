# Mode III weak layer fracture tests

Pipeline for processing Mode III displacement-controlled snow fracture experiments. The repository takes raw experimental data to publication-ready mixed-mode (I + III) fracture results, including preprocessing, interactive load-signal annotation, analytical modelling, and standardized visualizations.

## Entry points

- **Main workflow**: `M3DC_Notebook.ipynb`
- **Reusable code**: `data_config.py`, `visualisation.py`, `layout.py`
- **Interactive tool** (manual annotation of load–displacement curves): `load_signal_analyser.py`

## Setup

### Prerequisites

- Python 3.10+ (recommended; tested with modern `pandas/numpy/matplotlib`)
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

## Data: what you need and where it goes

The notebook expects a project data root that contains the experiment data folders (raw CSVs, Excel metadata, videos, and derived Parquet files). Paths are handled via `data_config.py`:

- **Server/shared-drive mode**: SMB mount paths can be configured in `data_config.py`
- **Local mode**: you can point the data root to a local folder with the same structure

Minimum required file for running most plots/analysis sections:

- `M3DC_raw.parquet` (typically located in `01_raw_data/04_metadata/`)

If you want to rebuild that Parquet file from raw inputs, you also need the corresponding Excel metadata files and load-signal CSV files as described in the notebook’s **Data** section.

## Project Structure

```
Mode3DC/
├── data_config.py          # Data configuration and import functions
├── layout.py               # Color palette and layout configuration
├── load_signal_analyser.py # Interactive load signal analysis tool
├── visualisation.py        # Plotting and figure-generation functions
├── M3DC_Notebook.ipynb     # Main analysis notebook
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Usage

### Running the Notebook

1. Start Jupyter (or JupyterLab):
   ```bash
   jupyter notebook
   ```

2. Open `M3DC_Notebook.ipynb` in your browser

3. Run cells from top to bottom. The notebook prints the resolved data paths early on; verify they point to your intended data location.

### Using the Load Signal Analyser

The interactive load signal analyser can be run from the command line:

```bash
python load_signal_analyser.py <path_to_parquet_file>
```

For example:
```bash
python load_signal_analyser.py "/path/to/data/M3DC_raw.parquet"
```

## Configuration

### Server Path Configuration

The project uses cross-platform server path management. Update the server paths in `data_config.py` if needed:

```python
SERVER_PATHS = {
    'macos': '/Volumes/<...>',
    'linux': '/mnt/<...>',
    'windows': 'Z:/<...>'
}
```

## Dependencies

Key dependencies:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **openpyxl**: Excel file reading/writing
- **fastparquet**: Parquet file support
- **scipy**: numerical optimization utilities used in fitting/processing
- **jupyter**: Notebook environment
- **weac**: weak-layer anticrack nucleation model (installed from GitHub as specified in `requirements.txt`)

See `requirements.txt` for the complete list with version specifications.

## Notes

- The project uses Parquet format for efficient data storage
- Color palette is defined in `layout.py` and should be used consistently across visualizations
- Data paths are configured for cross-platform server access

## Reproducibility tips (publication)

- Prefer running the notebook with **static figures** (`%matplotlib inline`) when preparing the final, shareable version.
- If you are producing a “clean” notebook for GitHub/publishing, consider clearing very large embedded outputs and regenerating figures as SVG or saving to disk via the notebook.

## Troubleshooting

### Matplotlib font / cache warnings

- If you don’t have **Minion Pro**, the code will fall back to an available serif font.
- If Matplotlib warns that its default cache directory is not writable, set `MPLCONFIGDIR` to a writable directory (or run in a local environment where your home directory is writable).

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

### SciPy missing

If you see errors about `scipy`, install/update dependencies:
```bash
pip install -r requirements.txt
```
