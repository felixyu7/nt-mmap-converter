## nt-mmap-converter

Convert neutrino telescope data (Prometheus, IceCube) into efficient memory-mapped files for ML training.

### Install

```bash
git clone https://github.com/yourname/nt-mmap-converter
cd nt-mmap-converter
```

### Usage

Convert parquet files to memory-mapped format:

```bash
# Convert 1 file
python converter.py --source prometheus --input /path/to/parquet/files/ --output mydata --max-files 1

# Convert 10 files
python converter.py --source prometheus --input /path/to/parquet/files/ --output mydata --max-files 10

# Convert all files
python converter.py --source prometheus --input /path/to/parquet/files/ --output mydata
```

### Load data

```python
import numpy as np

def load_ntmmap(path):
    import pickle, struct
    
    # Load index with auto-detected dtype
    with open(f"{path}.idx", 'rb') as f:
        size = struct.unpack('<I', f.read(4))[0]
        dtype = pickle.loads(f.read(size))
        events = np.memmap(f"{path}.idx", dtype=dtype, mode='r', offset=f.tell())
    
    # Load data with auto-detected dtype  
    with open(f"{path}.dat", 'rb') as f:
        size = struct.unpack('<I', f.read(4))[0]
        photon_dtype = pickle.loads(f.read(size))
        photons_array = np.memmap(f"{path}.dat", dtype=photon_dtype, mode='r', offset=f.tell())
    
    return events, photons_array, photon_dtype

# Usage
events, photons_array, photon_dtype = load_ntmmap("mydata")

# Access any event instantly
event = events[42]
energy = event['initial_energy']  # GeV
position = (event['initial_x'], event['initial_y'], event['initial_z'])

# Load photons with simple array slicing
start = event['photon_start_idx']
end = event['photon_end_idx']
photons = photons_array[start:end]

# ML-ready arrays
positions = np.column_stack([photons['x'], photons['y'], photons['z']])
times = photons['t']
```

### File Format

```
data.idx: [dtype_size][event_dtype][event_records...]
data.dat: [dtype_size][photon_dtype][photon_data...]
```

Self-describing files with embedded dtypes. No manual schema definitions needed.
