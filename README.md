## nt-mmap-converter

Convert neutrino telescope data (Prometheus, IceCube) into efficient memory-mapped files for ML training.

### Install

```bash
git clone https://github.com/yourname/nt-mmap-converter
cd nt-mmap-converter
```

### Convert

```bash
# Prometheus parquet files
python converter.py --source prometheus --input /path/to/parquet/files/ --output mydata

# IceCube i3 files (optional filter selection shown)
python converter.py --source icecube --input /path/to/i3/files/ --output mydata
python converter.py --source icecube --input /path/to/i3/files/ --output mydata \
    --filters CascadeFilter_13 MuonFilter_13
python converter.py --source icecube --input /path/to/i3/files/ --output mydata \
    --subevent-streams InIceSplit

# With hit grouping (time window in nanoseconds). highly recommend for prometheus datasets
python converter.py --source prometheus --input /path/to/parquet/files/ --output mydata --grouping-window-ns 2.0
```

### Load

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

# Auto-detects source (Prometheus/IceCube) from file header
events, photons_array, photon_dtype = load_ntmmap("mydata")

# Access any event instantly
event = events[42]
energy = event['initial_energy']  # GeV
hits = event['num_hits']          # Total photon hits
channels = event['num_chans']     # Unique sensors hit

# Load photons with O(1) indexing
photons = photons_array[event['photon_start_idx']:event['photon_end_idx']]
positions = np.column_stack([photons['x'], photons['y'], photons['z']])
```

### Filter & Split

```python
# Instant filtering on any field
energy_mask = (events['initial_energy'] >= 1e3) & (events['initial_energy'] <= 1e6)
high_activity = events['num_chans'] >= 10
filtered = events[energy_mask & high_activity]

# Morphology shorthand for IceCube datasets
is_track_like = np.isin(events['morphology'], [1, 2, 3, 4])
track_subset = events[is_track_like]

# Deterministic train/validation split in O(1)
np.random.seed(42)
n = len(events)
indices = np.arange(n)
np.random.shuffle(indices)

train_events = events[indices[:int(0.8*n)]]
val_events = events[indices[int(0.8*n):]]
```

### Fields

**Prometheus Events:**
- `photon_start_idx`, `photon_end_idx` - Photon array indices
- `num_hits`, `num_chans` - Hit count, unique channels hit
- `initial_energy`, `initial_zenith`, `initial_azimuth` - Primary particle
- `initial_x`, `initial_y`, `initial_z` - Interaction vertex  
- `bjorken_x`, `bjorken_y`, `column_depth` - Physics variables
- `interaction` - Generator-specific interaction code
- `final_energy[5]`, `final_type[5]`, `final_x[5]`, `final_y[5]`, `final_z[5]` - Final state particles

**IceCube Events (+ above):**
- `homogenized_qtot` - Total charge (when available)
- `event_class` - Detailed classification
  - 0 outside cascade
  - 1 contained cascade
  - 2 through-going track
  - 3 starting track
  - 4 stopping track
  - 5 double bang
  - 6 stopping tau
  - 7 Glashow cascade
  - 8 Glashow track
  - 9 Glashow tau
  - 10 stop-start track (reserved)
  - 11 passing track
  - 12 uncontained tau
  - 21 multiple coincident events
  - 22 through-going bundle
  - 23 stopping bundle
  - 100/101 unknown
- `morphology` - Simplified label (0=cascade, 1=through-going track, 2=starting track, 3=stopping track, 4=passing track, 5=bundle/other)
- `final_*[0]` - Final state lepton, `final_*[1]` - Hadron shower

**Photon Hits:**
- `x`, `y`, `z` - Sensor position (meters)  
- `t` - Hit time (nanoseconds)
- `charge` - Photoelectrons
- `string_id`, `sensor_id` - Detector identifiers

### Format

```
data.idx: [dtype_size][event_dtype][event_records...]
data.dat: [dtype_size][photon_dtype][photon_data...]
data.config.json: run metadata (CLI invocation, arguments, summary statistics)
```
