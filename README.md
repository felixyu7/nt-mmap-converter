## nt-mmap-converter

Convert neutrino telescope event data (Prometheus, IceCube) into compact, memory-mapped files optimized for ML training and large-scale analysis.

### Why

Raw neutrino telescope files (parquet, i3) are slow to load and scan repeatedly. This tool flattens events into two fixed-layout binary files you can `mmap` and index in O(1):

- **`.idx`** — one fixed-size event record per entry (MC truth, hit counts, photon-slice range)
- **`.dat`** — all photon hits packed contiguously, referenced by `[photon_start_idx : photon_end_idx)`
- **`.config.json`** — sidecar with the CLI invocation, flags, and summary stats for reproducibility

Filters (energy cut, morphology class, etc.) become boolean numpy masks on the structured index array, and train/val splits are just shuffled index slices. No pre-materialization, no per-epoch parsing, no RAM ceiling — the entire dataset is lazily paged from disk.

### Install

```bash
git clone https://github.com/felixyu7/nt-mmap-converter
cd nt-mmap-converter
```

There is no `setup.py` / `pyproject.toml` — run `converter.py` directly from the repo root, or add the repo to your `PYTHONPATH` when importing `core.mmap_format` from your own code.

**Dependencies**

- **Prometheus conversion:** `numpy`, `pandas`, `pyarrow`, `scipy`
- **IceCube conversion:** `icetray >= 1.17` (with `ml_suite` built — provides `I3EventLabeler`) in addition to the above

Prometheus events store a `vertex_x/y/z` following the same convention as the IceCube path: the neutrino interaction vertex when it lies inside the detector volume, otherwise the outgoing charged lepton's detector entry point. The detector volume is the convex hull of the DOM positions in `resources/icecube.geo` (used by default; override with the `geo_path` argument to `convert_prometheus_to_mmap`).

The IceCube path is optional. `converter.py` auto-detects whether `icecube` imports and only exposes `--source icecube` if it does. For IceCube runs, activate your IceTray environment before invoking the converter (e.g. `/path/to/icetray/build/env-shell.sh`).

### Convert

Basic usage:

```bash
# Prometheus parquet
python converter.py --source prometheus \
    --input /path/to/parquet_dir \
    --output mydata

# IceCube i3 / i3.zst
python converter.py --source icecube \
    --input /path/to/i3_dir \
    --output mydata
```

Output paths are a *base path without extension* — the converter writes `mydata.idx`, `mydata.dat`, and `mydata.config.json`.

Multiple input directories are merged into a single output dataset:

```bash
python converter.py --source prometheus \
    --input /data/numu /data/nue /data/nutau \
    --output combined
```

#### All CLI flags

| Flag | Applies to | Description |
|-|-|-|
| `--source {prometheus,icecube}` | both | Input format. `icecube` only available if IceTray is importable. |
| `--input DIR [DIR ...]` | both | One or more input directories. |
| `--output PATH` | both | Output base path (no extension). |
| `--file-range START-END` | both | Slice of discovered files to process, e.g. `0-100`. Useful for sharding. |
| `--info` | both | Print dataset statistics after conversion. |
| `--grouping-window-ns FLOAT` | prometheus | Per-sensor time-window grouping in ns (0 = raw hits). Recommended for Prometheus. |
| `--min-photons INT` | prometheus | Drop events with fewer raw hits (checked *before* grouping — fast skip). |
| `--max-photons INT` | prometheus | Drop events with more raw hits. |
| `--min-channels INT` | prometheus | Drop events with fewer unique sensors hit. |
| `--max-channels INT` | prometheus | Drop events with more unique sensors hit. |
| `--pulse-key NAME` | icecube | Pulse series to read from I3Frames (default `SplitInIceDSTPulses`). |
| `--filters NAME [NAME ...]` | icecube | Keep only frames whose `FilterMask` has any of these entries with `condition_passed`. |
| `--subevent-streams NAME [NAME ...]` | icecube | Restrict to specific sub-event streams (default `InIceSplit`). |

#### Prometheus examples

```bash
# With per-sensor grouping (strongly recommended for Prometheus)
python converter.py --source prometheus \
    --input /data/parquet --output mydata \
    --grouping-window-ns 2.0

# With pre-grouping event filters
python converter.py --source prometheus \
    --input /data/parquet --output mydata \
    --min-photons 50 --max-photons 20000 --min-channels 3

# File shard + dataset summary
python converter.py --source prometheus \
    --input /data/parquet --output mydata_shard0 \
    --file-range 0-100 --info
```

#### IceCube examples

```bash
# Baseline conversion (InIceSplit stream, default pulse key)
python converter.py --source icecube \
    --input /data/i3 --output mydata

# Keep only events passing Level2 cascade or muon filters
python converter.py --source icecube \
    --input /data/i3 --output mydata \
    --filters CascadeFilter_13 MuonFilter_13

# Use a different pulse series
python converter.py --source icecube \
    --input /data/i3 --output mydata \
    --pulse-key SRTInIcePulses
```

IceCube conversion uses the bundled GCD file at `resources/GeoCalibDetectorStatus_IC86.AVG_Pass2_SF0.99.i3` and runs `I3EventLabeler` (from `ml_suite`) on every frame that has an `I3MCTree`. Frames without MC truth (e.g. some CORSIKA Level2 files) still convert, with `event_class` and `morphology` set to `0`.

### Load

The repo ships with a loader helper that handles header parsing and auto-detects the source type:

```python
import numpy as np
from core.mmap_format import load_ntmmap

events, photons, photon_dtype = load_ntmmap("mydata")
# events   — structured np.memmap over mydata.idx  (zero-copy)
# photons  — structured np.memmap over mydata.dat  (zero-copy)

print(f"{len(events):,} events, fields: {events.dtype.names}")

# Pull an event and its photon hits
event = events[42]
hits = photons[event['photon_start_idx']:event['photon_end_idx']]
xyz = np.column_stack([hits['x'], hits['y'], hits['z']])
```

If you want a standalone loader with no dependency on this repo (e.g. to drop into a training script elsewhere), this is equivalent:

```python
import pickle, struct
import numpy as np

def load_ntmmap(path):
    with open(f"{path}.idx", 'rb') as f:
        dtype_size = struct.unpack('<I', f.read(4))[0]
        event_dtype = pickle.loads(f.read(dtype_size))
        events = np.memmap(f"{path}.idx", dtype=event_dtype, mode='r', offset=f.tell())

    with open(f"{path}.dat", 'rb') as f:
        dtype_size = struct.unpack('<I', f.read(4))[0]
        photon_dtype = pickle.loads(f.read(dtype_size))
        photons = np.memmap(f"{path}.dat", dtype=photon_dtype, mode='r', offset=f.tell())

    return events, photons, photon_dtype
```

The structured dtype is pickled into the file header, so the file is self-describing — you don't need to know the schema in advance. For a quick overview of any dataset:

```python
from core.utils import print_dataset_info
print_dataset_info("mydata")
```

### Filter, split, iterate

Everything below is O(1) or a single vectorized numpy pass:

```python
# Energy cut on structured fields
mask = (events['initial_energy'] >= 1e3) & (events['initial_energy'] <= 1e6)
hi_e = events[mask]

# IceCube morphology selection — tracks only (starting / throughgoing / stopping)
is_track = np.isin(events['morphology'], [1, 2, 3])
tracks = events[is_track]

# Deterministic train / val split
rng = np.random.default_rng(42)
idx = rng.permutation(len(events))
split = int(0.8 * len(events))
train, val = events[idx[:split]], events[idx[split:]]

# Iterate one event at a time (e.g. inside a torch Dataset)
for ev in events:
    hits = photons[ev['photon_start_idx']:ev['photon_end_idx']]
    ...
```

Because `events` and `photons` are memory maps, masking and indexing return views whenever possible — nothing is copied until you explicitly `np.array(...)` or arithmetic materializes a result.

### Fields

#### Photon hits (both sources)

| Field | Dtype | Units / Meaning |
|-|-|-|
| `x`, `y`, `z` | float32 | Sensor position (m, detector frame) |
| `t` | float32 | Hit time (ns) |
| `charge` | float32 | Photoelectrons. Prometheus raw hits get `charge=1`; grouped hits get the per-window summed PE. |
| `string_id`, `sensor_id` | uint16 | Detector identifiers |

#### Event record — common fields

| Field | Dtype | Meaning |
|-|-|-|
| `photon_start_idx`, `photon_end_idx` | uint64 | Photon array slice `[start:end)` |
| `num_hits` | uint32 | Total photons in the event (after grouping for Prometheus) |
| `num_chans` | uint32 | Unique `(string_id, sensor_id)` pairs hit |
| `initial_energy` | float32 | Primary energy (GeV) |
| `initial_zenith`, `initial_azimuth` | float32 | Primary direction (radians) |
| `initial_x`, `initial_y`, `initial_z` | float32 | Primary position (m) |
| `initial_type` | int32 | PDG code of the primary |

#### Event record — Prometheus only

| Field | Dtype | Meaning |
|-|-|-|
| `bjorken_x`, `bjorken_y` | float32 | DIS kinematic variables |
| `column_depth` | float32 | Column depth at interaction vertex |
| `interaction` | int32 | Generator-specific interaction code |
| `final_energy[5]`, `final_type[5]` | float32, int32 | Final-state particles, up to 5, zero-padded |
| `final_zenith[5]`, `final_azimuth[5]` | float32 | Final-state directions |
| `final_x[5]`, `final_y[5]`, `final_z[5]` | float32 | Final-state positions |

#### Event record — IceCube only

| Field | Dtype | Meaning |
|-|-|-|
| `homogenized_qtot` | float32 | `Homogenized_QTot` charge (when present in frame) |
| `event_class` | int16 | `I3EventLabeler` 34-class label (see below) |
| `morphology` | int8 | Simplified 6-class label (see below) |
| `vertex_x`, `vertex_y`, `vertex_z` | float32 | Neutrino interaction vertex for `event_class` 8–25; detector entry point otherwise |
| `final_energy[2]`, `final_type[2]` | float32, int32 | `[0]` = final lepton, `[1]` = hadron shower |
| `final_zenith[2]`, `final_azimuth[2]` | float32 | |
| `final_x[2]`, `final_y[2]`, `final_z[2]` | float32 | |

#### IceCube `morphology` (6 classes)

| Value | Name | Description |
|-|-|-|
| 0 | CASCADE | Shower-like (NC, NuE CC, tau → e/hadrons) |
| 1 | STARTING_TRACK | Track born inside detector, exits |
| 2 | THROUGHGOING_TRACK | Track enters and exits |
| 3 | STOPPING_TRACK | Track enters and stops/decays inside |
| 4 | UNCONTAINED | No detector primary (misses/skims detector) |
| 5 | BUNDLE | Multiple detector primaries (muon bundles) |

#### IceCube `event_class` (34 classes, matches `I3EventLabeler`)

| Range | Group |
|-|-|
| 0–5 | Uncontained (background, cascade, skimming track/bundle/spur, other) |
| 6–7 | Multiple primaries (bundle, other mixed) |
| 8 | NC hadronic cascade |
| 9 | NuE CC (EM + hadronic cascade) |
| 10–18 | Glashow resonance variants |
| 19–20 | NuMu CC (starting / contained track) |
| 21–25 | NuTau CC (inverted lollipop, double bang, tau→muon) |
| 26–27 | Entering muon (throughgoing / stopping) |
| 28–32 | Entering tau (throughgoing spur, lollipop, tau→muon) |
| 33 | Other |

See the `EventClass` enum in `data/icecube.py` for the full 34-class breakdown and the `DETAILED_TO_MORPHOLOGY` table that maps each class to a morphology.

### File format

```
mydata.idx
  [uint32 header_size][pickled event_dtype][event records ...]

mydata.dat
  [uint32 header_size][pickled photon_dtype][photon records ...]

mydata.config.json
  { timestamp_utc, command, args,
    summary: { events_converted, events_skipped,
               photons_written, elapsed_seconds } }
```

Both binary files are written sequentially during conversion. The `.idx` file starts with an estimated capacity and grows in place via `StreamingIndexWriter`, then is truncated to the exact final size on completion — so conversion runs in constant memory regardless of dataset size.

### Project layout

```
converter.py           # CLI entrypoint
core/
  mmap_format.py       # Event / photon dtypes, streaming writer, load_ntmmap
  utils.py             # Dataset stats / print_dataset_info
data/
  prometheus.py        # Parquet reader + per-sensor hit grouping
  icecube.py           # i3 reader + I3EventLabeler integration
resources/
  GeoCalibDetectorStatus_IC86.AVG_Pass2_SF0.99.i3   # Bundled GCD for IceCube
```
