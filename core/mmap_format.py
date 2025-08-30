"""
Memory-mapped file format for neutrino telescope data.
"""

import numpy as np
import os
from typing import Tuple, Dict, Any

# Define the filter result structure for IceCube
FILTER_RESULT_DTYPE = np.dtype([
    ('name', 'U32'),      # Filter name (32-char max)
    ('passed', np.bool_)   # True/False result
])

# Define Prometheus event record structure
PROMETHEUS_EVENT_RECORD_DTYPE = np.dtype([
    # Indexing (photon indices, not byte offsets) - uint64 to prevent overflow
    ('photon_start_idx', np.uint64),
    ('photon_end_idx', np.uint64),
    
    # Hit statistics
    ('num_hits', np.uint32),    # Total photon hits (photon_end_idx - photon_start_idx)
    ('num_chans', np.uint32),   # Number of unique sensors hit
    
    # MC Truth scalars
    ('initial_energy', np.float32),
    ('initial_zenith', np.float32),
    ('initial_azimuth', np.float32),
    ('initial_x', np.float32),
    ('initial_y', np.float32),
    ('initial_z', np.float32),
    ('bjorken_x', np.float32),
    ('bjorken_y', np.float32),
    ('column_depth', np.float32),
    ('interaction', np.int32),
    ('initial_type', np.int32),
    
    # Final state arrays (5 particles, zero-padded)
    ('final_energy', np.float32, (5,)),
    ('final_type', np.int32, (5,)),
    ('final_zenith', np.float32, (5,)),
    ('final_azimuth', np.float32, (5,)),
    ('final_x', np.float32, (5,)),
    ('final_y', np.float32, (5,)),
    ('final_z', np.float32, (5,)),
    ('final_parent', np.int32, (5,)),
])

# Define IceCube event record structure
ICECUBE_EVENT_RECORD_DTYPE = np.dtype([
    # Indexing (photon indices, not byte offsets) - uint64 to prevent overflow
    ('photon_start_idx', np.uint64),
    ('photon_end_idx', np.uint64),
    
    # Hit statistics
    ('num_hits', np.uint32),    # Total photon hits (photon_end_idx - photon_start_idx)
    ('num_chans', np.uint32),   # Number of unique sensors hit
    
    # IceCube-specific fields
    ('homogenized_qtot', np.float32),           # Homogenized total charge
    ('num_filters', np.uint8),                  # Number of active filters
    ('filter_results', FILTER_RESULT_DTYPE, 50), # Array of 50 filter results
    
    # MC Truth scalars
    ('initial_energy', np.float32),
    ('initial_zenith', np.float32),
    ('initial_azimuth', np.float32),
    ('initial_x', np.float32),
    ('initial_y', np.float32),
    ('initial_z', np.float32),
    ('bjorken_x', np.float32),
    ('bjorken_y', np.float32),
    ('column_depth', np.float32),
    ('interaction', np.int32),
    ('initial_type', np.int32),
    
    # Final state arrays (5 particles, zero-padded)
    # Index 0: lepton, Index 1: hadrons, rest zero-padded
    ('final_energy', np.float32, (5,)),
    ('final_type', np.int32, (5,)),
    ('final_zenith', np.float32, (5,)),
    ('final_azimuth', np.float32, (5,)),
    ('final_x', np.float32, (5,)),
    ('final_y', np.float32, (5,)),
    ('final_z', np.float32, (5,)),
    ('final_parent', np.int32, (5,)),
])

# Keep EVENT_RECORD_DTYPE as alias for backward compatibility
EVENT_RECORD_DTYPE = PROMETHEUS_EVENT_RECORD_DTYPE

# Define the photon hit structure  
PHOTON_HIT_DTYPE = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('t', np.float32),
    ('charge', np.float32),
    ('string_id', np.uint32),
    ('sensor_id', np.uint32),
    ('id_idx', np.uint64),
])

# Size constants
PROMETHEUS_EVENT_RECORD_SIZE = PROMETHEUS_EVENT_RECORD_DTYPE.itemsize
ICECUBE_EVENT_RECORD_SIZE = ICECUBE_EVENT_RECORD_DTYPE.itemsize
EVENT_RECORD_SIZE = PROMETHEUS_EVENT_RECORD_SIZE  # Backward compatibility
PHOTON_HIT_SIZE = PHOTON_HIT_DTYPE.itemsize


class EventRecord:
    """Helper class for creating EventRecord arrays."""
    
    @staticmethod
    def create_array(num_events: int, source_type: str = 'prometheus') -> np.ndarray:
        """Create a zeroed array of EventRecords."""
        if source_type.lower() == 'icecube':
            return np.zeros(num_events, dtype=ICECUBE_EVENT_RECORD_DTYPE)
        else:
            return np.zeros(num_events, dtype=PROMETHEUS_EVENT_RECORD_DTYPE)
    
    @staticmethod
    def from_dict(data: Dict[str, Any], source_type: str = 'prometheus') -> np.ndarray:
        """Create an EventRecord from a dictionary."""
        if source_type.lower() == 'icecube':
            record = np.zeros(1, dtype=ICECUBE_EVENT_RECORD_DTYPE)[0]
        else:
            record = np.zeros(1, dtype=PROMETHEUS_EVENT_RECORD_DTYPE)[0]
        
        # Fill scalar fields
        scalar_fields = ['initial_energy', 'initial_zenith', 'initial_azimuth',
                        'initial_x', 'initial_y', 'initial_z', 'bjorken_x',
                        'bjorken_y', 'column_depth', 'interaction', 'initial_type']
        
        for field in scalar_fields:
            if field in data:
                record[field] = data[field]
        
        # Fill hit statistics
        if 'num_hits' in data:
            record['num_hits'] = data['num_hits']
        if 'num_chans' in data:
            record['num_chans'] = data['num_chans']
            
        # Fill IceCube-specific fields
        if source_type.lower() == 'icecube':
            if 'homogenized_qtot' in data:
                record['homogenized_qtot'] = data['homogenized_qtot']
                
            # Handle FilterMask
            if 'filter_mask' in data and isinstance(data['filter_mask'], dict):
                filter_items = list(data['filter_mask'].items())
                num_filters = min(len(filter_items), 50)  # Max 50 filters
                record['num_filters'] = num_filters
                
                for i, (filter_name, passed) in enumerate(filter_items[:50]):
                    record['filter_results'][i]['name'] = filter_name[:32]  # Truncate to 32 chars
                    record['filter_results'][i]['passed'] = bool(passed)
        
        # Fill array fields (with zero-padding)
        array_fields = ['final_energy', 'final_type', 'final_zenith', 'final_azimuth',
                       'final_x', 'final_y', 'final_z', 'final_parent']
        
        for field in array_fields:
            if field in data:
                arr = np.array(data[field])
                # Pad or truncate to 5 elements
                if len(arr) < 5:
                    padded = np.zeros(5, dtype=arr.dtype)
                    padded[:len(arr)] = arr
                    record[field] = padded
                else:
                    record[field] = arr[:5]
        
        return record


class PhotonHit:
    """Helper class for creating PhotonHit arrays."""
    
    @staticmethod
    def from_dict(data: Dict[str, np.ndarray]) -> np.ndarray:
        """Create PhotonHit array from dictionary of arrays."""
        n_photons = len(data['sensor_pos_x'])
        photons = np.zeros(n_photons, dtype=PHOTON_HIT_DTYPE)
        
        # Map fields
        photons['x'] = data['sensor_pos_x'].astype(np.float32)
        photons['y'] = data['sensor_pos_y'].astype(np.float32) 
        photons['z'] = data['sensor_pos_z'].astype(np.float32)
        photons['t'] = data['t'].astype(np.float32)
        
        # Handle charge field - use provided charge or default to 1.0
        if 'charge' in data:
            photons['charge'] = data['charge'].astype(np.float32)
        else:
            photons['charge'] = np.ones(n_photons, dtype=np.float32)
            
        photons['string_id'] = data['string_id'].astype(np.uint32)
        photons['sensor_id'] = data['sensor_id'].astype(np.uint32)
        photons['id_idx'] = data['id_idx'].astype(np.uint64)
        
        return photons


def create_mmap_files_with_headers(output_path: str, num_events: int) -> Tuple[str, str]:
    """
    Create memory-mapped files with dtype headers for writing (fixed allocation).
    
    Use this function when the exact number of events is known upfront (e.g., filtering).
    For data conversion, prefer create_streaming_mmap_files() for better performance.
    
    Args:
        output_path: Base path for output files (without extension)
        num_events: Number of events to allocate space for
        
    Returns:
        Tuple of (idx_path, dat_path) for further writing
    """
    import pickle
    import struct
    
    idx_path = f"{output_path}.idx"
    dat_path = f"{output_path}.dat"
    
    # Create index file with header
    with open(idx_path, 'wb') as f:
        # Write event dtype header
        dtype_bytes = pickle.dumps(EVENT_RECORD_DTYPE)
        f.write(struct.pack('<I', len(dtype_bytes)))  # Size of dtype
        f.write(dtype_bytes)                         # Dtype definition
        
        # Write placeholder data (will be filled by memory mapping)
        placeholder = np.zeros(num_events, dtype=EVENT_RECORD_DTYPE)
        f.write(placeholder.tobytes())
    
    # Create data file with header
    with open(dat_path, 'wb') as f:
        # Write photon dtype header  
        dtype_bytes = pickle.dumps(PHOTON_HIT_DTYPE)
        f.write(struct.pack('<I', len(dtype_bytes)))
        f.write(dtype_bytes)
        # Data will be appended later
    
    return idx_path, dat_path


def create_streaming_mmap_files(output_path: str, initial_events_estimate: int = 10000, source_type: str = 'prometheus') -> Tuple[str, str]:
    """
    Create memory-mapped files for streaming/dynamic allocation.
    
    Args:
        output_path: Base path for output files (without extension)
        initial_events_estimate: Initial size estimate for the index file
        source_type: The source of the data ('prometheus' or 'icecube')
        
    Returns:
        Tuple of (idx_path, dat_path) for streaming writes
    """
    import pickle
    import struct
    
    idx_path = f"{output_path}.idx"
    dat_path = f"{output_path}.dat"

    if source_type.lower() == 'icecube':
        event_dtype = ICECUBE_EVENT_RECORD_DTYPE
    else:
        event_dtype = PROMETHEUS_EVENT_RECORD_DTYPE
    
    # Create index file with header and initial allocation
    with open(idx_path, 'wb') as f:
        # Write event dtype header
        dtype_bytes = pickle.dumps(event_dtype)
        f.write(struct.pack('<I', len(dtype_bytes)))  # Size of dtype
        f.write(dtype_bytes)                         # Dtype definition
        
        # Write initial placeholder data
        placeholder = np.zeros(initial_events_estimate, dtype=event_dtype)
        f.write(placeholder.tobytes())
    
    # Create data file with header only
    with open(dat_path, 'wb') as f:
        # Write photon dtype header  
        dtype_bytes = pickle.dumps(PHOTON_HIT_DTYPE)
        f.write(struct.pack('<I', len(dtype_bytes)))
        f.write(dtype_bytes)
    
    return idx_path, dat_path


def create_index_mmap_with_header(idx_path: str, num_events: int) -> np.memmap:
    """Create memory-mapped index file, skipping the dtype header."""
    import pickle
    import struct
    
    # Calculate header size
    with open(idx_path, 'rb') as f:
        dtype_size = struct.unpack('<I', f.read(4))[0]
        header_size = 4 + dtype_size
    
    # Create memory map starting after header
    return np.memmap(idx_path, dtype=EVENT_RECORD_DTYPE, mode='r+', 
                    offset=header_size, shape=(num_events,))


class StreamingIndexWriter:
    """
    Writer for dynamically growing index files.
    """
    
    def __init__(self, idx_path: str, initial_capacity: int = 10000, growth_factor: float = 1.5):
        """
        Initialize streaming index writer.
        
        Args:
            idx_path: Path to the index file
            initial_capacity: Initial number of events to allocate
            growth_factor: Factor by which to grow when capacity exceeded
        """
        import pickle
        import struct
        
        self.idx_path = idx_path
        self.growth_factor = growth_factor
        self.event_count = 0
        self.capacity = initial_capacity
        
        # Calculate header size and read dtype from file
        with open(idx_path, 'rb') as f:
            dtype_size = struct.unpack('<I', f.read(4))[0]
            self.event_dtype = pickle.loads(f.read(dtype_size))
            self.header_size = 4 + dtype_size
        
        self.event_record_size = self.event_dtype.itemsize

        # Create initial memory map
        self.mmap = np.memmap(idx_path, dtype=self.event_dtype, mode='r+',
                             offset=self.header_size, shape=(self.capacity,))
    
    def write_event(self, event_record: np.ndarray) -> None:
        """
        Write a single event record, growing the file if necessary.
        
        Args:
            event_record: Single EventRecord to write
        """
        # Check if we need to grow the file
        if self.event_count >= self.capacity:
            self._grow_file()
        
        # Write the event
        self.mmap[self.event_count] = event_record
        self.event_count += 1
    
    def _grow_file(self) -> None:
        """
        Grow the memory-mapped file to accommodate more events.
        """
        # Calculate new capacity
        new_capacity = int(self.capacity * self.growth_factor)
        
        # Close current mmap
        del self.mmap
        
        # Extend the file
        current_size = os.path.getsize(self.idx_path)
        additional_bytes = (new_capacity - self.capacity) * self.event_record_size
        
        with open(self.idx_path, 'r+b') as f:
            f.seek(0, 2)  # Seek to end
            f.write(b'\x00' * additional_bytes)
        
        # Create new memory map with larger capacity
        self.mmap = np.memmap(self.idx_path, dtype=self.event_dtype, mode='r+',
                             offset=self.header_size, shape=(new_capacity,))
        
        self.capacity = new_capacity
        print(f"Expanded index file capacity to {new_capacity:,} events")
    
    def finalize(self) -> int:
        """
        Finalize the file by truncating to actual size.
        
        Returns:
            Number of events written
        """
        if self.event_count < self.capacity:
            # Truncate file to actual size
            final_size = self.header_size + (self.event_count * self.event_record_size)
            
            # Close mmap before truncating
            del self.mmap
            
            with open(self.idx_path, 'r+b') as f:
                f.truncate(final_size)
        
        return self.event_count


def load_mmap_files(input_path: str) -> Tuple[np.memmap, np.memmap]:
    """
    Load existing memory-mapped files for reading (old format without headers).
    
    Args:
        input_path: Base path for input files (without extension)
        
    Returns:
        Tuple of (index_mmap, data_mmap)
    """
    idx_path = f"{input_path}.idx"
    dat_path = f"{input_path}.dat"
    
    if not os.path.exists(idx_path) or not os.path.exists(dat_path):
        raise FileNotFoundError(f"Memory-mapped files not found: {input_path}")
    
    # Load old format files (no headers)
    index_mmap = np.memmap(idx_path, dtype=EVENT_RECORD_DTYPE, mode='r')
    data_mmap = np.memmap(dat_path, dtype=np.uint8, mode='r')
    
    return index_mmap, data_mmap


def load_ntmmap(input_path: str) -> Tuple[np.memmap, np.memmap, np.dtype]:
    """
    Load memory-mapped files with automatic dtype detection (new format with headers).
    Auto-detects source type (Prometheus vs IceCube) from stored event dtype.
    
    Args:
        input_path: Base path for input files (without extension)
        
    Returns:
        Tuple of (index_mmap, data_mmap, photon_dtype)
    """
    import pickle
    import struct
    
    idx_path = f"{input_path}.idx"
    dat_path = f"{input_path}.dat"
    
    if not os.path.exists(idx_path) or not os.path.exists(dat_path):
        raise FileNotFoundError(f"Memory-mapped files not found: {input_path}")
    
    # Load index file with header
    with open(idx_path, 'rb') as f:
        dtype_size = struct.unpack('<I', f.read(4))[0]
        event_dtype = pickle.loads(f.read(dtype_size))
        data_start = f.tell()
    
    index_mmap = np.memmap(idx_path, dtype=event_dtype, mode='r', offset=data_start)
    
    # Load data file with header
    with open(dat_path, 'rb') as f:
        dtype_size = struct.unpack('<I', f.read(4))[0]
        photon_dtype = pickle.loads(f.read(dtype_size))
        data_start = f.tell()
    
    # Memory map photons as structured array (not raw bytes)
    photons_array = np.memmap(dat_path, dtype=photon_dtype, mode='r', offset=data_start)
    
    return index_mmap, photons_array, photon_dtype



def append_photons_to_file(dat_path: str, photon_array: np.ndarray) -> None:
    """
    Append photon data to the data file.
    
    Args:
        dat_path: Path to the data file
        photon_array: Array of photon hits to append
    """
    with open(dat_path, 'ab') as f:
        f.write(photon_array.tobytes())


def get_event_photons(photons_array: np.memmap, event_record: np.ndarray) -> np.ndarray:
    """
    Extract photons for a specific event using photon indices.
    
    Args:
        photons_array: Memory-mapped photon array (structured)
        event_record: Single EventRecord
        
    Returns:
        Array of PhotonHit records
    """
    start_idx = int(event_record['photon_start_idx'])
    end_idx = int(event_record['photon_end_idx'])
    
    if start_idx >= end_idx:
        return np.array([], dtype=PHOTON_HIT_DTYPE)
    
    # Direct array slicing - much cleaner!
    return photons_array[start_idx:end_idx]