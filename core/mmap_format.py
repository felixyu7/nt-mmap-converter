"""
Memory-mapped file format for neutrino telescope data.
"""

import numpy as np
import os
from typing import Tuple, Dict, Any

# Define the event record structure
EVENT_RECORD_DTYPE = np.dtype([
    # Indexing (photon indices, not byte offsets)
    ('photon_start_idx', np.uint32),
    ('photon_end_idx', np.uint32),
    
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
EVENT_RECORD_SIZE = EVENT_RECORD_DTYPE.itemsize
PHOTON_HIT_SIZE = PHOTON_HIT_DTYPE.itemsize


class EventRecord:
    """Helper class for creating EventRecord arrays."""
    
    @staticmethod
    def create_array(num_events: int) -> np.ndarray:
        """Create a zeroed array of EventRecords."""
        return np.zeros(num_events, dtype=EVENT_RECORD_DTYPE)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> np.ndarray:
        """Create an EventRecord from a dictionary."""
        record = np.zeros(1, dtype=EVENT_RECORD_DTYPE)[0]
        
        # Fill scalar fields
        for field in ['initial_energy', 'initial_zenith', 'initial_azimuth',
                     'initial_x', 'initial_y', 'initial_z', 'bjorken_x',
                     'bjorken_y', 'column_depth', 'interaction', 'initial_type']:
            if field in data:
                record[field] = data[field]
        
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
    Create memory-mapped files with dtype headers for writing.
    
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