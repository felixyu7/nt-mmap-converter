"""
Prometheus neutrino telescope data parser.
Handles conversion from parquet files to memory-mapped format.
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Iterator, Tuple, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.mmap_format import EventRecord, PhotonHit

def group_hits_by_window(hit_times, hit_charges, time_window, return_counts=False):
    """
    Group hits into fixed time windows, returning the first actual hit time
    in each non-empty window and the sum of charges in that window.

    Parameters
    ----------
    hit_times : array-like, shape (N,)
        Hit times in nanoseconds.
    hit_charges : array-like, shape (N,)
        Charge per hit (e.g., photoelectrons). Must align with hit_times.
    time_window : float
        Window size in nanoseconds (> 0).
    return_counts : bool, optional (default: False)
        If True, also return the number of hits per window.

    Returns
    -------
    grouped_times : np.ndarray, shape (M,)
        First actual hit time in each non-empty window (ascending by window).
    window_charges : np.ndarray, shape (M,)
        Sum of hit_charges within each window.
    hit_counts : np.ndarray, shape (M,), optional
        Number of hits in each window (only if return_counts=True).
    """
    ht = np.asarray(hit_times)
    hc = np.asarray(hit_charges)

    if ht.size == 0:
        if return_counts:
            return ht[:0], ht[:0].astype(float), ht[:0]
        else:
            return ht[:0], ht[:0].astype(float)

    if ht.shape != hc.shape:
        raise ValueError("hit_times and hit_charges must have the same shape.")
    if time_window <= 0:
        raise ValueError("time_window must be positive.")

    # Stable sort by time (stable ensures the first time in each bin is preserved if equal times occur).
    order = np.argsort(ht, kind="mergesort")
    st = ht[order]
    sc = hc[order]

    # Compute monotone bin labels with numerically robust arithmetic.
    if np.issubdtype(st.dtype, np.integer) and float(time_window).is_integer():
        tw = np.int64(time_window)
        bins = (st - st[0]) // tw
    else:
        # Cast to float64 and shift by st[0] for better precision at boundaries.
        bins = np.floor((st - st[0]).astype(np.float64) / float(time_window)).astype(np.int64)

    # Run-length encode the (sorted, hence monotone) bin labels.
    changes = np.empty(bins.size, dtype=bool)
    changes[0] = True
    np.not_equal(bins[1:], bins[:-1], out=changes[1:])
    starts = np.flatnonzero(changes)  # start index of each bin-run

    # First hit time per non-empty bin:
    grouped_times = st[starts]

    # Aggregate charges per bin efficiently:
    window_charges = np.add.reduceat(sc, starts)

    if return_counts:
        hit_counts = np.diff(np.r_[starts, st.size])
        return grouped_times, window_charges, hit_counts
    else:
        return grouped_times, window_charges

def find_parquet_files(input_path: str) -> list:
    """
    Find all parquet files in the input directory.
    
    Args:
        input_path: Directory containing .parquet files
        
    Returns:
        List of parquet file paths, sorted appropriately
    """
    if not os.path.isdir(input_path):
        raise ValueError(f"Input path is not a directory: {input_path}")
    
    # Find all parquet files
    pattern = os.path.join(input_path, "*.parquet")
    files = glob.glob(pattern)
    
    if not files:
        raise ValueError(f"No .parquet files found in {input_path}")
    
    # Sort alphabetically for consistent ordering
    files.sort()
    return files


def parse_mc_truth(mc_truth_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse MC truth information from Prometheus format.
    
    Args:
        mc_truth_dict: Raw mc_truth dictionary from parquet
        
    Returns:
        Cleaned dictionary suitable for EventRecord
    """
    # Copy scalar fields
    parsed = {
        'initial_energy': mc_truth_dict['initial_state_energy'],
        'initial_zenith': mc_truth_dict['initial_state_zenith'],
        'initial_azimuth': mc_truth_dict['initial_state_azimuth'],
        'initial_x': mc_truth_dict['initial_state_x'],
        'initial_y': mc_truth_dict['initial_state_y'],
        'initial_z': mc_truth_dict['initial_state_z'],
        'bjorken_x': mc_truth_dict['bjorken_x'],
        'bjorken_y': mc_truth_dict['bjorken_y'],
        'column_depth': mc_truth_dict['column_depth'],
        'interaction': mc_truth_dict['interaction'],
        'initial_type': mc_truth_dict['initial_state_type'],
    }
    
    # Handle final state arrays
    final_state_fields = [
        'final_state_energy', 'final_state_type', 'final_state_zenith',
        'final_state_azimuth', 'final_state_x', 'final_state_y', 
        'final_state_z', 'final_state_parent'
    ]
    
    for field in final_state_fields:
        if field in mc_truth_dict:
            # Remove "final_state_" prefix for our format
            clean_field = field.replace('final_state_', 'final_')
            parsed[clean_field] = mc_truth_dict[field]
    
    return parsed


def parse_photons(photons_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Parse photon information from Prometheus format.
    
    Args:
        photons_dict: Raw photons dictionary from parquet
        
    Returns:
        Cleaned dictionary suitable for PhotonHit
    """
    return {
        'sensor_pos_x': photons_dict['sensor_pos_x'],
        'sensor_pos_y': photons_dict['sensor_pos_y'], 
        'sensor_pos_z': photons_dict['sensor_pos_z'],
        't': photons_dict['t'],
        'string_id': photons_dict['string_id'],
        'sensor_id': photons_dict['sensor_id'],
        'id_idx': photons_dict['id_idx'],
    }


def process_photons_with_grouping(photons_dict: Dict[str, np.ndarray], 
                                 grouping_window_ns: float) -> Dict[str, np.ndarray]:
    """
    Process photons with optional hit grouping per sensor.
    
    Args:
        photons_dict: Raw photons dictionary from parquet
        grouping_window_ns: Time window for grouping hits by sensor (0 = no grouping)
        
    Returns:
        Processed photons dictionary with optional grouping applied
    """
    if grouping_window_ns <= 0:
        # No grouping - return raw hits with charge=1
        result = parse_photons(photons_dict)
        result['charge'] = np.ones(len(result['t']), dtype=np.float32)
        return result
    
    # Optimized unique sensor detection using safe hash-based approach
    string_ids = photons_dict['string_id']
    sensor_ids = photons_dict['sensor_id']
    
    # Create safe combined keys using hash of (string_id, sensor_id) pairs
    # This avoids overflow issues and works with any sensor ID size
    sensor_pairs = np.column_stack((string_ids, sensor_ids))
    
    # Use lexsort to get sorted indices, then find unique groups
    sort_indices = np.lexsort((sensor_ids, string_ids))
    sorted_pairs = sensor_pairs[sort_indices]
    
    # Find boundaries where (string_id, sensor_id) changes
    unique_mask = np.ones(len(sorted_pairs), dtype=bool)
    unique_mask[1:] = (sorted_pairs[1:] != sorted_pairs[:-1]).any(axis=1)
    unique_indices = np.where(unique_mask)[0]
    
    # Pre-allocate result arrays with better size estimation
    n_photons = len(photons_dict['t'])
    n_unique_sensors = len(unique_indices)
    estimated_grouped = max(n_photons // 5, n_unique_sensors)  # Better estimate
    
    all_times = np.empty(estimated_grouped, dtype=np.float32)
    all_charges = np.empty(estimated_grouped, dtype=np.float32)
    all_sensor_pos_x = np.empty(estimated_grouped, dtype=np.float32)
    all_sensor_pos_y = np.empty(estimated_grouped, dtype=np.float32)
    all_sensor_pos_z = np.empty(estimated_grouped, dtype=np.float32)
    all_string_ids = np.empty(estimated_grouped, dtype=np.uint32)
    all_sensor_ids = np.empty(estimated_grouped, dtype=np.uint32)
    all_id_idx = np.empty(estimated_grouped, dtype=np.uint64)
    
    result_idx = 0
    
    # Process each unique sensor
    for i, unique_idx in enumerate(unique_indices):
        string_id, sensor_id = sorted_pairs[unique_idx]
        
        # Find end of this sensor group
        if i < len(unique_indices) - 1:
            end_idx = unique_indices[i + 1]
        else:
            end_idx = len(sorted_pairs)
        
        # Get original indices for this sensor group
        sensor_original_indices = sort_indices[unique_idx:end_idx]
        sensor_times = photons_dict['t'][sensor_original_indices]
        sensor_charges = np.ones(len(sensor_times), dtype=np.float32)
        
        if len(sensor_times) == 0:
            continue
            
        # Group hits by time window for this sensor
        grouped_times, grouped_charges = group_hits_by_window(
            sensor_times, sensor_charges, grouping_window_ns
        )
        
        if len(grouped_times) == 0:
            continue
        
        # Get metadata from first photon of this sensor
        first_original_idx = sensor_original_indices[0]
        n_grouped = len(grouped_times)
        
        # Ensure we have enough space
        while result_idx + n_grouped > len(all_times):
            # Double the array size
            new_size = len(all_times) * 2
            all_times = np.resize(all_times, new_size)
            all_charges = np.resize(all_charges, new_size)
            all_sensor_pos_x = np.resize(all_sensor_pos_x, new_size)
            all_sensor_pos_y = np.resize(all_sensor_pos_y, new_size)
            all_sensor_pos_z = np.resize(all_sensor_pos_z, new_size)
            all_string_ids = np.resize(all_string_ids, new_size)
            all_sensor_ids = np.resize(all_sensor_ids, new_size)
            all_id_idx = np.resize(all_id_idx, new_size)
        
        # Copy data efficiently using array slicing
        end_result_idx = result_idx + n_grouped
        all_times[result_idx:end_result_idx] = grouped_times
        all_charges[result_idx:end_result_idx] = grouped_charges
        all_sensor_pos_x[result_idx:end_result_idx] = photons_dict['sensor_pos_x'][first_original_idx]
        all_sensor_pos_y[result_idx:end_result_idx] = photons_dict['sensor_pos_y'][first_original_idx]
        all_sensor_pos_z[result_idx:end_result_idx] = photons_dict['sensor_pos_z'][first_original_idx]
        all_string_ids[result_idx:end_result_idx] = string_id
        all_sensor_ids[result_idx:end_result_idx] = sensor_id
        all_id_idx[result_idx:end_result_idx] = photons_dict['id_idx'][first_original_idx]
        
        result_idx = end_result_idx
    
    # Trim arrays to actual size and return
    return {
        'sensor_pos_x': all_sensor_pos_x[:result_idx].copy(),
        'sensor_pos_y': all_sensor_pos_y[:result_idx].copy(),
        'sensor_pos_z': all_sensor_pos_z[:result_idx].copy(),
        't': all_times[:result_idx].copy(),
        'charge': all_charges[:result_idx].copy(),
        'string_id': all_string_ids[:result_idx].copy(),
        'sensor_id': all_sensor_ids[:result_idx].copy(),
        'id_idx': all_id_idx[:result_idx].copy(),
    }




def iter_prometheus_events(parquet_files: list) -> Iterator[Tuple[Dict[str, Any], Dict[str, np.ndarray]]]:
    """
    Iterate over all events in Prometheus parquet files.
    
    Args:
        parquet_files: List of parquet file paths
        
    Yields:
        Tuple of (mc_truth_dict, photons_dict) for each event
    """
    for file_path in parquet_files:
        print(f"Processing {os.path.basename(file_path)}...")
        
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
        
        for idx in range(len(df)):
            # Extract mc_truth
            mc_truth_raw = df['mc_truth'].iloc[idx]
            mc_truth = parse_mc_truth(mc_truth_raw)
            
            # Extract photons  
            photons_raw = df['photons'].iloc[idx]
            
            yield mc_truth, photons_raw


def convert_prometheus_to_mmap(input_path: str, output_path: str,
                              file_range: str = None, grouping_window_ns: float = 0.0) -> Tuple[int, int]:
    """
    Convert Prometheus parquet files to memory-mapped format using streaming approach.
    
    Args:
        input_path: Directory containing chunk_*.parquet files
        output_path: Output path for memory-mapped files (without extension)
        file_range: Range of files to convert, e.g., '0-100' or '100-115'
        grouping_window_ns: Time window for hit grouping per sensor (0 = no grouping)
        
    Returns:
        Tuple of (num_events_converted, total_photons)
    """
    
    # Find input files
    parquet_files = find_parquet_files(input_path)
    print(f"Found {len(parquet_files)} parquet files")
    
    # Limit files if specified
    if file_range:
        try:
            start, end = map(int, file_range.split('-'))
            parquet_files = parquet_files[start:end]
            print(f"Processing files from index {start} to {end}")
        except ValueError:
            print(f"Invalid file range format: {file_range}. Processing all files.")
    
    print(f"Converting events from {len(parquet_files)} files using streaming approach...")
    
    # Create streaming memory-mapped files
    from core.mmap_format import create_streaming_mmap_files, StreamingIndexWriter, append_photons_to_file
    
    # Estimate events per file for initial allocation (Prometheus files are typically larger)
    events_per_file_estimate = 5000  # Higher estimate for parquet files
    initial_estimate = len(parquet_files) * events_per_file_estimate
    
    idx_path, data_file_path = create_streaming_mmap_files(output_path, initial_estimate, source_type='prometheus')
    index_writer = StreamingIndexWriter(idx_path, initial_estimate)
    
    # Convert events
    total_photons = 0
    current_photon_idx = 0
    
    for mc_truth, photons_raw in iter_prometheus_events(parquet_files):
        # Process photons with optional grouping
        photons = process_photons_with_grouping(photons_raw, grouping_window_ns)
        
        # Create photon array
        photon_array = PhotonHit.from_dict(photons)
        num_photons = len(photon_array)
        
        # Skip events with no photons - they're not useful for ML training
        if num_photons == 0:
            continue
            
        # Compute hit statistics
        mc_truth['num_hits'] = num_photons
        # Count unique sensor/string ID pairs (channels) for Prometheus
        sensor_string_pairs = np.column_stack([photons['string_id'], photons['sensor_id']])
        unique_channels = np.unique(sensor_string_pairs, axis=0)
        mc_truth['num_chans'] = len(unique_channels)
        
        # Create event record using Prometheus-specific dtype
        event_record = EventRecord.from_dict(mc_truth, source_type='prometheus')
        
        # Set photon indexing information
        event_record['photon_start_idx'] = current_photon_idx
        event_record['photon_end_idx'] = current_photon_idx + num_photons
        
        # Write event record (with dynamic growth)
        index_writer.write_event(event_record)
        
        # Append photons to data file
        append_photons_to_file(data_file_path, photon_array)
        current_photon_idx += num_photons
        total_photons += num_photons
        
        # Progress reporting
        if index_writer.event_count % 1000 == 0:
            print(f"Processed {index_writer.event_count:,} events, {total_photons:,} photons")
    
    # Finalize index file
    final_event_count = index_writer.finalize()
    
    print(f"Conversion complete: {final_event_count:,} events, {total_photons:,} total photons")
    print(f"Output files: {output_path}.idx, {output_path}.dat")
    
    return final_event_count, total_photons