#!/usr/bin/env python3
"""
Filter Memory-Mapped Neutrino Telescope Datasets

Filters existing memory-mapped datasets based on various criteria
and creates new filtered datasets in the same format.
"""

import argparse
import os
import sys
import time
from pathlib import Path

from core.mmap_format import load_ntmmap, create_mmap_files_with_headers, create_index_mmap_with_header, get_event_photons
from core.filters import SensorCountFilter, EnergyRangeFilter, PhotonCountFilter, CoGFilter, StartingEventFilter, InelasticityFilter, apply_filters_to_dataset
from core.utils import print_dataset_info


def parse_energy_range(value_str: str):
    """Parse energy range from command line argument."""
    try:
        parts = value_str.split('-')
        if len(parts) != 2:
            raise ValueError("Energy range must be in format 'min-max'")
        min_energy = float(parts[0]) if parts[0] else None
        max_energy = float(parts[1]) if parts[1] else None
        return min_energy, max_energy
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid energy range '{value_str}': {e}")


def parse_photon_range(value_str: str):
    """Parse photon count range from command line argument."""
    try:
        parts = value_str.split('-')
        if len(parts) != 2:
            raise ValueError("Photon range must be in format 'min-max'")
        min_photons = int(parts[0]) if parts[0] else None
        max_photons = int(parts[1]) if parts[1] else None
        return min_photons, max_photons
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid photon range '{value_str}': {e}")


def parse_inelasticity_range(value_str: str):
    """Parse inelasticity range from command line argument."""
    try:
        parts = value_str.split('-')
        if len(parts) != 2:
            raise ValueError("Inelasticity range must be in format 'min-max'")
        min_inelasticity = float(parts[0]) if parts[0] else None
        max_inelasticity = float(parts[1]) if parts[1] else None
        return min_inelasticity, max_inelasticity
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid inelasticity range '{value_str}': {e}")


def copy_filtered_data(input_path: str, output_path: str, qualifying_indices, verbose: bool = False):
    """
    Copy filtered events and their photons to new memory-mapped files.
    
    Args:
        input_path: Input dataset path (without extension)
        output_path: Output dataset path (without extension)
        qualifying_indices: Array of event indices that pass the filter
        verbose: Enable verbose output
    """
    print("Loading input dataset...")
    events_mmap, photons_mmap, photon_dtype = load_ntmmap(input_path)
    
    if len(qualifying_indices) == 0:
        print("No events pass the filter. Creating empty dataset.")
        # Create empty files
        create_mmap_files_with_headers(output_path, 0)
        return
    
    # Count total photons needed
    print("Counting photons in filtered events...")
    total_photons = 0
    for i, event_idx in enumerate(qualifying_indices):
        event = events_mmap[event_idx]
        photon_count = int(event['photon_end_idx'] - event['photon_start_idx'])
        total_photons += photon_count
        
        if verbose and (i + 1) % 1000 == 0:
            print(f"  Counted {i + 1:,} events, {total_photons:,} photons so far")
    
    print(f"Creating output files with {len(qualifying_indices):,} events and {total_photons:,} photons...")
    
    # Create output files
    idx_path, dat_path = create_mmap_files_with_headers(output_path, len(qualifying_indices))
    
    # Create memory map for output index
    output_events_mmap = create_index_mmap_with_header(idx_path, len(qualifying_indices))
    
    # Copy filtered data
    print("Copying filtered data...")
    current_photon_idx = 0
    
    with open(dat_path, 'ab') as data_file:  # Append to preserve header
        for i, event_idx in enumerate(qualifying_indices):
            # Copy event record
            original_event = events_mmap[event_idx]
            event_copy = original_event.copy()
            
            # Get photons for this event
            photons = get_event_photons(photons_mmap, original_event)
            num_photons = len(photons)
            
            # Update photon indices for new file
            event_copy['photon_start_idx'] = current_photon_idx
            event_copy['photon_end_idx'] = current_photon_idx + num_photons
            
            # Store event record
            output_events_mmap[i] = event_copy
            
            # Write photons to data file
            if num_photons > 0:
                photon_bytes = photons.tobytes()
                data_file.write(photon_bytes)
                current_photon_idx += num_photons
            
            if verbose and (i + 1) % 1000 == 0:
                print(f"  Copied {i + 1:,} events, {current_photon_idx:,} photons")
    
    # Flush output
    output_events_mmap.flush()
    
    print(f"Filtered dataset created: {output_path}.idx, {output_path}.dat")


def main():
    parser = argparse.ArgumentParser(
        description="Filter memory-mapped neutrino telescope datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="Input dataset path (without extension)"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output dataset path (without extension)"
    )
    
    # Filter options
    parser.add_argument(
        "--min-unique-sensors",
        type=int,
        help="Minimum number of unique sensors hit per event"
    )
    
    parser.add_argument(
        "--energy-range",
        type=parse_energy_range,
        help="Energy range in GeV (format: 'min-max', use empty for unbounded: '-1e6' or '1e3-')"
    )
    
    parser.add_argument(
        "--photon-range",
        type=parse_photon_range,
        help="Photon count range (format: 'min-max', use empty for unbounded: '-1000' or '100-')"
    )
    
    parser.add_argument(
        "--cog-radius",
        type=float,
        help="Maximum center-of-gravity distance from detector center (meters, Prometheus center: [0,0,-2000])"
    )
    
    parser.add_argument(
        "--starting",
        action="store_true",
        help="Filter for starting events (interaction vertex within detector volume)"
    )
    
    parser.add_argument(
        "--inelasticity-range",
        type=parse_inelasticity_range,
        help="Inelasticity (bjorken_y) range (format: 'min-max', use empty for unbounded: '-0.8' or '0.1-')"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print dataset statistics after filtering"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(f"{args.input}.idx") or not os.path.exists(f"{args.input}.dat"):
        print(f"Error: Input dataset not found: {args.input}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Build filter list
    filters = []
    
    if args.min_unique_sensors is not None:
        filters.append(SensorCountFilter(args.min_unique_sensors))
    
    if args.energy_range is not None:
        min_energy, max_energy = args.energy_range
        filters.append(EnergyRangeFilter(min_energy, max_energy))
    
    if args.photon_range is not None:
        min_photons, max_photons = args.photon_range
        filters.append(PhotonCountFilter(min_photons, max_photons))
    
    if args.cog_radius is not None:
        filters.append(CoGFilter(args.cog_radius))
    
    if args.starting:
        filters.append(StartingEventFilter())
    
    if args.inelasticity_range is not None:
        min_inelasticity, max_inelasticity = args.inelasticity_range
        filters.append(InelasticityFilter(min_inelasticity, max_inelasticity))
    
    if not filters:
        print("Error: No filters specified. Use --help to see available filter options.")
        sys.exit(1)
    
    print("NT-MMap Dataset Filter")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print()
    
    # Load dataset and apply filters
    start_time = time.time()
    
    try:
        print("Loading input dataset...")
        events_mmap, photons_mmap, photon_dtype = load_ntmmap(args.input)
        
        # Apply filters to find qualifying events
        qualifying_indices, filter_stats = apply_filters_to_dataset(events_mmap, photons_mmap, filters)
        
        # Copy filtered data to new files
        copy_filtered_data(args.input, args.output, qualifying_indices, args.verbose)
        
    except Exception as e:
        print(f"Error during filtering: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Summary
    elapsed = time.time() - start_time
    
    print(f"\n=== Filtering Summary ===")
    print(f"Original events: {filter_stats['total_events']:,}")
    print(f"Filtered events: {filter_stats['filtered_events']:,}")
    print(f"Pass rate: {filter_stats['pass_rate']*100:.1f}%")
    print(f"Filter: {filter_stats['filter_description']}")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    
    # Get output file sizes
    idx_path = f"{args.output}.idx"
    dat_path = f"{args.output}.dat"
    
    if os.path.exists(idx_path) and os.path.exists(dat_path):
        idx_size = os.path.getsize(idx_path) / (1024 * 1024)
        dat_size = os.path.getsize(dat_path) / (1024 * 1024) 
        total_size = idx_size + dat_size
        
        print(f"\nOutput files:")
        print(f"  Index: {idx_size:.1f} MB ({idx_path})")
        print(f"  Data: {dat_size:.1f} MB ({dat_path})")
        print(f"  Total: {total_size:.1f} MB")
    
    # Dataset info
    if args.info:
        try:
            print_dataset_info(args.output)
        except Exception as e:
            print(f"Warning: Could not generate dataset info: {e}")
    
    print(f"\n✓ Filtering completed successfully!")


if __name__ == "__main__":
    main()