#!/usr/bin/env python3
"""
Neutrino Telescope Memory-Mapped File Converter

Converts neutrino telescope data (Prometheus, IceCube) into efficient 
memory-mapped files for ML training and analysis.
"""

import argparse
import os
import sys
import time
from pathlib import Path

from core.utils import print_dataset_info
from data.prometheus import convert_prometheus_to_mmap
from data.icecube import convert_icecube_to_mmap


def main():
    parser = argparse.ArgumentParser(
        description="Convert neutrino telescope data to memory-mapped format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--source", 
        choices=["prometheus", "icecube"], 
        required=True,
        help="Source data format"
    )
    
    parser.add_argument(
        "--input", 
        required=True,
        help="Input directory containing source data files"
    )
    
    parser.add_argument(
        "--output", 
        required=True,
        help="Output path for memory-mapped files (without extension)"
    )
    
    parser.add_argument(
        "--file-range",
        type=str,
        default=None,
        help="Range of files to convert, e.g., '0-100' or '100-115'"
    )
    
    
    parser.add_argument(
        "--info",
        action="store_true", 
        help="Print dataset statistics after conversion"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--grouping-window-ns",
        type=float,
        default=0.0,
        help="Time window for grouping hits by sensor in nanoseconds (0 = no grouping, use raw hits)"
    )
    
    parser.add_argument(
        "--pulse-key",
        type=str,
        default="SplitInIceDSTPulses",
        help="Name of the pulse series to extract from i3 files"
    )
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.exists(args.input):
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"NT-MMap-Converter")
    print(f"Source: {args.source}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    if args.file_range:
        print(f"File range: {args.file_range}")
    print()
    
    # Start conversion
    start_time = time.time()
    
    try:
        if args.source == "prometheus":
            num_events, total_photons = convert_prometheus_to_mmap(
                args.input, args.output, args.file_range, args.grouping_window_ns
            )
        elif args.source == "icecube":
            num_events, total_photons = convert_icecube_to_mmap(
                args.input, args.output, args.file_range, args.pulse_key
            )
        else:
            print(f"Error: Unsupported source format: {args.source}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Conversion complete
    elapsed = time.time() - start_time
    
    print(f"\n=== Conversion Summary ===")
    print(f"Events converted: {num_events:,}")
    print(f"Total photons: {total_photons:,}")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print(f"Events/sec: {num_events / elapsed:.1f}")
    print(f"Photons/sec: {total_photons / elapsed:,.0f}")
    
    # Get file sizes
    idx_path = f"{args.output}.idx"
    dat_path = f"{args.output}.dat"
    
    if os.path.exists(idx_path) and os.path.exists(dat_path):
        idx_size = os.path.getsize(idx_path) / (1024 * 1024)  # MB
        dat_size = os.path.getsize(dat_path) / (1024 * 1024)  # MB
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
    
    print(f"\n✓ Conversion completed successfully!")
    print(f"Memory-mapped files created: {args.output}.idx, {args.output}.dat")


if __name__ == "__main__":
    main()