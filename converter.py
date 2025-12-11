#!/usr/bin/env python3
"""
Neutrino Telescope Memory-Mapped File Converter

Converts neutrino telescope data (Prometheus, IceCube) into efficient 
memory-mapped files for ML training and analysis.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from core.utils import print_dataset_info
from data.prometheus import convert_prometheus_to_mmap

# Conditional import for IceCube functionality
try:
    from data.icecube import convert_icecube_to_mmap
    ICECUBE_AVAILABLE = True
except ImportError:
    ICECUBE_AVAILABLE = False
    convert_icecube_to_mmap = None

def write_run_config(output_base: str, args: argparse.Namespace, num_events: int,
                     total_photons: int, elapsed_seconds: float,
                     skipped_events: int = 0) -> str:
    """Persist conversion parameters and summary statistics alongside outputs."""
    config_path = f"{output_base}.config.json"
    payload = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "command": " ".join(sys.argv),
        "args": {k: v for k, v in vars(args).items()},
        "summary": {
            "events_converted": num_events,
            "events_skipped": skipped_events,
            "photons_written": total_photons,
            "elapsed_seconds": round(elapsed_seconds, 3),
        },
    }

    config_dir = os.path.dirname(config_path)
    if config_dir and not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return config_path


def print_conversion_summary(output_base: str, num_events: int,
                             total_photons: int, elapsed_seconds: float,
                             include_dataset_info: bool,
                             skipped_events: int = 0) -> None:
    """Display conversion throughput, file sizes, and optional dataset stats."""
    print(f"\n=== Conversion Summary ===")
    print(f"Events converted: {num_events:,}")
    if skipped_events:
        print(f"Events skipped by filters: {skipped_events:,}")
    print(f"Total photons: {total_photons:,}")
    print(f"Time elapsed: {elapsed_seconds:.1f} seconds")
    if elapsed_seconds > 0:
        print(f"Events/sec: {num_events / elapsed_seconds:.1f}")
        print(f"Photons/sec: {total_photons / elapsed_seconds:,.0f}")

    idx_path = f"{output_base}.idx"
    dat_path = f"{output_base}.dat"

    if os.path.exists(idx_path) and os.path.exists(dat_path):
        idx_size = os.path.getsize(idx_path) / (1024 * 1024)  # MB
        dat_size = os.path.getsize(dat_path) / (1024 * 1024)  # MB
        total_size = idx_size + dat_size

        print(f"\nOutput files:")
        print(f"  Index: {idx_size:.1f} MB ({idx_path})")
        print(f"  Data: {dat_size:.1f} MB ({dat_path})")
        print(f"  Total: {total_size:.1f} MB")

    if include_dataset_info:
        print_dataset_info(output_base)


def main():
    parser = argparse.ArgumentParser(
        description="Convert neutrino telescope data to memory-mapped format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Determine available source choices based on dependencies
    available_sources = ["prometheus"]
    if ICECUBE_AVAILABLE:
        available_sources.append("icecube")
    
    parser.add_argument(
        "--source", 
        choices=available_sources, 
        required=True,
        help=f"Source data format. Available: {', '.join(available_sources)}"
    )
    
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more input directories containing source data files"
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
        "--grouping-window-ns",
        type=float,
        default=0.0,
        help="Time window for grouping hits by sensor in nanoseconds (0 = no grouping, use raw hits)"
    )
    parser.add_argument(
        "--min-photons",
        type=int,
        default=None,
        help="Minimum number of photon hits required to keep a Prometheus event (checked before grouping)"
    )
    parser.add_argument(
        "--max-photons",
        type=int,
        default=None,
        help="Maximum number of photon hits allowed to keep a Prometheus event (checked before grouping)"
    )
    parser.add_argument(
        "--min-channels",
        type=int,
        default=None,
        help="Minimum number of hit sensors (unique string/sensor pairs) required to keep a Prometheus event"
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=None,
        help="Maximum number of hit sensors (unique string/sensor pairs) allowed to keep a Prometheus event"
    )
    
    parser.add_argument(
        "--pulse-key",
        type=str,
        default="SplitInIceDSTPulses",
        help="Name of the pulse series to extract from i3 files"
    )

    parser.add_argument(
        "--filters",
        nargs="+",
        default=None,
        help="Filter names requiring condition_passed to include an IceCube event"
    )

    parser.add_argument(
        "--subevent-streams",
        nargs="+",
        default=["InIceSplit"],
        help="Sub-event stream names to include for IceCube frames"
    )
    
    args = parser.parse_args()
    
    # Validate input paths (allow multiple directories)
    input_paths = args.input if isinstance(args.input, list) else [args.input]
    missing_inputs = [path for path in input_paths if not os.path.exists(path)]
    if missing_inputs:
        print("Error: The following input paths do not exist:")
        for path in missing_inputs:
            print(f"  - {path}")
        sys.exit(1)
    args.input = input_paths
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"NT-MMap-Converter")
    print(f"Source: {args.source}")
    print(f"Input: {', '.join(args.input)}")
    print(f"Output: {args.output}")
    if args.file_range:
        print(f"File range: {args.file_range}")
    print()
    
    # Start conversion
    start_time = time.time()
    
    if args.source == "prometheus":
        num_events, total_photons, skipped_events = convert_prometheus_to_mmap(
            args.input,
            args.output,
            args.file_range,
            args.grouping_window_ns,
            args.min_photons,
            args.max_photons,
            args.min_channels,
            args.max_channels,
        )
    elif args.source == "icecube":
        if not ICECUBE_AVAILABLE:
            print("Error: IceCube functionality requires the IceCube software framework.")
            print("Please install IceCube software or use --source prometheus instead.")
            sys.exit(1)
        num_events, total_photons = convert_icecube_to_mmap(
            args.input, args.output, args.file_range, args.pulse_key,
            args.filters, args.subevent_streams
        )
        skipped_events = 0
    else:
        print(f"Error: Unsupported source format: {args.source}")
        sys.exit(1)
    
    # Conversion complete
    elapsed = time.time() - start_time

    print_conversion_summary(
        args.output, num_events, total_photons, elapsed, args.info, skipped_events
    )

    config_path = write_run_config(
        args.output, args, num_events, total_photons, elapsed, skipped_events
    )

    print(f"\n✓ Conversion completed successfully!")
    print(f"Memory-mapped files created: {args.output}.idx, {args.output}.dat")
    print(f"Run configuration saved to: {config_path}")


if __name__ == "__main__":
    main()
