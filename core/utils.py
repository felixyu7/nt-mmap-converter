"""
Utilities for validation and dataset statistics.
"""

import numpy as np
import os
from typing import Dict, Any
from .mmap_format import load_ntmmap

def get_dataset_stats(input_path: str) -> Dict[str, Any]:
    """
    Get statistics about a memory-mapped dataset.
    
    Args:
        input_path: Base path to memory-mapped files (without extension)
        
    Returns:
        Dictionary with dataset statistics
    """
    index_mmap, data_mmap, photon_dtype = load_ntmmap(input_path)
    
    # Basic file info
    idx_size = os.path.getsize(f"{input_path}.idx")
    dat_size = os.path.getsize(f"{input_path}.dat")
    
    # Event statistics
    num_events = len(index_mmap)
    photon_counts = index_mmap['photon_end_idx'] - index_mmap['photon_start_idx']
    
    # Energy statistics
    energies = index_mmap['initial_energy']
    
    stats = {
        'num_events': int(num_events),
        'total_photons': int(np.sum(photon_counts)),
        'file_sizes': {
            'index_mb': idx_size / (1024 * 1024),
            'data_mb': dat_size / (1024 * 1024),
            'total_mb': (idx_size + dat_size) / (1024 * 1024),
        },
        'photon_stats': {
            'min': int(np.min(photon_counts)),
            'max': int(np.max(photon_counts)),
            'mean': float(np.mean(photon_counts)),
            'std': float(np.std(photon_counts)),
        },
        'energy_stats': {
            'min_gev': float(np.min(energies)),
            'max_gev': float(np.max(energies)),
            'mean_gev': float(np.mean(energies)),
            'std_gev': float(np.std(energies)),
        },
    }
    
    return stats


def print_dataset_info(input_path: str):
    """Print human-readable dataset information."""
    stats = get_dataset_stats(input_path)
    
    print(f"\n=== Dataset Statistics: {os.path.basename(input_path)} ===")
    print(f"Events: {stats['num_events']:,}")
    print(f"Total photons: {stats['total_photons']:,}")
    print(f"File sizes: {stats['file_sizes']['total_mb']:.1f} MB total")
    print(f"  - Index: {stats['file_sizes']['index_mb']:.1f} MB")
    print(f"  - Data: {stats['file_sizes']['data_mb']:.1f} MB")
    
    print(f"\nPhotons per event:")
    print(f"  - Range: {stats['photon_stats']['min']:,} to {stats['photon_stats']['max']:,}")
    print(f"  - Mean: {stats['photon_stats']['mean']:.1f} ± {stats['photon_stats']['std']:.1f}")
    
    print(f"\nInitial energy (GeV):")
    print(f"  - Range: {stats['energy_stats']['min_gev']:.2e} to {stats['energy_stats']['max_gev']:.2e}")
    print(f"  - Mean: {stats['energy_stats']['mean_gev']:.2e} ± {stats['energy_stats']['std_gev']:.2e}")

