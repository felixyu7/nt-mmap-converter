"""
Core module for neutrino telescope memory-mapped file format.
"""

from .mmap_format import EventRecord, PhotonHit, load_mmap_files, load_ntmmap
from .utils import get_dataset_stats

__all__ = ['EventRecord', 'PhotonHit', 'load_mmap_files', 'load_ntmmap',
           'get_dataset_stats']