"""
Filtering system for neutrino telescope memory-mapped datasets.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List
from .mmap_format import get_event_photons


class EventFilter(ABC):
    """Abstract base class for event filters."""
    
    @abstractmethod
    def passes_filter(self, event_record: np.ndarray, photons: np.ndarray) -> bool:
        """
        Check if an event passes this filter.
        
        Args:
            event_record: Single EventRecord from memory-mapped array
            photons: Array of PhotonHit records for this event
            
        Returns:
            True if event passes filter, False otherwise
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of this filter."""
        pass


class SensorCountFilter(EventFilter):
    """Filter events by minimum number of unique sensors hit."""
    
    def __init__(self, min_sensors: int):
        """
        Initialize sensor count filter.
        
        Args:
            min_sensors: Minimum number of unique sensors required
        """
        self.min_sensors = min_sensors
    
    def passes_filter(self, event_record: np.ndarray, photons: np.ndarray) -> bool:
        """Check if event has minimum unique sensors."""
        if len(photons) == 0:
            return self.min_sensors <= 0
        
        # Count unique (string_id, sensor_id) pairs efficiently
        sensor_pairs = np.column_stack((photons['string_id'], photons['sensor_id']))
        unique_sensors = np.unique(sensor_pairs, axis=0)
        
        return len(unique_sensors) >= self.min_sensors
    
    def get_description(self) -> str:
        return f"minimum {self.min_sensors} unique sensors"


class EnergyRangeFilter(EventFilter):
    """Filter events by initial energy range."""
    
    def __init__(self, min_energy: float = None, max_energy: float = None):
        """
        Initialize energy range filter.
        
        Args:
            min_energy: Minimum energy in GeV (None for no minimum)
            max_energy: Maximum energy in GeV (None for no maximum)
        """
        self.min_energy = min_energy
        self.max_energy = max_energy
        
        if min_energy is None and max_energy is None:
            raise ValueError("At least one of min_energy or max_energy must be specified")
    
    def passes_filter(self, event_record: np.ndarray, photons: np.ndarray) -> bool:
        """Check if event energy is within range."""
        energy = float(event_record['initial_energy'])
        
        if self.min_energy is not None and energy < self.min_energy:
            return False
        if self.max_energy is not None and energy > self.max_energy:
            return False
            
        return True
    
    def get_description(self) -> str:
        if self.min_energy is not None and self.max_energy is not None:
            return f"energy range {self.min_energy:.2e}-{self.max_energy:.2e} GeV"
        elif self.min_energy is not None:
            return f"energy >= {self.min_energy:.2e} GeV"
        else:
            return f"energy <= {self.max_energy:.2e} GeV"


class PhotonCountFilter(EventFilter):
    """Filter events by photon count range."""
    
    def __init__(self, min_photons: int = None, max_photons: int = None):
        """
        Initialize photon count filter.
        
        Args:
            min_photons: Minimum photon count (None for no minimum)
            max_photons: Maximum photon count (None for no maximum)
        """
        self.min_photons = min_photons
        self.max_photons = max_photons
        
        if min_photons is None and max_photons is None:
            raise ValueError("At least one of min_photons or max_photons must be specified")
    
    def passes_filter(self, event_record: np.ndarray, photons: np.ndarray) -> bool:
        """Check if event photon count is within range."""
        photon_count = len(photons)
        
        if self.min_photons is not None and photon_count < self.min_photons:
            return False
        if self.max_photons is not None and photon_count > self.max_photons:
            return False
            
        return True
    
    def get_description(self) -> str:
        if self.min_photons is not None and self.max_photons is not None:
            return f"photon count {self.min_photons}-{self.max_photons}"
        elif self.min_photons is not None:
            return f"photon count >= {self.min_photons}"
        else:
            return f"photon count <= {self.max_photons}"


class CoGFilter(EventFilter):
    """Filter events by center of gravity distance from detector center."""
    
    def __init__(self, max_radius: float, detector_center: Tuple[float, float, float] = (0.0, 0.0, -2000.0)):
        """
        Initialize center of gravity filter.
        
        Args:
            max_radius: Maximum allowed distance from detector center
            detector_center: Detector center coordinates (x, y, z). Default is Prometheus center.
        """
        self.max_radius = max_radius
        self.detector_center = np.array(detector_center, dtype=np.float64)
    
    def passes_filter(self, event_record: np.ndarray, photons: np.ndarray) -> bool:
        """Check if event center of gravity is within radius."""
        if len(photons) == 0:
            return False
        
        # Get photon positions and charges
        positions = np.column_stack([photons['x'], photons['y'], photons['z']])
        charges = photons['charge']
        
        # Handle case where all charges are zero
        total_charge = np.sum(charges)
        if total_charge == 0:
            return False
        
        # Calculate charge-weighted center of gravity
        cog = np.sum(positions * charges[:, np.newaxis], axis=0) / total_charge
        
        # Calculate distance from detector center
        distance = np.linalg.norm(cog - self.detector_center)
        
        return distance <= self.max_radius
    
    def get_description(self) -> str:
        return f"CoG within {self.max_radius:.1f}m of detector center {self.detector_center}"


class InelasticityFilter(EventFilter):
    """Filter events by inelasticity (bjorken_y) range."""
    
    def __init__(self, min_inelasticity: float = None, max_inelasticity: float = None):
        """
        Initialize inelasticity filter.
        
        Args:
            min_inelasticity: Minimum inelasticity (bjorken_y) (None for no minimum)
            max_inelasticity: Maximum inelasticity (bjorken_y) (None for no maximum)
        """
        self.min_inelasticity = min_inelasticity
        self.max_inelasticity = max_inelasticity
        
        if min_inelasticity is None and max_inelasticity is None:
            raise ValueError("At least one of min_inelasticity or max_inelasticity must be specified")
    
    def passes_filter(self, event_record: np.ndarray, photons: np.ndarray) -> bool:
        """Check if event inelasticity is within range."""
        inelasticity = float(event_record['bjorken_y'])
        
        if self.min_inelasticity is not None and inelasticity < self.min_inelasticity:
            return False
        if self.max_inelasticity is not None and inelasticity > self.max_inelasticity:
            return False
            
        return True
    
    def get_description(self) -> str:
        if self.min_inelasticity is not None and self.max_inelasticity is not None:
            return f"inelasticity range {self.min_inelasticity:.3f}-{self.max_inelasticity:.3f}"
        elif self.min_inelasticity is not None:
            return f"inelasticity >= {self.min_inelasticity:.3f}"
        else:
            return f"inelasticity <= {self.max_inelasticity:.3f}"


class StartingEventFilter(EventFilter):
    """Filter for starting events where interaction vertex is within detector volume."""
    
    def __init__(self, detector_center: Tuple[float, float, float] = (0.0, 0.0, -2000.0),
                 detector_radius: float = 450.0, detector_height: float = 900.0):
        """
        Initialize starting event filter.
        
        Args:
            detector_center: Center of cylindrical detector (x, y, z)
            detector_radius: Radius of cylindrical detector in meters
            detector_height: Height of cylindrical detector in meters
        """
        self.detector_center = np.array(detector_center, dtype=np.float64)
        self.detector_radius = detector_radius
        self.detector_height = detector_height
    
    def passes_filter(self, event_record: np.ndarray, photons: np.ndarray) -> bool:
        """Check if interaction vertex is within cylindrical detector volume."""
        # Get interaction vertex position
        vertex_x = float(event_record['initial_x'])
        vertex_y = float(event_record['initial_y'])
        vertex_z = float(event_record['initial_z'])
        
        # Calculate radial distance from detector center
        dx = vertex_x - self.detector_center[0]
        dy = vertex_y - self.detector_center[1]
        radial_distance = np.sqrt(dx*dx + dy*dy)
        
        # Check if within radius
        if radial_distance > self.detector_radius:
            return False
        
        # Check if within height bounds (centered on detector_center[2])
        z_min = self.detector_center[2] - self.detector_height / 2
        z_max = self.detector_center[2] + self.detector_height / 2
        
        return z_min <= vertex_z <= z_max
    
    def get_description(self) -> str:
        return (f"starting events (vertex within cylinder: center={tuple(self.detector_center)}, "
                f"radius={self.detector_radius}m, height={self.detector_height}m)")


class CompositeFilter(EventFilter):
    """Combine multiple filters with AND logic."""
    
    def __init__(self, filters: List[EventFilter]):
        """
        Initialize composite filter.
        
        Args:
            filters: List of filters to combine (all must pass)
        """
        self.filters = filters
        if not filters:
            raise ValueError("At least one filter must be provided")
    
    def passes_filter(self, event_record: np.ndarray, photons: np.ndarray) -> bool:
        """Check if event passes all component filters."""
        return all(f.passes_filter(event_record, photons) for f in self.filters)
    
    def get_description(self) -> str:
        descriptions = [f.get_description() for f in self.filters]
        return " AND ".join(descriptions)



def apply_filters_to_dataset(events_mmap: np.memmap, photons_mmap: np.memmap, 
                           filters: List[EventFilter]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply filters to entire dataset and return qualifying event indices.
    
    Args:
        events_mmap: Memory-mapped events array
        photons_mmap: Memory-mapped photons array
        filters: List of filters to apply
        
    Returns:
        Tuple of (qualifying_event_indices, filter_stats)
    """
    if not filters:
        # No filters - all events pass
        return np.arange(len(events_mmap)), {"total_events": len(events_mmap), "filtered_events": len(events_mmap)}
    
    # Combine filters
    composite_filter = CompositeFilter(filters)
    
    qualifying_indices = []
    total_events = len(events_mmap)
    
    print(f"Applying filters: {composite_filter.get_description()}")
    print(f"Scanning {total_events:,} events...")
    
    for i in range(total_events):
        if i % 10000 == 0 and i > 0:
            print(f"  Processed {i:,} events, {len(qualifying_indices):,} qualify so far")
        
        event = events_mmap[i]
        photons = get_event_photons(photons_mmap, event)
        
        if composite_filter.passes_filter(event, photons):
            qualifying_indices.append(i)
    
    qualifying_indices = np.array(qualifying_indices)
    
    stats = {
        "total_events": total_events,
        "filtered_events": len(qualifying_indices),
        "filter_description": composite_filter.get_description(),
        "pass_rate": len(qualifying_indices) / total_events if total_events > 0 else 0.0
    }
    
    print(f"Filtering complete: {len(qualifying_indices):,} / {total_events:,} events pass ({stats['pass_rate']*100:.1f}%)")
    
    return qualifying_indices, stats