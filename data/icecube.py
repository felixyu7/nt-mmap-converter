"""
IceCube neutrino telescope data parser.
Handles conversion from i3 files to memory-mapped format.
"""

import os
import glob
from typing import List, Iterator, Tuple, Dict, Any
import icecube
from icecube import dataio, dataclasses, icetray
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.mmap_format import EventRecord, PhotonHit


_PDGMAP = {
    dataclasses.I3Particle.ParticleType.Gamma: 22,
    dataclasses.I3Particle.ParticleType.EPlus: -11,
    dataclasses.I3Particle.ParticleType.EMinus: 11,
    dataclasses.I3Particle.ParticleType.MuPlus: -13,
    dataclasses.I3Particle.ParticleType.MuMinus: 13,
    dataclasses.I3Particle.ParticleType.TauPlus: -15,
    dataclasses.I3Particle.ParticleType.TauMinus: 15,
    dataclasses.I3Particle.ParticleType.NuE: 12,
    dataclasses.I3Particle.ParticleType.NuEBar: -12,
    dataclasses.I3Particle.ParticleType.NuMu: 14,
    dataclasses.I3Particle.ParticleType.NuMuBar: -14,
    dataclasses.I3Particle.ParticleType.NuTau: 16,
    dataclasses.I3Particle.ParticleType.NuTauBar: -16,
    dataclasses.I3Particle.ParticleType.Pi0: 111,
    dataclasses.I3Particle.ParticleType.PiPlus: 211,
    dataclasses.I3Particle.ParticleType.PiMinus: -211
}

def find_i3_files(input_path: str) -> List[str]:
    """Find all i3 files (including .i3.zst) in the input directory."""
    if not os.path.isdir(input_path):
        raise ValueError(f"Input path is not a directory: {input_path}")
    
    pattern_i3 = os.path.join(input_path, "*.i3")
    pattern_i3_zst = os.path.join(input_path, "*.i3.zst")
    
    files = glob.glob(pattern_i3) + glob.glob(pattern_i3_zst)
    
    if not files:
        raise ValueError(f"No .i3 or .i3.zst files found in {input_path}")
    
    files.sort()
    return files

def load_geometry(gcd_file: str) -> dataclasses.I3Geometry:
    """Load the I3Geometry from a GCD file."""
    if not os.path.exists(gcd_file):
        raise FileNotFoundError(f"GCD file not found: {gcd_file}")
    
    i3_file = dataio.I3File(gcd_file)
    g_frame = i3_file.pop_frame()
    while "I3Geometry" not in g_frame:
        g_frame = i3_file.pop_frame()
    i3_file.close()
    return g_frame["I3Geometry"]

def iter_i3_events(i3_files: List[str]) -> Iterator[icetray.I3Frame]:
    """Iterate over all physics frames in IceCube i3 files."""
    for file_path in i3_files:
        print(f"Processing {os.path.basename(file_path)}...")
        i3_file = dataio.I3File(file_path)
        while i3_file.more():
            frame = i3_file.pop_physics()
            if frame and frame.Has("I3EventHeader") and frame["I3EventHeader"].sub_event_stream != "NullSplit":
                yield frame
        i3_file.close()

def parse_pulses(frame: icetray.I3Frame, pulse_key: str, geometry: dataclasses.I3Geometry) -> Dict[str, np.ndarray]:
    """Parse pulse data from an I3Frame."""
    if pulse_key not in frame:
        return {
            'sensor_pos_x': np.array([]), 'sensor_pos_y': np.array([]),
            'sensor_pos_z': np.array([]), 't': np.array([]),
            'charge': np.array([]), 'string_id': np.array([]),
            'sensor_id': np.array([]), 'id_idx': np.array([])
        }

    pulses = frame[pulse_key]
    
    # Handle I3RecoPulseSeriesMapMask - get underlying pulse map
    if hasattr(pulses, 'apply'):
        source_key = pulse_key.replace('SplitInIce', 'InIce')
        if source_key in frame:
            pulses = pulses.apply(frame)
        else:
            pulses = []
    
    # Return empty arrays if no valid pulses
    if not pulses or not hasattr(pulses, '__iter__'):
        return {
            'sensor_pos_x': np.array([]), 'sensor_pos_y': np.array([]),
            'sensor_pos_z': np.array([]), 't': np.array([]),
            'charge': np.array([]), 'string_id': np.array([]),
            'sensor_id': np.array([]), 'id_idx': np.array([])
        }
    
    all_x, all_y, all_z, all_t, all_charge, all_string_id, all_sensor_id, all_id_idx = [], [], [], [], [], [], [], []
    
    for omkey, reco_pulses in pulses:
        if omkey not in geometry.omgeo:
            continue
            
        string_id = omkey.string
        sensor_id = omkey.om
        pos = geometry.omgeo[omkey].position
        x, y, z = pos.x, pos.y, pos.z
        
        for pulse in reco_pulses:
            all_x.append(x)
            all_y.append(y)
            all_z.append(z)
            all_t.append(pulse.time)
            all_charge.append(pulse.charge)
            all_string_id.append(string_id)
            all_sensor_id.append(sensor_id)
            all_id_idx.append(0)
            
    return {
        'sensor_pos_x': np.array(all_x, dtype=np.float32),
        'sensor_pos_y': np.array(all_y, dtype=np.float32),
        'sensor_pos_z': np.array(all_z, dtype=np.float32),
        't': np.array(all_t, dtype=np.float32),
        'charge': np.array(all_charge, dtype=np.float32),
        'string_id': np.array(all_string_id, dtype=np.uint32),
        'sensor_id': np.array(all_sensor_id, dtype=np.uint32),
        'id_idx': np.array(all_id_idx, dtype=np.uint64),
    }

def count_total_events(i3_files: List[str]) -> int:
    """Count the total number of physics events across all i3 files."""
    total = 0
    for file_path in i3_files:
        i3_file = dataio.I3File(file_path)
        while i3_file.more():
            frame = i3_file.pop_physics()
            if frame and frame.Has("I3EventHeader") and frame["I3EventHeader"].sub_event_stream != "NullSplit":
                total += 1
        i3_file.close()
    return total

def parse_mc_truth(frame: icetray.I3Frame) -> Dict[str, Any]:
    """Parse MC truth information from an I3Frame."""
    if "I3MCTree" not in frame:
        return {}

    mc_tree = frame["I3MCTree"]
    
    if not hasattr(mc_tree, 'primaries') or len(mc_tree.primaries) == 0:
        return {}
    
    primary = mc_tree.primaries[0]
    
    # Find final state lepton
    lepton_types = {
        dataclasses.I3Particle.ParticleType.EMinus,
        dataclasses.I3Particle.ParticleType.EPlus,
        dataclasses.I3Particle.ParticleType.MuMinus,
        dataclasses.I3Particle.ParticleType.MuPlus,
        dataclasses.I3Particle.ParticleType.TauMinus,
        dataclasses.I3Particle.ParticleType.TauPlus
    }
    
    final_lepton = None
    for particle in mc_tree:
        if particle.type in lepton_types and particle.id != primary.id:
            final_lepton = particle
            break
    
    parsed = {
        'initial_energy': primary.energy,
        'initial_zenith': primary.dir.zenith,
        'initial_azimuth': primary.dir.azimuth,
        'initial_x': primary.pos.x,
        'initial_y': primary.pos.y,
        'initial_z': primary.pos.z,
        'initial_type': _PDGMAP.get(primary.type),
    }
    
    # Store final lepton as arrays (EventRecord expects arrays, not scalars)
    if final_lepton:
        parsed['final_energy'] = [final_lepton.energy]
        parsed['final_zenith'] = [final_lepton.dir.zenith]
        parsed['final_azimuth'] = [final_lepton.dir.azimuth]
        parsed['final_x'] = [final_lepton.pos.x]
        parsed['final_y'] = [final_lepton.pos.y]
        parsed['final_z'] = [final_lepton.pos.z]
        parsed['final_type'] = [_PDGMAP.get(final_lepton.type)]

    return parsed

def convert_icecube_to_mmap(input_path: str, output_path: str,
                               file_range: str = None, pulse_key: str = "SplitInIceDSTPulses") -> Tuple[int, int]:
    """Convert IceCube i3 files to memory-mapped format."""
    
    # Find and filter input files
    i3_files = find_i3_files(input_path)
    print(f"Found {len(i3_files)} i3 files")
    
    if file_range:
        start, end = map(int, file_range.split('-'))
        i3_files = i3_files[start:end]
        print(f"Processing files from index {start} to {end}")
    
    # Load geometry and count events
    gcd_file = os.path.join(os.path.dirname(__file__), '..', 'resources', 'GeoCalibDetectorStatus_IC86.AVG_Pass2_SF0.99.i3')
    geometry = load_geometry(gcd_file)
    
    total_events = count_total_events(i3_files)
    print(f"Converting {total_events} events from {len(i3_files)} files...")
    
    # Create memory-mapped files
    from core.mmap_format import create_mmap_files_with_headers, create_index_mmap_with_header
    
    idx_path, data_file_path = create_mmap_files_with_headers(output_path, total_events)
    index_mmap = create_index_mmap_with_header(idx_path, total_events)
    
    # Convert events
    event_idx = 0
    total_photons = 0
    current_photon_idx = 0
    
    with open(data_file_path, 'ab') as data_file:
        for frame in iter_i3_events(i3_files):
            
            # Create event record from MC truth
            mc_truth = parse_mc_truth(frame)
            event_record = EventRecord.from_dict(mc_truth)
            
            # Process photons
            photons = parse_pulses(frame, pulse_key, geometry)
            photon_array = PhotonHit.from_dict(photons)
            num_photons = len(photon_array)
            
            # Set photon indexing
            event_record['photon_start_idx'] = current_photon_idx
            event_record['photon_end_idx'] = current_photon_idx + num_photons
            
            # Store event record
            index_mmap[event_idx] = event_record
            
            # Write photons to data file
            if num_photons > 0:
                data_file.write(photon_array.tobytes())
                current_photon_idx += num_photons
                total_photons += num_photons
            
            event_idx += 1
            
            if event_idx % 1000 == 0:
                print(f"Processed {event_idx:,} events, {total_photons:,} photons")
    
    # Finalize
    index_mmap.flush()
    
    print(f"Conversion complete: {event_idx} events, {total_photons:,} total photons")
    print(f"Output files: {output_path}.idx, {output_path}.dat")
    
    return event_idx, total_photons