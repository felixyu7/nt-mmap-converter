"""
IceCube neutrino telescope data parser.
Handles conversion from i3 files to memory-mapped format.
"""

import os
import glob
from typing import List, Iterator, Tuple, Dict, Any, Optional, Set
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
    dataclasses.I3Particle.ParticleType.PiMinus: -211,
    dataclasses.I3Particle.ParticleType.Hadrons: 99  # Special IceCube hadron shower
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

def iter_i3_events(i3_files: List[str],
                   allowed_streams: Optional[Set[str]] = None) -> Iterator[icetray.I3Frame]:
    """Iterate over physics frames, gating by sub-event stream if requested."""
    for path in i3_files:
        label = os.path.basename(path)
        print(f"Processing {label}...")

        i3_file = dataio.I3File(path)
        try:
            while i3_file.more():
                frame = i3_file.pop_physics()
                if not frame or not frame.Has("I3EventHeader"):
                    continue

                stream = frame["I3EventHeader"].sub_event_stream

                if allowed_streams is None:
                    if stream == "NullSplit":
                        continue
                elif stream not in allowed_streams:
                    continue

                yield frame
        finally:
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
    
    if hasattr(pulses, 'items'):
        pulse_iter = pulses.items()
    else:
        pulse_iter = pulses

    for entry in pulse_iter:
        if isinstance(entry, tuple):
            if not entry:
                continue
            omkey = entry[0]
            reco_pulses = entry[1] if len(entry) > 1 else pulses[omkey]
        else:
            omkey = entry
            if hasattr(pulses, '__getitem__'):
                reco_pulses = pulses[omkey]
            else:
                continue
        
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


def parse_mc_truth(frame: icetray.I3Frame) -> Dict[str, Any]:
    """Parse MC truth information from an I3Frame."""
    parsed = {}
    
    # Try to get MC truth from I3MCTree_preMuonProp first (preferred for hadrons)
    mc_tree_key = "I3MCTree_preMuonProp" if "I3MCTree_preMuonProp" in frame else "I3MCTree"
    
    if mc_tree_key not in frame:
        return {}

    mc_tree = frame[mc_tree_key]
    
    if not hasattr(mc_tree, 'primaries') or len(mc_tree.primaries) == 0:
        return {}
    
    primary = mc_tree.primaries[0]
    
    # Fill basic MC truth
    parsed.update({
        'initial_energy': primary.energy,
        'initial_zenith': primary.dir.zenith,
        'initial_azimuth': primary.dir.azimuth,
        'initial_x': primary.pos.x,
        'initial_y': primary.pos.y,
        'initial_z': primary.pos.z,
        'initial_type': _PDGMAP.get(primary.type),
    })
    
    # Initialize final state arrays (lepton + hadrons)
    final_energy = [0.0] * 2
    final_type = [0] * 2
    final_zenith = [0.0] * 2
    final_azimuth = [0.0] * 2
    final_x = [0.0] * 2
    final_y = [0.0] * 2
    final_z = [0.0] * 2
    
    # Look for final state particles
    lepton_types = {
        dataclasses.I3Particle.ParticleType.EMinus,
        dataclasses.I3Particle.ParticleType.EPlus,
        dataclasses.I3Particle.ParticleType.MuMinus,
        dataclasses.I3Particle.ParticleType.MuPlus,
        dataclasses.I3Particle.ParticleType.TauMinus,
        dataclasses.I3Particle.ParticleType.TauPlus
    }
    
    final_lepton = None
    final_hadrons = None
    
    for particle in mc_tree:
        # Skip primary
        if particle.id == primary.id:
            continue
            
        # Look for final state lepton
        if particle.type in lepton_types and final_lepton is None:
            final_lepton = particle
            
        # Look for hadrons (special IceCube particle type)
        if particle.type == dataclasses.I3Particle.ParticleType.Hadrons and final_hadrons is None:
            final_hadrons = particle

    # Determine interaction type (CC/NC) using FIRST child of the primary
    def _ptype_name(pt):
        name = getattr(pt, 'name', str(pt))
        if '.' in name:
            name = name.split('.')[-1]
        return name

    def _neutrino_flavor(pt):
        n = _ptype_name(pt)
        if n.endswith('Bar'):
            n = n[:-3]
        return n if n in ('NuE', 'NuMu', 'NuTau') else None

    def _lepton_family(pt):
        n = _ptype_name(pt)
        if n.startswith('E'):
            return 'E'
        if n.startswith('Mu'):
            return 'Mu'
        if n.startswith('Tau'):
            return 'Tau'
        return None

    primary_flavor = _neutrino_flavor(primary.type)

    if hasattr(mc_tree, 'children'):
        children = list(mc_tree.children(primary))
    elif hasattr(mc_tree, 'get_daughters'):
        children = list(mc_tree.get_daughters(primary))
    else:
        children = []

    cc_nc = None
    base_name = _ptype_name(primary.type)
    if base_name.endswith('Bar') and base_name.startswith('Nu'):
        base_name = base_name[:-3]

    first_child = children[0] if children else None
    if primary_flavor is not None and first_child is not None:
        fam = _lepton_family(first_child.type)
        if ((primary_flavor == 'NuE' and fam == 'E') or
            (primary_flavor == 'NuMu' and fam == 'Mu') or
            (primary_flavor == 'NuTau' and fam == 'Tau')):
            cc_nc = 'CC'
        else:
            child_flavor = _neutrino_flavor(first_child.type)
            if child_flavor == primary_flavor:
                cc_nc = 'NC'

    if cc_nc is None and primary_flavor is not None:
        lepton_types = {
            dataclasses.I3Particle.ParticleType.EMinus,
            dataclasses.I3Particle.ParticleType.EPlus,
            dataclasses.I3Particle.ParticleType.MuMinus,
            dataclasses.I3Particle.ParticleType.MuPlus,
            dataclasses.I3Particle.ParticleType.TauMinus,
            dataclasses.I3Particle.ParticleType.TauPlus
        }
        found_lepton = any((p.type in lepton_types) and (p.id != primary.id) for p in mc_tree)
        if found_lepton:
            cc_nc = 'CC'
        else:
            for p in mc_tree:
                if p.id == primary.id:
                    continue
                if _neutrino_flavor(p.type) == primary_flavor:
                    cc_nc = 'NC'
                    break

    interaction_str = f"{base_name}_{cc_nc}" if (cc_nc and base_name.startswith('Nu')) else base_name
    parsed['interaction'] = interaction_str
    
    # Store final lepton at index 0
    if final_lepton:
        final_energy[0] = final_lepton.energy
        final_type[0] = _PDGMAP.get(final_lepton.type, 0)
        final_zenith[0] = final_lepton.dir.zenith
        final_azimuth[0] = final_lepton.dir.azimuth
        final_x[0] = final_lepton.pos.x
        final_y[0] = final_lepton.pos.y
        final_z[0] = final_lepton.pos.z
    
    # Store hadrons at index 1
    if final_hadrons:
        final_energy[1] = final_hadrons.energy
        final_type[1] = _PDGMAP.get(final_hadrons.type, 0)
        final_zenith[1] = final_hadrons.dir.zenith
        final_azimuth[1] = final_hadrons.dir.azimuth
        final_x[1] = final_hadrons.pos.x
        final_y[1] = final_hadrons.pos.y
        final_z[1] = final_hadrons.pos.z
    
    # Add final state arrays to parsed data
    parsed.update({
        'final_energy': final_energy,
        'final_type': final_type,
        'final_zenith': final_zenith,
        'final_azimuth': final_azimuth,
        'final_x': final_x,
        'final_y': final_y,
        'final_z': final_z,
    })
    
    # Add IceCube-specific fields
    if "Homogenized_QTot" in frame:
        parsed['homogenized_qtot'] = float(frame["Homogenized_QTot"].value)

    return parsed

def frame_passes_filters(frame: icetray.I3Frame, filter_names: Optional[Set[str]]) -> bool:
    """Return True if the frame passes at least one requested filter (condition only)."""
    if not filter_names:
        return True
    if "FilterMask" not in frame:
        return False

    for filter_name, result in frame["FilterMask"].items():
        if filter_name in filter_names and bool(getattr(result, 'condition_passed', False)):
            return True
    return False


def convert_icecube_to_mmap(input_path: str, output_path: str,
                               file_range: str = None, pulse_key: str = "SplitInIceDSTPulses",
                               filter_names: Optional[List[str]] = None,
                               subevent_streams: Optional[List[str]] = None) -> Tuple[int, int]:
    """Convert IceCube i3 files to memory-mapped format using streaming approach.

    Args:
        input_path: Directory containing i3/i3.zst files.
        output_path: Base path for emitted mmap artifacts.
        file_range: Optional "start-end" slice of discovered files.
        pulse_key: Name of the pulse series to extract.
        filter_names: Optional list of FilterMask names; keep events if any condition passes.
    """
    
    # Find and filter input files
    i3_files = find_i3_files(input_path)
    print(f"Found {len(i3_files)} i3 files")
    
    if file_range:
        start, end = map(int, file_range.split('-'))
        i3_files = i3_files[start:end]
        print(f"Processing files from index {start} to {end}")
    
    # Load geometry
    gcd_file = os.path.join(os.path.dirname(__file__), '..', 'resources', 'GeoCalibDetectorStatus_IC86.AVG_Pass2_SF0.99.i3')
    geometry = load_geometry(gcd_file)
    
    print(f"Converting events from {len(i3_files)} files using streaming approach...")

    filter_lookup: Optional[Set[str]] = None
    if filter_names:
        # Preserve user order for logging but use set for membership tests
        ordered_filters = list(dict.fromkeys(filter_names))
        print(f"Applying IceCube filter conditions: {', '.join(ordered_filters)}")
        filter_lookup = set(ordered_filters)

    stream_lookup: Optional[Set[str]] = None
    if subevent_streams:
        ordered_streams = list(dict.fromkeys(subevent_streams))
        print(f"Limiting to sub-event streams: {', '.join(ordered_streams)}")
        stream_lookup = set(ordered_streams)
    
    # Create streaming memory-mapped files
    from core.mmap_format import create_streaming_mmap_files, StreamingIndexWriter, append_photons_to_file
    
    # Estimate events per file for initial allocation
    events_per_file_estimate = 1000  # Conservative estimate
    initial_estimate = len(i3_files) * events_per_file_estimate
    
    idx_path, data_file_path = create_streaming_mmap_files(output_path, initial_estimate, source_type='icecube')
    index_writer = StreamingIndexWriter(idx_path, initial_estimate)
    
    # Convert events
    total_photons = 0
    current_photon_idx = 0
    
    for frame in iter_i3_events(i3_files, stream_lookup):
        if not frame_passes_filters(frame, filter_lookup):
            continue

        # Create event record from MC truth
        mc_truth = parse_mc_truth(frame)
        
        # Process photons
        photons = parse_pulses(frame, pulse_key, geometry)
        photon_array = PhotonHit.from_dict(photons)
        num_photons = len(photon_array)
        
        # Skip events with no photons - they're not useful for ML training
        if num_photons == 0:
            continue
            
        # Compute hit statistics
        mc_truth['num_hits'] = num_photons
        # Count unique OMKeys (string_id, sensor_id pairs) for IceCube
        omkey_pairs = np.column_stack([photons['string_id'], photons['sensor_id']])
        unique_omkeys = np.unique(omkey_pairs, axis=0)
        mc_truth['num_chans'] = len(unique_omkeys)
        
        # Create event record using IceCube-specific dtype
        event_record = EventRecord.from_dict(mc_truth, source_type='icecube')
        
        # Set photon indexing
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
