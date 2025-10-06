"""
IceCube neutrino telescope data parser.
Handles conversion from i3 files to memory-mapped format.
"""

import glob
import os
from typing import List, Iterator, Tuple, Dict, Any, Optional, Set

import icecube
import numpy as np
from icecube import dataio, dataclasses, icetray

from core.mmap_format import EventRecord, PhotonHit

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
    mc_tree = None
    if "I3MCTree_preMuonProp" in frame:
        mc_tree = frame["I3MCTree_preMuonProp"]
    elif "I3MCTree" in frame:
        mc_tree = frame["I3MCTree"]

    tree_primary = None
    if mc_tree and hasattr(mc_tree, "primaries") and mc_tree.primaries:
        tree_primary = mc_tree.primaries[0]

    primary = frame["PolyplopiaPrimary"] if "PolyplopiaPrimary" in frame else tree_primary
    if primary is None:
        return {}

    parsed = {
        'initial_energy': primary.energy,
        'initial_zenith': primary.dir.zenith,
        'initial_azimuth': primary.dir.azimuth,
        'initial_x': primary.pos.x,
        'initial_y': primary.pos.y,
        'initial_z': primary.pos.z,
        'initial_type': int(getattr(primary, "pdg_encoding", 0) or 0),
    }

    final_energy = [0.0, 0.0]
    final_type = [0, 0]
    final_zenith = [0.0, 0.0]
    final_azimuth = [0.0, 0.0]
    final_x = [0.0, 0.0]
    final_y = [0.0, 0.0]
    final_z = [0.0, 0.0]

    lepton_codes = {11, 13, 15}
    final_lepton = None
    final_hadrons = None

    skip_id = None
    if tree_primary is not None and hasattr(tree_primary, "id"):
        skip_id = tree_primary.id
    elif hasattr(primary, "id"):
        skip_id = primary.id

    if mc_tree:
        for particle in mc_tree:
            if skip_id is not None and getattr(particle, "id", None) == skip_id:
                continue
            pdg_code = int(getattr(particle, "pdg_encoding", 0) or 0)
            if final_lepton is None and abs(pdg_code) in lepton_codes:
                final_lepton = particle
            if final_hadrons is None and particle.type == dataclasses.I3Particle.ParticleType.Hadrons:
                final_hadrons = particle

    def neutrino_flavor(pdg_code: int) -> Optional[str]:
        return {12: 'NuE', 14: 'NuMu', 16: 'NuTau'}.get(abs(pdg_code))

    def charged_lepton_family(pdg_code: int) -> Optional[str]:
        return {11: 'E', 13: 'Mu', 15: 'Tau'}.get(abs(pdg_code))

    primary_pdg = int(getattr(primary, "pdg_encoding", 0) or 0)
    primary_flavor = neutrino_flavor(primary_pdg)

    children: List[dataclasses.I3Particle] = []
    if mc_tree and tree_primary:
        if hasattr(mc_tree, "children"):
            children = list(mc_tree.children(tree_primary))
        elif hasattr(mc_tree, "get_daughters"):
            children = list(mc_tree.get_daughters(tree_primary))

    cc_nc = None
    base_name = primary.type.name if hasattr(primary.type, "name") else str(primary.type)
    if base_name.endswith('Bar') and base_name.startswith('Nu'):
        base_name = base_name[:-3]

    first_child = children[0] if children else None
    if primary_flavor and first_child is not None:
        child_pdg = int(getattr(first_child, "pdg_encoding", 0) or 0)
        family = charged_lepton_family(child_pdg)
        expected_family = primary_flavor.replace('Nu', '')
        if family and family == expected_family:
            cc_nc = 'CC'
        elif neutrino_flavor(child_pdg) == primary_flavor:
            cc_nc = 'NC'

    if cc_nc is None and primary_flavor and mc_tree:
        found_lepton = any(
            abs(int(getattr(p, "pdg_encoding", 0) or 0)) in lepton_codes
            and (skip_id is None or getattr(p, "id", None) != skip_id)
            for p in mc_tree
        )
        if found_lepton:
            cc_nc = 'CC'
        else:
            for p in mc_tree:
                if skip_id is not None and getattr(p, "id", None) == skip_id:
                    continue
                if neutrino_flavor(int(getattr(p, "pdg_encoding", 0) or 0)) == primary_flavor:
                    cc_nc = 'NC'
                    break

    interaction = f"{base_name}_{cc_nc}" if (cc_nc and base_name.startswith('Nu')) else base_name
    parsed['interaction'] = interaction

    if final_lepton:
        final_energy[0] = final_lepton.energy
        final_type[0] = int(getattr(final_lepton, "pdg_encoding", 0) or 0)
        final_zenith[0] = final_lepton.dir.zenith
        final_azimuth[0] = final_lepton.dir.azimuth
        final_x[0] = final_lepton.pos.x
        final_y[0] = final_lepton.pos.y
        final_z[0] = final_lepton.pos.z

    if final_hadrons:
        final_energy[1] = final_hadrons.energy
        final_type[1] = int(getattr(final_hadrons, "pdg_encoding", 0) or 0)
        final_zenith[1] = final_hadrons.dir.zenith
        final_azimuth[1] = final_hadrons.dir.azimuth
        final_x[1] = final_hadrons.pos.x
        final_y[1] = final_hadrons.pos.y
        final_z[1] = final_hadrons.pos.z

    parsed.update({
        'final_energy': final_energy,
        'final_type': final_type,
        'final_zenith': final_zenith,
        'final_azimuth': final_azimuth,
        'final_x': final_x,
        'final_y': final_y,
        'final_z': final_z,
    })

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
