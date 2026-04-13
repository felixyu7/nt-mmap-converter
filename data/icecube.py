"""
IceCube neutrino telescope data parser.
Handles conversion from i3 files to memory-mapped format.
Uses I3EventLabeler from icetray ml_suite for event classification.
"""

import glob
import os
from typing import List, Iterator, Tuple, Dict, Any, Optional, Set, Sequence
from enum import IntEnum

import numpy as np
from icecube import dataio, dataclasses, icetray, phys_services

from core.mmap_format import EventRecord, PhotonHit

# Morphology classes (simplified, 6 classes)
class Morphology(IntEnum):
    CASCADE = 0          # Shower-like events (neutrino NC, NuE CC, tau decays to e/hadrons)
    STARTING_TRACK = 1   # Track born inside detector, exits
    THROUGHGOING_TRACK = 2  # Track enters from outside and exits
    STOPPING_TRACK = 3   # Track enters and stops/decays inside
    UNCONTAINED = 4      # No detector primary (misses/skims detector)
    BUNDLE = 5           # Multiple detector primaries (muon bundles)


# Detailed event classes (34 classes, matching I3EventLabeler)
class EventClass(IntEnum):
    # Uncontained events (0-5) -> Morphology.UNCONTAINED
    BACKGROUND = 0               # No particle within search radius
    UNCONTAINED_CASCADE = 1      # Cascade outside detector
    SKIMMING_TRACK = 2           # Single muon skims past detector
    SKIMMING_BUNDLE = 3          # Multiple muons skim past
    SKIMMING_SPUR = 4            # Tau skims past
    OTHER_UNCONTAINED = 5        # Other uncontained

    # Multiple primaries (6-7) -> Morphology.BUNDLE
    BUNDLE = 6                   # All primaries are muons
    OTHER_MULTIPLE = 7           # Mixed particle types

    # NC interactions (8) -> Morphology.CASCADE
    HADR_CASCADE = 8             # All neutrino NC interactions

    # NuE CC (9) -> Morphology.CASCADE
    EM_HADR_CASCADE = 9          # NuE CC interaction

    # Glashow resonance (10-18)
    GLASHOW_ELECTRON = 10        # W -> electron -> CASCADE
    GLASHOW_HADR = 11            # W -> hadrons -> CASCADE
    GLASHOW_STARTING_TRACK = 12  # W -> muon, exits -> STARTING_TRACK
    GLASHOW_CONTAINED_TRACK = 13 # W -> muon, stops -> STOPPING_TRACK
    GLASHOW_STARTING_SPUR = 14   # W -> tau, exits -> STARTING_TRACK
    GLASHOW_LOLLIPOP_HADR = 15   # W -> tau -> hadrons -> CASCADE
    GLASHOW_LOLLIPOP_EM = 16     # W -> tau -> electron -> CASCADE
    GLASHOW_TAU_STARTING_TRACK = 17   # W -> tau -> muon, exits -> STARTING_TRACK
    GLASHOW_TAU_CONTAINED_TRACK = 18  # W -> tau -> muon, stops -> STOPPING_TRACK

    # NuMu CC (19-20)
    STARTING_TRACK = 19          # Muon exits -> STARTING_TRACK
    CONTAINED_TRACK = 20         # Muon stops -> STOPPING_TRACK

    # NuTau CC (21-25)
    INVERTED_LOLLIPOP = 21             # Tau exits -> STARTING_TRACK
    DOUBLE_BANG_EM = 22                # Tau -> electron -> CASCADE
    DOUBLE_BANG_HADR = 23              # Tau -> hadrons -> CASCADE
    INVERTED_LOLLIPOP_STARTING_TRACK = 24  # Tau -> muon, exits -> STARTING_TRACK
    INVERTED_LOLLIPOP_CONTAINED_TRACK = 25 # Tau -> muon, stops -> STOPPING_TRACK

    # Muon entering detector (26-27)
    THROUGHGOING_TRACK = 26      # Muon traverses -> THROUGHGOING_TRACK
    STOPPING_TRACK = 27          # Muon stops -> STOPPING_TRACK

    # Tau entering detector (28-32)
    THROUGHGOING_SPUR = 28       # Tau traverses -> THROUGHGOING_TRACK
    LOLLIPOP_EM = 29             # Tau -> electron -> CASCADE
    LOLLIPOP_HADR = 30           # Tau -> hadrons -> CASCADE
    TAU_STARTING_TRACK = 31      # Tau -> muon, exits -> STARTING_TRACK
    TAU_CONTAINED_TRACK = 32     # Tau -> muon, stops -> STOPPING_TRACK

    # Other
    OTHER = 33                   # Edge cases -> UNCONTAINED


# Map detailed event classes to simplified morphologies
DETAILED_TO_MORPHOLOGY = {
    # Uncontained -> UNCONTAINED
    EventClass.BACKGROUND: Morphology.UNCONTAINED,
    EventClass.UNCONTAINED_CASCADE: Morphology.UNCONTAINED,
    EventClass.SKIMMING_TRACK: Morphology.UNCONTAINED,
    EventClass.SKIMMING_BUNDLE: Morphology.UNCONTAINED,
    EventClass.SKIMMING_SPUR: Morphology.UNCONTAINED,
    EventClass.OTHER_UNCONTAINED: Morphology.UNCONTAINED,

    # Bundle -> BUNDLE
    EventClass.BUNDLE: Morphology.BUNDLE,
    EventClass.OTHER_MULTIPLE: Morphology.BUNDLE,

    # Cascades -> CASCADE
    EventClass.HADR_CASCADE: Morphology.CASCADE,
    EventClass.EM_HADR_CASCADE: Morphology.CASCADE,
    EventClass.GLASHOW_ELECTRON: Morphology.CASCADE,
    EventClass.GLASHOW_HADR: Morphology.CASCADE,
    EventClass.DOUBLE_BANG_EM: Morphology.CASCADE,
    EventClass.DOUBLE_BANG_HADR: Morphology.CASCADE,
    EventClass.LOLLIPOP_EM: Morphology.CASCADE,
    EventClass.LOLLIPOP_HADR: Morphology.CASCADE,
    EventClass.GLASHOW_LOLLIPOP_EM: Morphology.CASCADE,
    EventClass.GLASHOW_LOLLIPOP_HADR: Morphology.CASCADE,

    # Starting tracks -> STARTING_TRACK
    EventClass.STARTING_TRACK: Morphology.STARTING_TRACK,
    EventClass.GLASHOW_STARTING_TRACK: Morphology.STARTING_TRACK,
    EventClass.GLASHOW_STARTING_SPUR: Morphology.STARTING_TRACK,
    EventClass.INVERTED_LOLLIPOP: Morphology.STARTING_TRACK,
    EventClass.INVERTED_LOLLIPOP_STARTING_TRACK: Morphology.STARTING_TRACK,
    EventClass.TAU_STARTING_TRACK: Morphology.STARTING_TRACK,
    EventClass.GLASHOW_TAU_STARTING_TRACK: Morphology.STARTING_TRACK,

    # Throughgoing -> THROUGHGOING_TRACK
    EventClass.THROUGHGOING_TRACK: Morphology.THROUGHGOING_TRACK,
    EventClass.THROUGHGOING_SPUR: Morphology.THROUGHGOING_TRACK,

    # Stopping/contained -> STOPPING_TRACK
    EventClass.STOPPING_TRACK: Morphology.STOPPING_TRACK,
    EventClass.CONTAINED_TRACK: Morphology.STOPPING_TRACK,
    EventClass.GLASHOW_CONTAINED_TRACK: Morphology.STOPPING_TRACK,
    EventClass.INVERTED_LOLLIPOP_CONTAINED_TRACK: Morphology.STOPPING_TRACK,
    EventClass.TAU_CONTAINED_TRACK: Morphology.STOPPING_TRACK,
    EventClass.GLASHOW_TAU_CONTAINED_TRACK: Morphology.STOPPING_TRACK,

    # Other -> UNCONTAINED
    EventClass.OTHER: Morphology.UNCONTAINED,
}


def compute_morphology_labels(frame: icetray.I3Frame) -> Tuple[int, int]:
    """
    Extract (event_class, morphology) from I3EventLabeler's "EventLabels" in frame.
    Returns (0, 0) for frames without EventLabels (e.g., CORSIKA without I3MCTree).
    """
    if "EventLabels" not in frame:
        # No MC truth available (common for CORSIKA Level2 files) - use 0s
        return 0, 0

    labels = frame["EventLabels"]
    event_class = int(labels.get("classification", EventClass.OTHER))
    morphology = DETAILED_TO_MORPHOLOGY.get(event_class, Morphology.UNCONTAINED)
    return event_class, int(morphology)

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
                   gcd_file: str,
                   allowed_streams: Optional[Set[str]] = None,
                   sig_padding: float = 50.0,
                   bg_padding: float = 150.0) -> Iterator[Tuple[icetray.I3Frame, str]]:
    """
    Iterate over physics frames, running I3EventLabeler when I3MCTree is available.

    For frames without I3MCTree (common in CORSIKA Level2), EventLabels is not added
    and compute_morphology_labels will return defaults (0, 0).
    """
    from icecube import ml_suite  # noqa: F401 - loads the C++ module
    from icecube.simclasses import I3ParticleIDMap

    for path in i3_files:
        label = os.path.basename(path)
        print(f"Processing {label}...")

        collected_frames = []
        stats = {'with_mctree': 0, 'without_mctree': 0}

        def make_preparer(stats_dict):
            def prepare_and_count(frame):
                if "I3MCTree" in frame:
                    stats_dict['with_mctree'] += 1
                    if "I3MCPESeriesMapParticleIDMap" not in frame:
                        frame["I3MCPESeriesMapParticleIDMap"] = I3ParticleIDMap()
                else:
                    stats_dict['without_mctree'] += 1
                return True
            return prepare_and_count

        def make_collector(frame_list):
            def collector(frame):
                frame_list.append(frame)
            return collector

        # Single pass: conditionally run I3EventLabeler only on frames with I3MCTree
        tray = icetray.I3Tray()
        tray.Add("I3Reader", FilenameList=[gcd_file, path])
        tray.Add(make_preparer(stats), Streams=[icetray.I3Frame.Physics])
        tray.Add("I3EventLabeler",
                 Name="EventLabels",
                 gcd=gcd_file,
                 sig_padding=sig_padding,
                 bg_padding=bg_padding,
                 If=lambda frame: "I3MCTree" in frame)
        tray.Add(make_collector(collected_frames), Streams=[icetray.I3Frame.Physics])
        tray.Execute()

        print(f"  Frames with MCTree: {stats['with_mctree']}, without: {stats['without_mctree']}")
        print(f"  Collected {len(collected_frames)} frames from {label}")

        for frame in collected_frames:
            if not frame.Has("I3EventHeader"):
                continue
            stream = frame["I3EventHeader"].sub_event_stream
            if allowed_streams is None:
                if stream == "NullSplit":
                    continue
            elif stream not in allowed_streams:
                continue
            yield frame, path

def parse_pulses(frame: icetray.I3Frame, pulse_key: str, geometry: dataclasses.I3Geometry) -> Dict[str, np.ndarray]:
    """Parse pulse data from an I3Frame."""
    if pulse_key not in frame:
        return {
            'sensor_pos_x': np.array([]), 'sensor_pos_y': np.array([]),
            'sensor_pos_z': np.array([]), 't': np.array([]),
            'charge': np.array([]), 'string_id': np.array([]),
            'sensor_id': np.array([])
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
            'sensor_id': np.array([])
        }
    
    all_x, all_y, all_z, all_t, all_charge, all_string_id, all_sensor_id = [], [], [], [], [], [], []
    
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
            
    return {
        'sensor_pos_x': np.array(all_x, dtype=np.float32),
        'sensor_pos_y': np.array(all_y, dtype=np.float32),
        'sensor_pos_z': np.array(all_z, dtype=np.float32),
        't': np.array(all_t, dtype=np.float32),
        'charge': np.array(all_charge, dtype=np.float32),
        'string_id': np.array(all_string_id, dtype=np.uint32),
        'sensor_id': np.array(all_sensor_id, dtype=np.uint32),
    }


def parse_mc_truth(frame: icetray.I3Frame) -> Dict[str, Any]:
    """Parse MC truth information from an I3Frame (labels from I3EventLabeler)."""
    mc_tree = None
    if "I3MCTree" in frame:
        mc_tree = frame["I3MCTree"]
    elif "I3MCTree_preMuonProp" in frame:
        mc_tree = frame["I3MCTree_preMuonProp"]

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
            if final_lepton is not None and final_hadrons is not None:
                break

    # Get labels from I3EventLabeler (attached to frame by iter_i3_events)
    event_class, morphology = compute_morphology_labels(frame)
    parsed['event_class'] = event_class
    parsed['morphology'] = morphology

    # Vertex position: use interaction vertex (first_vertex) for events where the
    # detector primary is a neutrino (classes 8-25), otherwise use detector entry
    # point (classes 0-7 uncontained/bundles, 26-32 muon/tau detector primaries).
    labels = frame["EventLabels"] if "EventLabels" in frame else {}
    if 8 <= event_class <= 25:
        parsed['vertex_x'] = float(labels.get("first_vertex_x", 0.0))
        parsed['vertex_y'] = float(labels.get("first_vertex_y", 0.0))
        parsed['vertex_z'] = float(labels.get("first_vertex_z", 0.0))
    else:
        parsed['vertex_x'] = float(labels.get("detector_entry_x", 0.0))
        parsed['vertex_y'] = float(labels.get("detector_entry_y", 0.0))
        parsed['vertex_z'] = float(labels.get("detector_entry_z", 0.0))

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

    # Note: 'starting' is now implicit in morphology (CASCADE=0 or STARTING_TRACK=1)

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


def convert_icecube_to_mmap(input_paths: Sequence[str], output_path: str,
                               file_range: str = None, pulse_key: str = "SplitInIceDSTPulses",
                               filter_names: Optional[List[str]] = None,
                               subevent_streams: Optional[List[str]] = None) -> Tuple[int, int]:
    """Convert IceCube i3 files to memory-mapped format using streaming approach.

    Args:
        input_paths: One or more directories containing i3/i3.zst files.
        output_path: Base path for emitted mmap artifacts.
        file_range: Optional "start-end" slice of discovered files.
        pulse_key: Name of the pulse series to extract.
        filter_names: Optional list of FilterMask names; keep events if any condition passes.
    """
    
    # Find and filter input files
    search_paths: List[str] = [input_paths] if isinstance(input_paths, str) else list(input_paths)
    i3_files: List[str] = []
    for path in search_paths:
        i3_files.extend(find_i3_files(path))

    print(f"Found {len(i3_files)} i3 files from {len(search_paths)} director{'y' if len(search_paths) == 1 else 'ies'}")
    
    if file_range:
        start, end = map(int, file_range.split('-'))
        i3_files = i3_files[start:end]
        print(f"Processing files from index {start} to {end}")
    
    # Load geometry for pulse parsing
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
    
    for frame, source_path in iter_i3_events(i3_files, gcd_file, stream_lookup):
        if not frame_passes_filters(frame, filter_lookup):
            continue

        # Create event record from MC truth (labels come from I3EventLabeler via frame)
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
