"""
IceCube neutrino telescope data parser.
Handles conversion from i3 files to memory-mapped format.
"""

import glob
import os
from contextlib import closing
from typing import List, Iterator, Tuple, Dict, Any, Optional, Set, Sequence

import icecube
import numpy as np
from icecube import dataio, dataclasses, icetray, phys_services

from core.mmap_format import EventRecord, PhotonHit

NEUTRINO_PDGS = {12, -12, 14, -14, 16, -16}
NEUTRINO_TYPES = {
    dataclasses.I3Particle.ParticleType.NuE,
    dataclasses.I3Particle.ParticleType.NuEBar,
    dataclasses.I3Particle.ParticleType.NuMu,
    dataclasses.I3Particle.ParticleType.NuMuBar,
    dataclasses.I3Particle.ParticleType.NuTau,
    dataclasses.I3Particle.ParticleType.NuTauBar,
}
LEPTON_TYPES = {
    dataclasses.I3Particle.ParticleType.EPlus,
    dataclasses.I3Particle.ParticleType.EMinus,
    dataclasses.I3Particle.ParticleType.MuPlus,
    dataclasses.I3Particle.ParticleType.MuMinus,
    dataclasses.I3Particle.ParticleType.TauPlus,
    dataclasses.I3Particle.ParticleType.TauMinus,
}
ELECTRON_TYPES = {
    dataclasses.I3Particle.ParticleType.EPlus,
    dataclasses.I3Particle.ParticleType.EMinus,
}

CLASSIFICATION_TO_MORPHOLOGY = {
    0: 0,   # Outside cascade
    1: 0,   # Cascade
    5: 0,   # Double bang
    6: 0,   # Stopping tau
    7: 0,   # Glashow cascade
    8: 0,   # Glashow track counted as cascade morphology per reference
    9: 0,   # Glashow tau
    12: 0,  # Uncontained tau
    2: 1,   # Through-going track
    3: 2,   # Starting track
    4: 3,   # Stopping track
    10: 3,  # Stop-start track (reserved)
    11: 4,  # Passing track
}

ATMOSPHERIC_INTERACTION = 0
CC_INTERACTION = 1
NC_INTERACTION = 2
GLASHOW_INTERACTION = 3


def _get_children(mc_tree: "dataclasses.I3MCTree",
                  particle: dataclasses.I3Particle) -> List[dataclasses.I3Particle]:
    if hasattr(mc_tree, "children"):
        return list(mc_tree.children(particle))
    if hasattr(mc_tree, "get_daughters"):
        return list(mc_tree.get_daughters(particle))
    return []


def _get_parent(mc_tree: "dataclasses.I3MCTree",
                particle: dataclasses.I3Particle) -> Optional[dataclasses.I3Particle]:
    if hasattr(mc_tree, "parent"):
        return mc_tree.parent(particle)
    if hasattr(mc_tree, "get_parent"):
        return mc_tree.get_parent(particle)
    return None


def _safe_intersection(surface: phys_services.ExtrudedPolygon,
                       position: dataclasses.I3Position,
                       direction: dataclasses.I3Direction) -> Optional[Tuple[float, float]]:
    try:
        result = surface.intersection(position, direction)
    except RuntimeError:
        return None

    first, second = float(result.first), float(result.second)
    if not (np.isfinite(first) and np.isfinite(second)):
        return None
    return first, second


def _is_track(particle: dataclasses.I3Particle) -> bool:
    return bool(getattr(particle, "is_track", False))


def _is_cascade(particle: dataclasses.I3Particle) -> bool:
    return bool(getattr(particle, "is_cascade", False))


def _has_signature(particle: dataclasses.I3Particle,
                   surface: phys_services.ExtrudedPolygon) -> int:
    if getattr(particle, "is_neutrino", False):
        return -1

    pos = getattr(particle, "pos", None)
    direction = getattr(particle, "dir", None)
    if pos is None or direction is None:
        return -1

    intersection = _safe_intersection(surface, pos, direction)
    if intersection is None:
        return -1

    first, second = intersection
    length = float(getattr(particle, "length", np.nan))

    if _is_cascade(particle):
        return 0 if first <= 0.0 and second > 0.0 else -1

    if _is_track(particle):
        if first <= 0.0 and second > 0.0:
            return 0
        if first > 0.0 and second > 0.0:
            if np.isfinite(length) and length <= first:
                return -1
            if (np.isfinite(length) and length > second) or not np.isfinite(length):
                return 1
            return 2

    return -1


def _find_particle(candidate: dataclasses.I3Particle,
                   mc_tree: "dataclasses.I3MCTree",
                   surface: phys_services.ExtrudedPolygon) -> List[dataclasses.I3Particle]:
    children = _get_children(mc_tree, candidate)
    if len(children) > 3:
        return []

    interacts_in_detector = any(
        (_has_signature(child, surface) != -1) and np.isfinite(getattr(child, "length", np.nan))
        for child in children
    )
    if interacts_in_detector:
        pdg_code = int(getattr(candidate, "pdg_encoding", 0) or 0)
        if abs(pdg_code) not in NEUTRINO_PDGS and not getattr(candidate, "is_neutrino", False):
            parent = _get_parent(mc_tree, candidate)
            return [parent] if parent is not None else []
        return [candidate]

    if len(children) < 3:
        found: List[dataclasses.I3Particle] = []
        for child in children:
            found.extend(_find_particle(child, mc_tree, surface))
        return found

    return []


def _locate_interaction_neutrino(mc_tree: Optional["dataclasses.I3MCTree"],
                                 surface: Optional[phys_services.ExtrudedPolygon]) -> Optional[dataclasses.I3Particle]:
    if mc_tree is None or surface is None:
        return None

    if hasattr(mc_tree, "primaries") and mc_tree.primaries:
        primaries = list(mc_tree.primaries)
    elif hasattr(mc_tree, "get_primaries"):
        primaries = list(mc_tree.get_primaries())
    else:
        primaries = [p for p in mc_tree]

    seed = None
    for particle in primaries:
        pdg_code = int(getattr(particle, "pdg_encoding", 0) or 0)
        if abs(pdg_code) in NEUTRINO_PDGS or getattr(particle, "is_neutrino", False):
            seed = particle
            break

    if seed is None and primaries:
        seed = primaries[0]

    if seed is None:
        return None

    found = _find_particle(seed, mc_tree, surface)
    if len(found) == 1:
        return found[0]
    return None


def compute_starting_flag(mc_tree: Optional["dataclasses.I3MCTree"],
                          surface: Optional[phys_services.ExtrudedPolygon]) -> bool:
    """Return True when the neutrino interaction point is inside the detector volume."""
    neutrino = _locate_interaction_neutrino(mc_tree, surface)
    if neutrino is None or surface is None:
        return False

    length = float(getattr(neutrino, "length", np.nan))
    base_pos = neutrino.pos
    if np.isfinite(length):
        direction = neutrino.dir
        interaction_pos = dataclasses.I3Position(
            base_pos.x + length * direction.x,
            base_pos.y + length * direction.y,
            base_pos.z + length * direction.z,
        )
    else:
        interaction_pos = base_pos

    intersection = _safe_intersection(surface, interaction_pos, neutrino.dir)
    if intersection is None:
        return False

    first, second = intersection
    return first <= 0.0 and second > 0.0


def _infer_interaction_type(frame: icetray.I3Frame) -> int:
    """Infer interaction type following the DeepIceLearning reference logic."""
    if frame.Has("I3MCWeightDict"):
        weights = frame["I3MCWeightDict"]
        if "InteractionType" in weights:
            return int(weights["InteractionType"])
        return ATMOSPHERIC_INTERACTION

    if frame.Has("EventProperties"):
        props = frame["EventProperties"]
        initial = props.initialType
        final1 = props.finalType1
        final2 = props.finalType2

        if initial in NEUTRINO_TYPES and final1 in LEPTON_TYPES:
            return CC_INTERACTION
        if initial in NEUTRINO_TYPES and final1 in NEUTRINO_TYPES and final2 not in ELECTRON_TYPES:
            return NC_INTERACTION
        if (initial == dataclasses.I3Particle.ParticleType.NuEBar and
                final1 in NEUTRINO_TYPES and final2 in ELECTRON_TYPES):
            return GLASHOW_INTERACTION
        return -1

    if frame.Has("I3CorsikaInfo") or frame.Has("CorsikaWeightMap"):
        return ATMOSPHERIC_INTERACTION

    return ATMOSPHERIC_INTERACTION


def _gather_detector_muons(particle: dataclasses.I3Particle,
                           mc_tree: "dataclasses.I3MCTree",
                           surface: phys_services.ExtrudedPolygon) -> List[dataclasses.I3Particle]:
    """Recursively collect muons that intersect the detector volume."""
    stack = [particle]
    muons: List[dataclasses.I3Particle] = []
    while stack:
        current = stack.pop()
        pdg = int(getattr(current, "pdg_encoding", 0) or 0)
        if abs(pdg) == 13 and _has_signature(current, surface) != -1:
            muons.append(current)

        for child in _get_children(mc_tree, current):
            stack.append(child)
    return muons


def _classify_corsika_event(mc_tree: Optional["dataclasses.I3MCTree"],
                            surface: Optional[phys_services.ExtrudedPolygon]) -> int:
    if mc_tree is None or surface is None:
        return 100

    if hasattr(mc_tree, "primaries") and mc_tree.primaries:
        primaries = list(mc_tree.primaries)
    elif hasattr(mc_tree, "get_primaries"):
        primaries = list(mc_tree.get_primaries())
    else:
        primaries = [particle for particle in mc_tree]

    per_primary_muons: List[List[dataclasses.I3Particle]] = []
    for primary in primaries:
        muons = _gather_detector_muons(primary, mc_tree, surface)
        if muons:
            per_primary_muons.append(muons)

    if not per_primary_muons:
        return 11  # Passing track with no clear in-ice muon

    if len(per_primary_muons) > 1:
        return 21  # Multiple coincident events

    muons = per_primary_muons[0]
    if len(muons) > 1:
        signatures = np.array([_has_signature(mu, surface) for mu in muons])
        if np.any(signatures == 1):
            return 22  # Through-going bundle
        return 23  # Stopping bundle

    signature = _has_signature(muons[0], surface)
    if signature == 2:
        return 4  # Stopping track
    return 2  # Reference treats all other signatures as through-going


def _classify_neutrino_event(mc_tree: Optional["dataclasses.I3MCTree"],
                             primary: Optional[dataclasses.I3Particle],
                             surface: Optional[phys_services.ExtrudedPolygon],
                             interaction_type: int) -> int:
    if mc_tree is None or primary is None or surface is None:
        return 100

    children = _get_children(mc_tree, primary)
    if not children:
        return 100

    pclass = 101
    particle_types = [abs(int(getattr(child, "pdg_encoding", 0) or 0)) for child in children]
    if interaction_type == CC_INTERACTION and 14 in particle_types:
        idx = particle_types.index(14)
        next_children = _get_children(mc_tree, children[idx])
        if next_children:
            children = next_children
            particle_types = [abs(int(getattr(child, "pdg_encoding", 0) or 0)) for child in children]

    particle_strings = [getattr(child, "type_string", "") for child in children]
    ic_hit = any(
        (_has_signature(child, surface) != -1) and np.isfinite(getattr(child, "length", np.nan))
        for child in children
    )
    ic_hit = True  # Reference implementation forces IC_hit True after computation

    if (interaction_type == GLASHOW_INTERACTION and len(particle_types) == 1
            and particle_strings[0] == 'Hadrons'):
        return 7

    if (11 in particle_types) or (interaction_type == NC_INTERACTION):
        return 1 if ic_hit else 0

    if 13 in particle_types:
        mu_index = particle_types.index(13)
        mu_particle = children[mu_index]
        mu_signature = _has_signature(mu_particle, surface)

        if not ic_hit:
            return 11
        elif interaction_type == GLASHOW_INTERACTION:
            if mu_signature == 0:
                return 8
        elif mu_signature == 0:
            return 3
        elif mu_signature == 1:
            return 2
        elif mu_signature == 2:
            return 4
        elif mu_signature == -1:
            return 11
        return pclass

    if 15 in particle_types:
        tau_index = particle_types.index(15)
        tau_particle = children[tau_index]
        if not ic_hit:
            return 12
        elif interaction_type == GLASHOW_INTERACTION:
            return 9

        tau_children = _get_children(mc_tree, tau_particle)
        tau_child = tau_children[-1] if tau_children else None
        tau_signature = _has_signature(tau_particle, surface)

        if tau_child is not None:
            tau_child_pdg = abs(int(getattr(tau_child, "pdg_encoding", 0) or 0))
            tau_child_sig = _has_signature(tau_child, surface)
            if tau_child_pdg == 13:
                if tau_child_sig == 0:
                    return 3
                if tau_child_sig == 1:
                    return 2
                if tau_child_sig == 2:
                    return 4
            else:
                if tau_signature == 0 and tau_child_sig == 0:
                    return 5
                if tau_signature == 0 and tau_child_sig == -1:
                    return 3
                if tau_signature == 2 and tau_child_sig == 0:
                    return 6
                if tau_signature == 1:
                    return 2

        if tau_signature == 0:
            return 3
        if tau_signature == 1:
            return 2
        if tau_signature == 2:
            return 4

    return 100


def compute_morphology_labels(frame: icetray.I3Frame,
                              mc_tree: Optional["dataclasses.I3MCTree"],
                              primary: Optional[dataclasses.I3Particle],
                              surface: Optional[phys_services.ExtrudedPolygon]) -> Tuple[int, int]:
    """Return (classification_code, morphology_label) from MC truth."""
    if frame is None:
        return 100, CLASSIFICATION_TO_MORPHOLOGY.get(100, 5)

    interaction_type = _infer_interaction_type(frame)
    if interaction_type == ATMOSPHERIC_INTERACTION and not frame.Has("I3MCWeightDict") and not frame.Has("EventProperties"):
        pclass = _classify_corsika_event(mc_tree, surface)
    else:
        pclass = _classify_neutrino_event(mc_tree, primary, surface, interaction_type)

    morphology = CLASSIFICATION_TO_MORPHOLOGY.get(pclass, 5)
    return pclass, morphology

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

def _should_emit_frame(frame: icetray.I3Frame,
                       allowed_streams: Optional[Set[str]]) -> bool:
    if frame is None or frame.Stop != icetray.I3Frame.Physics:
        return False
    if not frame.Has("I3EventHeader"):
        return False

    stream = frame["I3EventHeader"].sub_event_stream
    if allowed_streams is None:
        return stream != "NullSplit"
    return stream in allowed_streams


def iter_i3_events(i3_files: List[str],
                   allowed_streams: Optional[Set[str]] = None) -> Iterator[Tuple[icetray.I3Frame, str]]:
    """Iterate over physics frames, yielding each frame with its source file path."""
    for path in i3_files:
        label = os.path.basename(path)
        print(f"Processing {label}...")

        with closing(dataio.I3File(path)) as i3_file:
            for frame in i3_file:
                if _should_emit_frame(frame, allowed_streams):
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


def parse_mc_truth(frame: icetray.I3Frame,
                   surface: Optional[phys_services.ExtrudedPolygon] = None,
                   skip_starting_flag: bool = False) -> Dict[str, Any]:
    """Parse MC truth information from an I3Frame."""
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

    event_class, morphology = compute_morphology_labels(frame, mc_tree, primary, surface)
    parsed['event_class'] = event_class
    parsed['morphology'] = morphology

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

    if skip_starting_flag:
        parsed['starting'] = False
    else:
        parsed['starting'] = compute_starting_flag(mc_tree, surface)

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
    
    # Load geometry
    gcd_file = os.path.join(os.path.dirname(__file__), '..', 'resources', 'GeoCalibDetectorStatus_IC86.AVG_Pass2_SF0.99.i3')
    geometry = load_geometry(gcd_file)
    detector_surface = phys_services.ExtrudedPolygon(geometry, 0.0)
    
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
    
    for frame, source_path in iter_i3_events(i3_files, stream_lookup):
        if not frame_passes_filters(frame, filter_lookup):
            continue

        # Create event record from MC truth
        skip_starting_label = "corsika" in os.path.basename(source_path).lower()
        mc_truth = parse_mc_truth(frame, detector_surface, skip_starting_flag=skip_starting_label)
        
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
