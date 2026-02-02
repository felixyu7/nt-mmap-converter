"""
IceCube neutrino telescope data parser.
Handles conversion from i3 files to memory-mapped format.
"""

import glob
import os
from contextlib import closing
from typing import List, Iterator, Tuple, Dict, Any, Optional, Set, Sequence
from enum import IntEnum

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

# PDG codes for particle classification
MUON_PDGS = {13, -13}
TAU_PDGS = {15, -15}
ELECTRON_PDGS = {11, -11}
ELECTRON_NEUTRINO_PDGS = {12, -12}
MUON_NEUTRINO_PDGS = {14, -14}
TAU_NEUTRINO_PDGS = {16, -16}
HADRON_PDG = -2000001006  # IceCube's Hadrons pseudo-particle

# Common hadron PDGs (matching I3EventLabeler's cascade_types)
MESON_PDGS = {111, 211, -211, 321, -321, 310, 130, 221}  # pions, kaons, eta
BARYON_PDGS = {2212, -2212, 2112, -2112, 3122, -3122}  # protons, neutrons, lambdas
LOSS_PDGS = {-2000001001, -2000001002, -2000001003, -2000001004, -2000001007}

# Cascade-type PDGs (electrons, gammas, hadrons) - matches I3EventLabeler
CASCADE_PDGS = ELECTRON_PDGS | {22, HADRON_PDG} | MESON_PDGS | BARYON_PDGS | LOSS_PDGS

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


def _get_siblings(mc_tree: "dataclasses.I3MCTree",
                  particle: dataclasses.I3Particle) -> List[dataclasses.I3Particle]:
    """Get sibling particles (children of the same parent)."""
    parent = _get_parent(mc_tree, particle)
    if parent is None:
        return []
    siblings = _get_children(mc_tree, parent)
    return [s for s in siblings if s != particle]


def _get_detector_primaries(mc_tree: "dataclasses.I3MCTree",
                            surface: phys_services.ExtrudedPolygon) -> List[dataclasses.I3Particle]:
    """
    Identify particles that actually intersect the detector volume.

    A "detector primary" is a particle that:
    1. Is born before entering the detector (0 < isect.first)
    2. Dies after entering the detector (isect.first < length)
    3. For neutrinos: must interact inside (length < isect.second)

    This matches I3EventLabeler's get_primaries() logic.
    """
    primaries = []

    for particle in mc_tree:
        pos = getattr(particle, "pos", None)
        direction = getattr(particle, "dir", None)
        if pos is None or direction is None:
            continue

        isect = _safe_intersection(surface, pos, direction)
        if isect is None:
            continue

        first, second = isect
        length = float(getattr(particle, "length", np.nan))

        if not np.isfinite(length):
            continue

        # Check: particle born before entry AND dies after entry
        if 0 < first < length:
            pdg = int(getattr(particle, "pdg_encoding", 0) or 0)

            if abs(pdg) in NEUTRINO_PDGS:
                # Neutrinos must interact inside detector
                if length < second:
                    primaries.append(particle)
            else:
                # Non-neutrinos just need to enter
                primaries.append(particle)

    return primaries


def _get_closest_particle(mc_tree: "dataclasses.I3MCTree",
                          surface: phys_services.ExtrudedPolygon,
                          max_distance: float = 300.0) -> Optional[dataclasses.I3Particle]:
    """
    Find the signal non-neutrino particle that closest approaches the detector.
    Uses perpendicular closest approach distance, matching I3EventLabeler::get_closest().
    """
    primaries = list(mc_tree.primaries) if hasattr(mc_tree, "primaries") else []
    cosmic_primary = primaries[0] if primaries else None

    closest_particle = None
    min_distance = max_distance

    for particle in mc_tree:
        # Only consider signal particles (in subtree of cosmic primary)
        if cosmic_primary is not None:
            try:
                if not mc_tree.is_in_subtree(cosmic_primary, particle):
                    continue
            except (AttributeError, RuntimeError):
                pass  # Fall back to considering all particles if method unavailable

        pdg = int(getattr(particle, "pdg_encoding", 0) or 0)
        if abs(pdg) in NEUTRINO_PDGS:
            continue

        energy = float(getattr(particle, "energy", 0.0))
        if energy <= 0.1:  # Skip low-energy particles (matches C++: p.GetEnergy() > 0.1)
            continue

        # Use closest_approach for perpendicular distance (matches I3EventLabeler)
        approach = surface.closest_approach(particle)
        distance = approach.distance  # NaN if particle intersects detector

        # NaN < min_distance is False, so intersecting particles are skipped naturally
        if distance < min_distance:
            min_distance = distance
            closest_particle = particle

    return closest_particle


def _get_track_containment(particle: dataclasses.I3Particle,
                           surface: phys_services.ExtrudedPolygon) -> str:
    """
    Determine track containment: 'starting', 'throughgoing', 'stopping', or 'none'.

    Based on I3EventLabeler logic:
    - starting: particle born inside detector (first <= 0), exits (length >= second)
    - stopping: particle stops inside detector (length < second)
    - throughgoing: enters (first > 0) and exits (length >= second)
    - none: doesn't intersect meaningfully
    """
    pos = getattr(particle, "pos", None)
    direction = getattr(particle, "dir", None)
    if pos is None or direction is None:
        return "none"

    isect = _safe_intersection(surface, pos, direction)
    if isect is None:
        return "none"

    first, second = isect
    length = float(getattr(particle, "length", np.nan))

    # Handle infinite/missing length as exiting
    if not np.isfinite(length):
        if first <= 0 and second > 0:
            return "starting"
        elif first > 0:
            return "throughgoing"
        return "none"

    # Particle starts inside detector
    if first <= 0 and second > 0:
        if length < second:
            return "stopping"  # Born inside, stops inside
        return "starting"  # Born inside, exits

    # Particle enters from outside
    if first > 0 and first < length:
        if length < second:
            return "stopping"
        else:
            return "throughgoing"

    return "none"


def _classify_uncontained(closest: Optional[dataclasses.I3Particle],
                          mc_tree: "dataclasses.I3MCTree") -> EventClass:
    """Classify events with no detector primaries."""
    if closest is None:
        return EventClass.BACKGROUND

    pdg = abs(int(getattr(closest, "pdg_encoding", 0) or 0))

    if pdg in MUON_PDGS:
        siblings = _get_siblings(mc_tree, closest)
        muon_siblings = [s for s in siblings if abs(int(getattr(s, "pdg_encoding", 0) or 0)) in MUON_PDGS]
        if len(muon_siblings) > 1:  # Match I3EventLabeler: requires >1 muon siblings
            return EventClass.SKIMMING_BUNDLE
        return EventClass.SKIMMING_TRACK
    elif pdg in TAU_PDGS:
        return EventClass.SKIMMING_SPUR
    elif pdg in CASCADE_PDGS:
        return EventClass.UNCONTAINED_CASCADE
    else:
        return EventClass.OTHER_UNCONTAINED


def _classify_bundle(primaries: List[dataclasses.I3Particle]) -> EventClass:
    """Classify events with multiple detector primaries."""
    all_muons = all(
        abs(int(getattr(p, "pdg_encoding", 0) or 0)) in MUON_PDGS
        for p in primaries
    )

    if all_muons:
        return EventClass.BUNDLE
    else:
        return EventClass.OTHER_MULTIPLE


def _classify_single_primary(primary: dataclasses.I3Particle,
                             mc_tree: "dataclasses.I3MCTree",
                             surface: phys_services.ExtrudedPolygon) -> EventClass:
    """
    Classify events with exactly one detector primary.

    This is the most complex classification function, handling all neutrino
    interaction types and track containment scenarios.
    """
    pdg = int(getattr(primary, "pdg_encoding", 0) or 0)
    abs_pdg = abs(pdg)
    children = _get_children(mc_tree, primary)

    # Get child PDGs and types
    child_pdgs = [abs(int(getattr(c, "pdg_encoding", 0) or 0)) for c in children]
    child_types = [getattr(c, "type_string", "") for c in children]

    # Check for hadrons in children
    has_hadrons = any(t == "Hadrons" for t in child_types)

    # Helper to find specific particle in children
    def find_child(pdg_set):
        for c in children:
            if abs(int(getattr(c, "pdg_encoding", 0) or 0)) in pdg_set:
                return c
        return None

    # ===== ELECTRON NEUTRINO PRIMARY =====
    if abs_pdg in ELECTRON_NEUTRINO_PDGS:
        if has_hadrons:
            # Check if there's an outgoing neutrino (NC) or electron (CC)
            if any(abs(p) in ELECTRON_NEUTRINO_PDGS for p in child_pdgs):
                return EventClass.HADR_CASCADE  # NC interaction
            elif any(p in ELECTRON_PDGS for p in child_pdgs):
                return EventClass.EM_HADR_CASCADE  # NuE CC
            else:
                return EventClass.GLASHOW_HADR  # Glashow -> hadrons only
        else:
            # No hadrons - Glashow resonance W decay
            if any(p in ELECTRON_PDGS for p in child_pdgs):
                return EventClass.GLASHOW_ELECTRON
            elif any(p in MUON_PDGS for p in child_pdgs):
                muon = find_child(MUON_PDGS)
                containment = _get_track_containment(muon, surface) if muon else "none"
                if containment == "stopping":
                    return EventClass.GLASHOW_CONTAINED_TRACK
                else:
                    return EventClass.GLASHOW_STARTING_TRACK
            elif any(p in TAU_PDGS for p in child_pdgs):
                tau = find_child(TAU_PDGS)
                tau_containment = _get_track_containment(tau, surface) if tau else "none"
                if tau_containment in ("throughgoing", "starting"):
                    return EventClass.GLASHOW_STARTING_SPUR
                else:
                    # Tau decays inside - check decay products
                    tau_children = _get_children(mc_tree, tau) if tau else []
                    tau_child_pdgs = [abs(int(getattr(c, "pdg_encoding", 0) or 0)) for c in tau_children]
                    tau_child_types = [getattr(c, "type_string", "") for c in tau_children]

                    if any(t == "Hadrons" for t in tau_child_types):
                        return EventClass.GLASHOW_LOLLIPOP_HADR
                    elif any(p in ELECTRON_PDGS for p in tau_child_pdgs):
                        return EventClass.GLASHOW_LOLLIPOP_EM
                    elif any(p in MUON_PDGS for p in tau_child_pdgs):
                        mu_from_tau = next((c for c in tau_children if abs(int(getattr(c, "pdg_encoding", 0) or 0)) in MUON_PDGS), None)
                        mu_containment = _get_track_containment(mu_from_tau, surface) if mu_from_tau else "none"
                        if mu_containment == "stopping":
                            return EventClass.GLASHOW_TAU_CONTAINED_TRACK
                        else:
                            return EventClass.GLASHOW_TAU_STARTING_TRACK
            return EventClass.OTHER

    # ===== MUON NEUTRINO PRIMARY =====
    if abs_pdg in MUON_NEUTRINO_PDGS:
        if has_hadrons:
            if any(abs(p) in MUON_NEUTRINO_PDGS for p in child_pdgs):
                return EventClass.HADR_CASCADE  # NC interaction
            elif any(p in MUON_PDGS for p in child_pdgs):
                muon = find_child(MUON_PDGS)
                containment = _get_track_containment(muon, surface) if muon else "none"
                if containment == "stopping":
                    return EventClass.CONTAINED_TRACK
                else:
                    return EventClass.STARTING_TRACK
        return EventClass.OTHER

    # ===== TAU NEUTRINO PRIMARY =====
    if abs_pdg in TAU_NEUTRINO_PDGS:
        if has_hadrons:
            if any(abs(p) in TAU_NEUTRINO_PDGS for p in child_pdgs):
                return EventClass.HADR_CASCADE  # NC interaction
            elif any(p in TAU_PDGS for p in child_pdgs):
                tau = find_child(TAU_PDGS)
                tau_containment = _get_track_containment(tau, surface) if tau else "none"
                if tau_containment in ("throughgoing", "starting"):
                    return EventClass.INVERTED_LOLLIPOP
                else:
                    # Tau decays inside - check decay products
                    tau_children = _get_children(mc_tree, tau) if tau else []
                    tau_child_pdgs = [abs(int(getattr(c, "pdg_encoding", 0) or 0)) for c in tau_children]
                    tau_child_types = [getattr(c, "type_string", "") for c in tau_children]

                    if any(t == "Hadrons" for t in tau_child_types):
                        return EventClass.DOUBLE_BANG_HADR
                    elif any(p in ELECTRON_PDGS for p in tau_child_pdgs):
                        return EventClass.DOUBLE_BANG_EM
                    elif any(p in MUON_PDGS for p in tau_child_pdgs):
                        mu_from_tau = next((c for c in tau_children if abs(int(getattr(c, "pdg_encoding", 0) or 0)) in MUON_PDGS), None)
                        mu_containment = _get_track_containment(mu_from_tau, surface) if mu_from_tau else "none"
                        if mu_containment == "stopping":
                            return EventClass.INVERTED_LOLLIPOP_CONTAINED_TRACK
                        else:
                            return EventClass.INVERTED_LOLLIPOP_STARTING_TRACK
        return EventClass.OTHER

    # ===== MUON PRIMARY (entering from outside) =====
    if abs_pdg in MUON_PDGS:
        containment = _get_track_containment(primary, surface)
        if containment == "stopping":
            return EventClass.STOPPING_TRACK
        else:
            return EventClass.THROUGHGOING_TRACK

    # ===== TAU PRIMARY (entering from outside) =====
    if abs_pdg in TAU_PDGS:
        containment = _get_track_containment(primary, surface)
        if containment in ("throughgoing", "starting"):
            return EventClass.THROUGHGOING_SPUR
        else:
            # Tau decays inside - check decay products
            tau_children = _get_children(mc_tree, primary)
            tau_child_pdgs = [abs(int(getattr(c, "pdg_encoding", 0) or 0)) for c in tau_children]
            tau_child_types = [getattr(c, "type_string", "") for c in tau_children]

            if any(t == "Hadrons" for t in tau_child_types):
                return EventClass.LOLLIPOP_HADR
            elif any(p in ELECTRON_PDGS for p in tau_child_pdgs):
                return EventClass.LOLLIPOP_EM
            elif any(p in MUON_PDGS for p in tau_child_pdgs):
                mu_from_tau = next((c for c in tau_children if abs(int(getattr(c, "pdg_encoding", 0) or 0)) in MUON_PDGS), None)
                mu_containment = _get_track_containment(mu_from_tau, surface) if mu_from_tau else "none"
                if mu_containment == "stopping":
                    return EventClass.TAU_CONTAINED_TRACK
                else:
                    return EventClass.TAU_STARTING_TRACK

    # ===== CASCADE PRIMARY (shouldn't normally be a detector primary) =====
    if abs_pdg in CASCADE_PDGS:
        return EventClass.UNCONTAINED_CASCADE

    return EventClass.OTHER


def compute_morphology_labels(frame: icetray.I3Frame,
                              mc_tree: Optional["dataclasses.I3MCTree"],
                              primary: Optional[dataclasses.I3Particle],
                              surface: Optional[phys_services.ExtrudedPolygon]) -> Tuple[int, int]:
    """
    Return (event_class, morphology) using detector-primary approach.

    This uses the I3EventLabeler approach: first identify particles that
    actually intersect the detector ("detector primaries"), then classify
    based on the number and type of primaries.
    """
    if mc_tree is None or surface is None:
        return EventClass.OTHER, Morphology.UNCONTAINED

    # Get detector primaries using I3EventLabeler approach
    detector_primaries = _get_detector_primaries(mc_tree, surface)

    if len(detector_primaries) == 0:
        # No particles intersect detector - classify based on closest
        closest = _get_closest_particle(mc_tree, surface)
        event_class = _classify_uncontained(closest, mc_tree)
    elif len(detector_primaries) == 1:
        # Single primary - detailed classification
        event_class = _classify_single_primary(detector_primaries[0], mc_tree, surface)
    else:
        # Multiple primaries
        event_class = _classify_bundle(detector_primaries)

    morphology = DETAILED_TO_MORPHOLOGY.get(event_class, Morphology.UNCONTAINED)
    return int(event_class), int(morphology)

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
                   surface: Optional[phys_services.ExtrudedPolygon] = None) -> Dict[str, Any]:
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
    
    # Load geometry
    gcd_file = os.path.join(os.path.dirname(__file__), '..', 'resources', 'GeoCalibDetectorStatus_IC86.AVG_Pass2_SF0.99.i3')
    geometry = load_geometry(gcd_file)
    detector_surface = phys_services.ExtrudedPolygon(geometry, 50.0)  # Match I3EventLabeler default padding
    
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
        mc_truth = parse_mc_truth(frame, detector_surface)
        
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
