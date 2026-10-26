from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import re

import numpy as np


@dataclass
class TrackerConfig:
    # A capsule may move between frames by at most this fraction of its radius (after the field shift is removed)
    max_distance: float = 0.5
    # Maximum relative change of the capsule radius between frames
    max_size_change: float = 0.3
    # Tolerances are multiplied by this factor when a capsule is cropped by the image border:
    # its ellipse is fitted to the visible part only and jumps between frames
    cropped_tolerance: float = 2.0
    # A track survives this many frames without a detection (the model may miss a capsule in some frames)
    max_gap: int = 3


@dataclass
class _Capsule:
    center: np.ndarray
    radius: float
    complete: bool


@dataclass
class _Track:
    track_id: int
    center: np.ndarray  # in the coordinates of the current frame
    radius: float
    complete: bool
    gap: int = 0


@dataclass
class TrackingResult:
    # track_ids[frame][capsule] -> track id, in the same order as the capsules passed in
    track_ids: List[List[int]] = field(default_factory=list)
    # Field (stage) shift of every frame relative to the first one, px
    field_shifts: List[Tuple[float, float]] = field(default_factory=list)


def natural_sort_key(text: str):
    """'frame_2' < 'frame_10' (plain string sorting puts frame_10 first)."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', text)]


def _tolerance(a, b, config: TrackerConfig) -> float:
    return 1.0 if a.complete and b.complete else config.cropped_tolerance


def _size_ok(a, b, config: TrackerConfig) -> bool:
    limit = config.max_size_change * _tolerance(a, b, config)
    return abs(a.radius - b.radius) <= limit * max(a.radius, b.radius)


def _close(a, b, shift: np.ndarray, config: TrackerConfig) -> bool:
    limit = config.max_distance * _tolerance(a, b, config)
    return np.linalg.norm(b.center - (a.center + shift)) <= limit * a.radius and _size_ok(a, b, config)


def estimate_field_shift(previous: Sequence, current: Sequence, config: TrackerConfig) -> np.ndarray:
    """Shift of the whole field between two frames, estimated from capsule positions.

    Only complete capsules are used when both frames have them: ellipses of cropped capsules are fitted
    to the visible part and do not follow the stage. Every pair of similar capsules proposes a shift;
    the shift that brings the most capsules into agreement wins (ties go to the smaller shift),
    then it is refined by averaging the agreeing pairs.
    """
    if any(c.complete for c in previous) and any(c.complete for c in current):
        previous = [c for c in previous if c.complete]
        current = [c for c in current if c.complete]
    if not previous or not current:
        return np.zeros(2)

    candidates = [np.zeros(2)] + [b.center - a.center for a in previous for b in current if _size_ok(a, b, config)]

    def agreeing_pairs(shift):
        pairs = []
        for a in previous:
            b = min(current, key=lambda c: np.linalg.norm(c.center - (a.center + shift)))
            if _close(a, b, shift, config):
                pairs.append((a, b))
        return pairs

    best_shift, best_pairs = np.zeros(2), []
    for shift in candidates:
        pairs = agreeing_pairs(shift)
        if len(pairs) > len(best_pairs) or (len(pairs) == len(best_pairs) and pairs
                                            and np.linalg.norm(shift) < np.linalg.norm(best_shift)):
            best_shift, best_pairs = shift, pairs

    if not best_pairs:
        return np.zeros(2)
    return np.mean([b.center - a.center for a, b in best_pairs], axis=0)


def _match(tracks: List[_Track], capsules: List[_Capsule], config: TrackerConfig) -> List[Tuple[int, int]]:
    """Greedy one-to-one matching by normalized distance; there are only a few capsules per frame."""
    costs = []
    for t, track in enumerate(tracks):
        for j, capsule in enumerate(capsules):
            if _close(track, capsule, np.zeros(2), config):
                distance = np.linalg.norm(capsule.center - track.center) / track.radius
                size_change = abs(track.radius - capsule.radius) / track.radius
                costs.append((distance + size_change, t, j))

    matches, used_tracks, used_capsules = [], set(), set()
    for _, t, j in sorted(costs):
        if t not in used_tracks and j not in used_capsules:
            matches.append((t, j))
            used_tracks.add(t)
            used_capsules.add(j)
    return matches


def track_capsules(frames: List[List[tuple]], config: Optional[TrackerConfig] = None) -> TrackingResult:
    """Assign persistent track ids to capsules in an ordered sequence of frames.

    frames[k] is a list of (center_x, center_y, radius) or (center_x, center_y, radius, is_complete)
    for every capsule in frame k; an empty list means a frame without detections (or a failed image).
    """
    config = config or TrackerConfig()
    result = TrackingResult()
    tracks: List[_Track] = []
    next_id = 1
    total_shift = np.zeros(2)

    for frame in frames:
        capsules = [_Capsule(center=np.array(c[:2], dtype=np.float64), radius=float(c[2]),
                             complete=bool(c[3]) if len(c) > 3 else True) for c in frame]

        shift = estimate_field_shift(tracks, capsules, config)
        total_shift = total_shift + shift
        for track in tracks:
            track.center = track.center + shift

        frame_ids = [0] * len(capsules)
        matched_tracks = set()
        for t, j in _match(tracks, capsules, config):
            track, capsule = tracks[t], capsules[j]
            track.center, track.radius, track.complete, track.gap = capsule.center, capsule.radius, capsule.complete, 0
            frame_ids[j] = track.track_id
            matched_tracks.add(t)

        # New tracks are numbered left to right, not in the (arbitrary) detection order
        for j in sorted(range(len(capsules)), key=lambda j: tuple(capsules[j].center)):
            if frame_ids[j] == 0:
                capsule = capsules[j]
                tracks.append(_Track(track_id=next_id, center=capsule.center, radius=capsule.radius,
                                     complete=capsule.complete))
                frame_ids[j] = next_id
                matched_tracks.add(len(tracks) - 1)
                next_id += 1

        for t, track in enumerate(tracks):
            if t not in matched_tracks:
                track.gap += 1
        tracks = [track for track in tracks if track.gap <= config.max_gap]

        result.track_ids.append(frame_ids)
        result.field_shifts.append((float(total_shift[0]), float(total_shift[1])))

    return result
