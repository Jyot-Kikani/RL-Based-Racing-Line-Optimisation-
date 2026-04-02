# ── env/track.py ──────────────────────────────────────────────────────────────
import numpy as np
import csv
from scipy.interpolate import splprep, splev


class Track:
    def __init__(self, csv_path: str, smooth: bool = True, n_points: int = 300):
        self.csv_path = csv_path
        self.n_points = n_points
        self._load(csv_path)
        if smooth:
            self._smooth()
        self._compute_bounds()
        self._compute_distances()

    def _load(self, path: str):
        xs, ys, wr, wl = [], [], [], []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                xs.append(float(row["x_m"]))
                ys.append(float(row["y_m"]))
                wr.append(float(row["w_tr_right_m"]))
                wl.append(float(row["w_tr_left_m"]))
        if len(xs) > 1 and abs(xs[-1]-xs[0]) < 1e-6 and abs(ys[-1]-ys[0]) < 1e-6:
            xs, ys, wr, wl = xs[:-1], ys[:-1], wr[:-1], wl[:-1]
        self._raw_xy      = np.column_stack([xs, ys])
        self._raw_w_right = np.array(wr)
        self._raw_w_left  = np.array(wl)

    def _smooth(self):
        xy = self._raw_xy
        x  = np.append(xy[:, 0], xy[0, 0])
        y  = np.append(xy[:, 1], xy[0, 1])
        tck, _ = splprep([x, y], s=10.0, per=True, k=3)
        u_new  = np.linspace(0, 1, self.n_points, endpoint=False)
        sx, sy = splev(u_new, tck)
        self.centerline = np.column_stack([sx, sy])
        old_u = np.linspace(0, 1, len(self._raw_w_right))
        new_u = np.linspace(0, 1, self.n_points)
        self._w_right = np.interp(new_u, old_u, self._raw_w_right)
        self._w_left  = np.interp(new_u, old_u, self._raw_w_left)

    def _compute_bounds(self):
        cl       = self.centerline
        tangents = np.roll(cl, -1, axis=0) - cl
        norms    = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / np.where(norms > 1e-9, norms, 1e-9)
        normals  = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        self.right_bound = cl + normals * self._w_right[:, None]
        self.left_bound  = cl - normals * self._w_left[:, None]

    def _compute_distances(self):
        cl    = self.centerline
        diffs = np.diff(cl, axis=0, append=cl[:1])
        segs  = np.linalg.norm(diffs, axis=1)
        self.distances    = np.cumsum(segs)
        self.track_length = self.distances[-1]

    # ── Nearest waypoint ──────────────────────────────────────────────────────
    def nearest_waypoint(self, x: float, y: float) -> int:
        diffs = self.centerline - np.array([x, y])
        return int(np.argmin(np.sum(diffs ** 2, axis=1)))

    def progress(self, x: float, y: float) -> float:
        return float(self.distances[self.nearest_waypoint(x, y)])

    # ── is_on_track: point-to-segment distance ────────────────────────────────
    def is_on_track(self, x: float, y: float) -> bool:
        """
        Uses true perpendicular distance from the point to the nearest
        centerline SEGMENT (not just the nearest waypoint).

        This fixes false off-track calls on the drag strip where two
        parallel straights are close together — the nearest *waypoint*
        can be on the wrong straight, but the nearest *segment* is always
        correct because we only check the segment local to the nearest idx.
        """
        pt  = np.array([x, y])
        idx = self.nearest_waypoint(x, y)
        n   = len(self.centerline)

        # Check the segment before and after the nearest waypoint
        best_dist = np.inf
        best_idx  = idx
        for i in [idx, (idx - 1) % n]:
            a = self.centerline[i]
            b = self.centerline[(i + 1) % n]
            d, t = _point_segment_dist(pt, a, b)
            if d < best_dist:
                best_dist = d
                # Interpolate width at the projection point t along segment
                w_r = self._w_right[i] * (1 - t) + self._w_right[(i+1) % n] * t
                w_l = self._w_left[i]  * (1 - t) + self._w_left[(i+1)  % n] * t
                best_w = min(w_r, w_l)

        return best_dist <= best_w

    # ── Convenience ───────────────────────────────────────────────────────────
    @property
    def start_pos(self):
        return self.centerline[0]

    @property
    def start_heading(self):
        p0, p1 = self.centerline[0], self.centerline[1]
        return float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))


# ── Helper (module-level, reused by track) ────────────────────────────────────
def _point_segment_dist(pt: np.ndarray, a: np.ndarray, b: np.ndarray):
    """
    Returns (distance, t) where t in [0,1] is how far along segment a→b
    the closest point lies. Pure vector math — no loops.
    """
    ab = b - a
    ab_sq = np.dot(ab, ab)
    if ab_sq < 1e-12:          # degenerate segment (a == b)
        return float(np.linalg.norm(pt - a)), 0.0
    t  = float(np.clip(np.dot(pt - a, ab) / ab_sq, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(pt - closest)), t