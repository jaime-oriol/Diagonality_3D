"""
vision — Bekkers (SSAC 2026) Vision model, ported to TRACAB meters.

This is Bekkers' exact code copied and adapted for:
  - Coordinates: TRACAB meters centered (-52.5 to 52.5, -34 to 34)
    instead of Respo.Vision (0 to 100, 0 to 64)
  - human_width: 0.45m (real skeleton) instead of 1.8 (arbitrary)

All formulas, Gaussians, occlusion logic: UNCHANGED from Bekkers.
"""

import numpy as np
from dataclasses import dataclass


# --- Helpers (from Bekkers helpers.py, adapted to centered coords) --------

def _get_grid(v, dim, grid_dim):
    """Convert real coordinate to grid pixel index. Centered coords."""
    return int(((v / dim) + 0.5) * grid_dim)

def _get_meshgrid(grid_length, grid_width):
    y_idxs, x_idxs = np.meshgrid(
        np.arange(grid_width), np.arange(grid_length), indexing="ij"
    )
    return y_idxs, x_idxs

def _get_meshgrid_linspace(pitch_length, pitch_width, grid_length, grid_width):
    """Linspace grids in CENTERED meters (-half to +half)."""
    x_space = np.linspace(-pitch_length/2, pitch_length/2, int(grid_length))
    y_space = np.linspace(-pitch_width/2, pitch_width/2, int(grid_width))
    return np.meshgrid(x_space, y_space)

def _get_empty_grid(grid_length, grid_width):
    return np.zeros((int(grid_width), int(grid_length)))


# --- Vision class (Bekkers exact, centered coords) ------------------------

@dataclass
class Vision:
    x: float
    y: float
    head_angle: float
    speed: float
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    max_occlusion_alpha: float = 0.7
    wedge_radius: float = 150.0
    vision_angle: int = 120
    human_width: float = 1.8  # visual occlusion width (Bekkers default, not anatomical)
    smoothing_multiplier: float = 3.0

    def __post_init__(self):
        # Scale human_width by smoothing (Bekkers exact — makes shadows proportional to grid)
        self.human_width = self.human_width * self.smoothing_multiplier
        self.human_depth = self.human_width / 2.0

        grid_length = self.pitch_length * self.smoothing_multiplier
        grid_width = self.pitch_width * self.smoothing_multiplier

        self.radius = self.wedge_radius * self.smoothing_multiplier
        # Half-angle in radians: 120 deg total FOV -> 60 deg half = ~1.047 rad
        self.vision_radians = self.vision_angle * (np.pi / 360)

        self.grid_x = _get_grid(self.x, self.pitch_length, grid_length)
        self.grid_y = _get_grid(self.y, self.pitch_width, grid_width)

        self.y_idxs, self.x_idxs = _get_meshgrid(grid_length, grid_width)
        self.x_idxs_ls, self.y_idxs_ls = _get_meshgrid_linspace(
            self.pitch_length, self.pitch_width, grid_length, grid_width,
        )
        self.grid = _get_empty_grid(grid_length, grid_width)
        self._set_grids()

    def _set_grids(self):
        self.grid_dx = self.x_idxs - self.grid_x
        self.grid_dy = self.y_idxs - self.grid_y
        self.self_dx = self.x_idxs_ls - self.x
        self.self_dy = self.y_idxs_ls - self.y

    @staticmethod
    def _distance(dx, dy):
        return np.sqrt(dx**2 + dy**2)

    @staticmethod
    def _angle(dx, dy):
        return np.arctan2(dy, dx)

    def _theta(self, angle, player_angle):
        if isinstance(player_angle, np.ndarray):
            player_angle = player_angle[:, np.newaxis, np.newaxis]
        return np.arctan2(np.sin(angle - player_angle), np.cos(angle - player_angle))

    def _datm(self, dx, dy, player_angle):
        distance = self._distance(dx, dy)
        angle = self._angle(dx, dy)
        theta = self._theta(angle, player_angle)
        mask = (distance <= self.radius) & (np.abs(theta) <= self.vision_radians)
        return distance, angle, theta, mask

    @staticmethod
    def _gaussian(c, a, sigma):
        return np.exp(-c * (a / sigma) ** 2)

    @staticmethod
    def _normalize(m):
        if len(m.shape) == 3:
            mx = np.max(m, axis=(1, 2), keepdims=True)
            mx[mx == 0] = 1
            return m / mx
        elif len(m.shape) == 2:
            mx = np.max(m)
            return m / mx if mx > 0 else m
        return m

    # --- FOV (formulas 1-3) -----------------------------------------------

    def binary_field_of_view(self):
        player_grid = np.copy(self.grid)
        self.distance, self.angle_head, self.theta_head, self.mask_head = (
            self._datm(self.grid_dx, self.grid_dy, self.head_angle)
        )
        player_grid[self.mask_head] = 1
        self.fov_b = player_grid
        return self.fov_b

    def field_of_view(self, sigma_r=75, sigma_a=0.4):
        cr = min(0.25 * self.speed + 0.1, 2.6)
        ca = min(0.03 * self.speed + 0.2, 0.5)
        if not hasattr(self, "fov_b"):
            self.binary_field_of_view()
        sigma_r_scaled = sigma_r * self.smoothing_multiplier
        radial = self._gaussian(cr, self.distance, sigma_r_scaled)
        angular = self._gaussian(ca, self.theta_head, sigma_a)
        pvp = radial * angular * self.fov_b
        self.fov = self._normalize(pvp)
        return self.fov

    # --- Occlusion (formulas 4-7) — Bekkers exact -------------------------

    def _occlusion_vec(self, x2s, y2s, shoulder_angles, dist_ij):
        # Shoulder offset in REAL meters (not divided by smoothing — our linspace is in meters)
        def _sh_loc(c, func, perp):
            return c + (self.human_width / 2.0) * func(perp)

        perp_l = shoulder_angles + np.pi / 2
        perp_r = shoulder_angles - np.pi / 2
        sl_x = _sh_loc(x2s, np.cos, perp_l)
        sl_y = _sh_loc(y2s, np.sin, perp_l)
        sr_x = _sh_loc(x2s, np.cos, perp_r)
        sr_y = _sh_loc(y2s, np.sin, perp_r)

        d_sl = self._distance(sl_x - self.x, sl_y - self.y)
        d_sr = self._distance(sr_x - self.x, sr_y - self.y)
        min_d = np.minimum.reduce([dist_ij, d_sl, d_sr])

        distance = self._distance(self.x_idxs_ls - self.x, self.y_idxs_ls - self.y)
        distance = distance[np.newaxis, :, :]
        min_d = min_d[:, np.newaxis, np.newaxis]

        shadow_mask = distance > min_d
        player_masks = np.zeros((x2s.shape[0], *self.grid.shape))
        player_masks[shadow_mask] = self.max_occlusion_alpha
        return player_masks

    def _apparent_width_vec(self, x2s, y2s, angles):
        hw = self.human_width / 2
        hd = self.human_depth / 2
        corners = np.array([[hw,hd],[-hw,hd],[-hw,-hd],[hw,-hd]])
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        rot = np.array([[[c,-s],[s,c]] for c,s in zip(cos_a, sin_a)])
        gc = np.einsum("nij,kj->nki", rot, corners) + np.array([x2s,y2s]).T[:,np.newaxis,:]
        obs = np.array([self.x, self.y])
        vecs = gc - obs[np.newaxis, np.newaxis, :]
        v1, v2 = vecs[:,[0,1],:], vecs[:,[2,3],:]
        dots = np.sum(v1*v2, axis=2)
        mags = np.linalg.norm(v1, axis=2) * np.linalg.norm(v2, axis=2)
        ab = np.arccos(np.clip(dots/(mags+1e-10), -1, 1))
        return np.max(ab, axis=1)

    def _inter_player_angle_difference(self, angle):
        _, _, td, _ = self._datm(self.self_dx, self.self_dy, angle)
        return td

    def _occlusion_profile(self, cs, alpha, sigma_a=0.2):
        if isinstance(cs, np.ndarray):
            cs = cs[:, np.newaxis, np.newaxis]
        ag = self._gaussian(cs, alpha, sigma_a)
        return self._normalize(ag)

    def occlusions(self, x2s, y2s, shoulder_angles, sigma_a=0.2):
        x2s = np.asarray(x2s, dtype=float)
        y2s = np.asarray(y2s, dtype=float)
        shoulder_angles = np.asarray(shoulder_angles, dtype=float)
        theta_ij = self._angle(dx=x2s - self.x, dy=y2s - self.y)
        dist_ij = self._distance(dx=x2s - self.x, dy=y2s - self.y)
        theta_diff = self._inter_player_angle_difference(angle=theta_ij)
        omega = self._apparent_width_vec(x2s, y2s, shoulder_angles)
        c_s = 1 / (omega / (dist_ij + 1e-10))
        occ_prof = self._occlusion_profile(c_s, theta_diff, sigma_a)
        p_mask = self._occlusion_vec(x2s, y2s, shoulder_angles, dist_ij)
        masks = 1 - (occ_prof * p_mask)
        self.occs = np.prod(masks, axis=0)
        return self.occs

    def compute(self, x2s, y2s, shoulder_angles):
        fov = self.field_of_view()
        occs = self.occlusions(x2s, y2s, shoulder_angles)
        self.vision = fov * occs
        return self.vision


# --- Convenience wrappers -------------------------------------------------

def compute_player_vision(player_x, player_y, head_angle, speed,
                          other_x, other_y, other_shoulder_angles,
                          shoulder_width=0.45, smoothing=3.0):
    v = Vision(x=player_x, y=player_y, head_angle=head_angle, speed=speed,
               human_width=shoulder_width, smoothing_multiplier=smoothing)
    return v.compute(np.asarray(other_x, dtype=float),
                     np.asarray(other_y, dtype=float),
                     np.asarray(other_shoulder_angles, dtype=float))


def compute_team_vision(orientations_frame, team, smoothing=3.0):
    all_p = orientations_frame
    team_p = all_p[all_p["team"] == team]
    dummy = Vision(x=0, y=0, head_angle=0, speed=0, smoothing_multiplier=smoothing)
    combined = np.zeros_like(dummy.grid)
    grids = {}
    for _, p in team_p.iterrows():
        j = int(p["jersey"])
        others = all_p[all_p.index != p.name]
        if len(others) == 0: continue
        g = compute_player_vision(
            p["x"], p["y"], p["head_angle"],
            p.get("speed", 0.0) if not np.isnan(p.get("speed", 0.0)) else 0.0,
            others["x"].values, others["y"].values,
            others["shoulder_angle"].values,
            shoulder_width=p.get("shoulder_width", 0.45), smoothing=smoothing)
        grids[j] = g
        combined = 1.0 - (1.0 - combined) * (1.0 - g)
    return {"grids": grids, "combined": combined, "blind_spots": 1.0 - combined}
