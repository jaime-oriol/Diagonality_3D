"""
test_ppcf — Exhaustive tests for the Immediate Orientation-Aware PPCF
reach-field model.

Run: cd ~/TFG/DIAG && source ~/anaconda3/bin/activate footballdecoded \
     && python3 -m pytest test/test_ppcf.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.ppcf import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    default_params,
    reaction_time,
    cod_penalty,
    reaction_delay,
    _theta_player_to_targets,
    _player_influence,
    _extract_team_arrays,
    _aggregate_team_shares,
    compute_ppcf_surfaces,
    ppcf_at_targets,
)


def _make_frame(players):
    """Build a minimal orientations frame from a list of player dicts."""
    return pd.DataFrame(players)


@pytest.fixture
def trivial_frame():
    """1 attacker (team 1, jersey 10) at (-5, 0), 1 defender (team 0,
    jersey 5) at (5, 0), both stationary, both facing +X."""
    return _make_frame([
        dict(team=1, jersey=10, x=-5.0, y=0.0, shoulder_angle=0.0, vx=0.0, vy=0.0),
        dict(team=0, jersey=5, x=5.0, y=0.0, shoulder_angle=0.0, vx=0.0, vy=0.0),
    ])


class TestDefaultParams:
    def test_returns_dict_with_required_keys(self):
        required = frozenset({
            "rt_base", "rt_gain", "rt_theta_c_deg", "rt_theta_w_deg",
            "cod_amplitude", "blob_sigma_base", "blob_orientation_mod",
            "max_player_speed", "max_int_time",
        })
        assert required <= set(default_params().keys())

    def test_no_stale_spearman_keys(self):
        """Ensure all Spearman/Opta legacy parameters are gone."""
        p = default_params()
        forbidden = frozenset({
            "lambda_att", "lambda_def", "lambda_gk", "lambda_gk_boost",
            "tti_sigma", "kappa_def",
            "time_to_control_att", "time_to_control_def", "time_to_control_veto",
            "int_dt", "model_converge_tol", "average_ball_speed",
        })
        assert forbidden.isdisjoint(p.keys())

    def test_default_window_is_2_5s(self):
        assert default_params()["max_int_time"] == 2.5

    def test_custom_window(self):
        assert default_params(immediate_window=0.5)["max_int_time"] == 0.5

    def test_rt_base_and_gain(self):
        p = default_params()
        assert p["rt_base"] == 0.7
        assert p["rt_gain"] == 0.4

    def test_cod_amplitude_positive(self):
        assert default_params()["cod_amplitude"] > 0

    def test_blob_sigma_base_sensible(self):
        p = default_params()
        assert 5.0 <= p["blob_sigma_base"] <= 12.0

    def test_blob_orientation_mod_range(self):
        p = default_params()
        assert 0.0 < p["blob_orientation_mod"] < 1.0

    def test_max_player_speed_is_5(self):
        assert default_params()["max_player_speed"] == 5.0


# ── Reaction time ────────────────────────────────────────────────────────

class TestReactionTime:
    def test_rt_at_theta_0_near_base(self):
        p = default_params()
        rt = reaction_time(np.array(0.0), p)
        assert abs(rt - 0.707) < 0.01

    def test_rt_at_theta_pi_near_base_plus_gain(self):
        p = default_params()
        rt = reaction_time(np.array(np.pi), p)
        assert abs(rt - 1.1) < 0.001

    def test_rt_monotonic(self):
        p = default_params()
        thetas = np.linspace(0, np.pi, 30)
        rts = reaction_time(thetas, p)
        assert np.all(np.diff(rts) >= 0)

    def test_rt_at_sigmoid_center(self):
        p = default_params()
        rt = reaction_time(np.array(np.radians(60)), p)
        assert abs(rt - (0.7 + 0.4 * 0.5)) < 1e-06

    def test_rt_bounds(self):
        p = default_params()
        thetas = np.linspace(0, np.pi, 100)
        rts = reaction_time(thetas, p)
        assert np.all(rts >= 0.7)
        assert np.all(rts <= 1.11)

    def test_rt_accepts_2d_input(self):
        p = default_params()
        thetas = np.random.uniform(0, np.pi, (5, 7))
        rts = reaction_time(thetas, p)
        assert rts.shape == (5, 7)

    def test_rt_no_overflow_at_extremes(self):
        p = default_params()
        rts = reaction_time(np.array([-10, 20]), p)
        assert np.all(np.isfinite(rts))


# ── COD penalty ──────────────────────────────────────────────────────────

class TestCodPenalty:
    def test_zero_at_theta_0(self):
        assert cod_penalty(np.array(0.0), default_params()) == 0.0

    def test_max_at_theta_pi(self):
        p = default_params()
        assert abs(cod_penalty(np.array(np.pi), p) - p["cod_amplitude"]) < 1e-09

    def test_half_amplitude_at_90deg(self):
        p = default_params()
        cod = cod_penalty(np.array(np.pi / 2), p)
        assert abs(cod - 0.5 * p["cod_amplitude"]) < 1e-09

    def test_monotonic(self):
        p = default_params()
        thetas = np.linspace(0, np.pi, 50)
        cods = cod_penalty(thetas, p)
        assert np.all(np.diff(cods) >= 0)


# ── Reaction delay ───────────────────────────────────────────────────────

class TestReactionDelay:
    def test_total_at_0(self):
        p = default_params()
        d = reaction_delay(np.array(0.0), p)
        assert abs(d - 0.707) < 0.01

    def test_total_at_pi(self):
        p = default_params()
        d = reaction_delay(np.array(np.pi), p)
        expected = p["rt_base"] + p["rt_gain"] + p["cod_amplitude"]
        assert abs(d - expected) < 0.001

    def test_monotonic(self):
        p = default_params()
        thetas = np.linspace(0, np.pi, 100)
        delays = reaction_delay(thetas, p)
        assert np.all(np.diff(delays) >= 0)


# ── Theta geometry ───────────────────────────────────────────────────────

class TestThetaGeometry:
    def test_shape(self):
        pos = np.array([[0.0, 0.0], [10.0, 5.0]])
        facing = np.array([0.0, 0.0])
        targets = np.array([[10.0, 0.0], [0.0, 5.0], [-5.0, -5.0]])
        theta = _theta_player_to_targets(pos, facing, targets)
        assert theta.shape == (2, 3)

    def test_facing_directly_at_target_is_zero(self):
        theta = _theta_player_to_targets(
            np.array([[0.0, 0.0]]),
            np.array([0.0]),
            np.array([[10.0, 0.0]]),
        )
        assert abs(theta[0, 0]) < 1e-09

    def test_facing_opposite_is_pi(self):
        theta = _theta_player_to_targets(
            np.array([[0.0, 0.0]]),
            np.array([0.0]),
            np.array([[-10.0, 0.0]]),
        )
        assert abs(theta[0, 0] - np.pi) < 1e-09

    def test_facing_perpendicular_is_half_pi(self):
        theta = _theta_player_to_targets(
            np.array([[0.0, 0.0]]),
            np.array([0.0]),
            np.array([[0.0, 10.0]]),
        )
        assert abs(theta[0, 0] - np.pi / 2) < 1e-09

    def test_unsigned(self):
        """+Y and -Y should both give pi/2 when facing +X."""
        theta = _theta_player_to_targets(
            np.array([[0.0, 0.0]]),
            np.array([0.0]),
            np.array([[0.0, 5.0], [0.0, -5.0]]),
        )
        assert abs(theta[0, 0] - np.pi / 2) < 1e-09
        assert abs(theta[0, 1] - np.pi / 2) < 1e-09

    def test_empty_players(self):
        theta = _theta_player_to_targets(
            np.zeros((0, 2)),
            np.zeros(0),
            np.array([[1.0, 0.0]]),
        )
        assert theta.shape == (0, 1)


# ── Player influence ─────────────────────────────────────────────────────

class TestPlayerInfluence:
    def test_peak_at_player_position(self):
        """Stationary player -> influence = 1 at their own position."""
        p = default_params()
        pos = np.array([[0.0, 0.0]])
        vel = np.zeros_like(pos)
        facing = np.array([0.0])
        targets = np.array([[0.0, 0.0]])
        infl = _player_influence(pos, vel, facing, targets, p)
        assert abs(infl[0, 0] - 1.0) < 1e-09

    def test_decays_with_distance(self):
        p = default_params()
        pos = np.array([[0.0, 0.0]])
        vel = np.zeros_like(pos)
        facing = np.array([0.0])
        targets = np.array([[0.0, 0.0], [5.0, 0.0], [15.0, 0.0], [25.0, 0.0], [40.0, 0.0]])
        infl = _player_influence(pos, vel, facing, targets, p)
        assert infl[0, 0] > infl[0, 1] > infl[0, 2] > infl[0, 3] > infl[0, 4]
        assert infl[0, 4] < 0.01

    def test_facing_target_larger_than_facing_away(self):
        """A player facing a target reaches it; facing away, reach is 0
        and only min_sigma applies -> smaller blob."""
        p = default_params()
        pos = np.array([[0.0, 0.0]])
        vel = np.zeros_like(pos)
        targets = np.array([[2.0, 0.0]])
        infl_facing = _player_influence(pos, vel, np.array([0.0]), targets, p)
        infl_away = _player_influence(pos, vel, np.array([np.pi]), targets, p)
        assert infl_facing[0, 0] > infl_away[0, 0]

    def test_blind_spot_has_smaller_sigma(self):
        """Blind-spot sigma should be smaller than front sigma, producing
        lower influence at the same distance."""
        p = default_params()
        pos = np.array([[0.0, 0.0]])
        vel = np.zeros_like(pos)
        facing = np.array([0.0])
        front = np.array([[8.0, 0.0]])
        back = np.array([[-8.0, 0.0]])
        i_front = _player_influence(pos, vel, facing, front, p)[0, 0]
        i_back = _player_influence(pos, vel, facing, back, p)[0, 0]
        assert i_back > 0.05
        assert i_front > 2.0 * i_back

    def test_anisotropic_blob_is_asymmetric(self):
        """Same distance in front vs behind should give different influence.
        Uses 12m where the +/-30% sigma modulation produces a visible ratio."""
        p = default_params()
        pos = np.array([[0.0, 0.0]])
        vel = np.zeros_like(pos)
        facing = np.array([0.0])
        targets = np.array([[12.0, 0.0], [-12.0, 0.0]])
        infl = _player_influence(pos, vel, facing, targets, p)
        assert infl[0, 0] > 1.3 * infl[0, 1]

    def test_drift_with_velocity(self):
        """A player moving in +X direction should effectively "reach further"
        toward +X than a stationary player."""
        p = default_params()
        pos = np.array([[0.0, 0.0]])
        facing = np.array([0.0])
        targets = np.array([[3.0, 0.0]])
        infl_still = _player_influence(pos, np.array([[0.0, 0.0]]), facing, targets, p)
        infl_moving = _player_influence(pos, np.array([[2.0, 0.0]]), facing, targets, p)
        assert infl_moving[0, 0] > infl_still[0, 0]

    def test_empty_players(self):
        infl = _player_influence(
            np.zeros((0, 2)),
            np.zeros((0, 2)),
            np.zeros(0),
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            default_params(),
        )
        assert infl.shape == (0, 2)

    def test_multiple_players(self):
        p = default_params()
        pos = np.array([[-5.0, 0.0], [5.0, 0.0]])
        vel = np.zeros_like(pos)
        facing = np.array([0.0, np.pi])
        targets = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        infl = _player_influence(pos, vel, facing, targets, p)
        assert infl.shape == (2, 3)
        assert abs(infl[1, 1] - infl[0, 0]) < 1e-09

    def test_values_in_valid_range(self):
        p = default_params()
        rng = np.random.default_rng(42)
        pos = rng.uniform(-40, 40, (5, 2))
        vel = rng.uniform(-3, 3, (5, 2))
        facing = rng.uniform(-np.pi, np.pi, 5)
        targets = rng.uniform(-50, 50, (20, 2))
        infl = _player_influence(pos, vel, facing, targets, p)
        assert np.all(infl >= 0)
        assert np.all(infl <= 1.0 + 1e-09)
        assert np.all(np.isfinite(infl))


# ── Aggregate team shares ────────────────────────────────────────────────

class TestAggregateTeamShares:
    def test_empty_zero_output(self):
        I_att = np.array([0.0, 0.0])
        I_def = np.array([0.0, 0.0])
        att, def_ = _aggregate_team_shares(I_att, I_def)
        assert np.allclose(att, 0.0)
        assert np.allclose(def_, 0.0)

    def test_att_only_resolves_to_att(self):
        I_att = np.array([2.0])
        I_def = np.array([0.0])
        att, def_ = _aggregate_team_shares(I_att, I_def)
        assert att[0] > 0.8
        assert def_[0] == 0.0

    def test_def_only_resolves_to_def(self):
        att, def_ = _aggregate_team_shares(np.array([0.0]), np.array([2.0]))
        assert att[0] == 0.0
        assert def_[0] > 0.8

    def test_equal_influence_splits_equally(self):
        att, def_ = _aggregate_team_shares(np.array([1.0]), np.array([1.0]))
        assert abs(att[0] - def_[0]) < 1e-09
        assert (att[0] + def_[0]) < 1.0

    def test_sum_equals_resolved(self):
        I_att = np.array([0.5, 1.0, 0.0, 3.0])
        I_def = np.array([0.0, 0.5, 0.5, 2.0])
        att, def_ = _aggregate_team_shares(I_att, I_def)
        expected_resolved = 1.0 - np.exp(-(I_att + I_def))
        assert np.allclose(att + def_, expected_resolved)

    def test_all_values_in_unit_interval(self):
        rng = np.random.default_rng(0)
        I_att = rng.uniform(0, 5, 50)
        I_def = rng.uniform(0, 5, 50)
        att, def_ = _aggregate_team_shares(I_att, I_def)
        assert np.all(att >= 0) and np.all(att <= 1)
        assert np.all(def_ >= 0) and np.all(def_ <= 1)
        assert np.all((att + def_) <= 1.0 + 1e-09)


# ── Extract team arrays ──────────────────────────────────────────────────

class TestExtractTeamArrays:
    def test_extracts_pos_vel_facing(self, trivial_frame):
        pos, vel, facing = _extract_team_arrays(trivial_frame, 1)
        assert pos.shape == (1, 2)
        assert vel.shape == (1, 2)
        assert facing.shape == (1,)
        assert np.allclose(pos, [[-5.0, 0.0]])

    def test_handles_missing_velocity(self):
        frame = pd.DataFrame([dict(team=1, jersey=10, x=0.0, y=0.0, shoulder_angle=0.0)])
        _, vel, _ = _extract_team_arrays(frame, 1)
        assert np.allclose(vel, 0.0)

    def test_handles_nan_velocity(self):
        frame = pd.DataFrame([dict(team=1, jersey=10, x=0.0, y=0.0, shoulder_angle=0.0, vx=np.nan, vy=np.nan)])
        _, vel, _ = _extract_team_arrays(frame, 1)
        assert np.allclose(vel, 0.0)

    def test_empty_team(self, trivial_frame):
        pos, vel, facing = _extract_team_arrays(trivial_frame, 99)
        assert pos.shape == (0, 2)
        assert vel.shape == (0, 2)
        assert facing.shape == (0,)

    def test_handles_missing_shoulder_angle(self):
        frame = pd.DataFrame([dict(team=1, jersey=10, x=0.0, y=0.0, vx=0.0, vy=0.0)])
        _, _, facing = _extract_team_arrays(frame, 1)
        assert np.allclose(facing, 0.0)


# ── Compute PPCF surfaces ───────────────────────────────────────────────

class TestComputePpcfSurfaces:
    def test_returns_grid_shapes(self, trivial_frame):
        att, def_, xg, yg = compute_ppcf_surfaces(
            trivial_frame, attacking_team=1, n_grid_x=50,
        )
        n_grid_y = int(round(PITCH_WIDTH / PITCH_LENGTH * 50))
        assert att.shape == (n_grid_y, 50)
        assert def_.shape == (n_grid_y, 50)
        assert xg.shape == (50,)
        assert yg.shape == (n_grid_y,)

    def test_grid_spans_pitch(self, trivial_frame):
        _, _, xg, yg = compute_ppcf_surfaces(trivial_frame, attacking_team=1)
        assert xg.min() > -PITCH_LENGTH / 2
        assert xg.max() < PITCH_LENGTH / 2
        assert (yg.max() - yg.min()) > 0.95 * PITCH_WIDTH

    def test_values_in_valid_range(self, trivial_frame):
        att, def_, _ , _ = compute_ppcf_surfaces(
            trivial_frame, attacking_team=1, n_grid_x=40,
        )
        assert np.all(att >= 0) and np.all(att <= 1)
        assert np.all(def_ >= 0) and np.all(def_ <= 1)
        assert np.all((att + def_) <= 1.0 + 1e-09)

    def test_missing_columns_raises(self):
        bad = pd.DataFrame([dict(team=1, x=0.0, y=0.0)])
        with pytest.raises(ValueError, match="shoulder_angle"):
            compute_ppcf_surfaces(bad, attacking_team=1)

    def test_attacker_zone_att_dominates(self, trivial_frame):
        att, def_, xg, yg = compute_ppcf_surfaces(
            trivial_frame, attacking_team=1, n_grid_x=60,
        )
        ix = np.argmin(np.abs(xg - (-5.0)))
        iy = np.argmin(np.abs(yg - 0.0))
        assert att[iy, ix] > def_[iy, ix]

    def test_defender_zone_def_dominates(self, trivial_frame):
        att, def_, xg, yg = compute_ppcf_surfaces(
            trivial_frame, attacking_team=1, n_grid_x=60,
        )
        ix = np.argmin(np.abs(xg - 5.0))
        iy = np.argmin(np.abs(yg - 0.0))
        assert def_[iy, ix] > att[iy, ix]

    def test_empty_frame_gives_zero_surfaces(self):
        empty = pd.DataFrame({
            "team": pd.Series(dtype=int),
            "x": pd.Series(dtype=float),
            "y": pd.Series(dtype=float),
            "shoulder_angle": pd.Series(dtype=float),
        })
        att, def_, _, _ = compute_ppcf_surfaces(empty, attacking_team=1, n_grid_x=20)
        assert np.allclose(att, 0.0)
        assert np.allclose(def_, 0.0)

    def test_only_attackers(self, trivial_frame):
        frame = _make_frame([
            dict(team=1, jersey=9, x=0.0, y=0.0, shoulder_angle=0.0, vx=0.0, vy=0.0),
        ])
        att, def_, _, _ = compute_ppcf_surfaces(
            frame, attacking_team=1, n_grid_x=30,
        )
        assert att.sum() > 0
        assert def_.sum() == 0


# ── Orientation signature ────────────────────────────────────────────────

class TestOrientationSignature:
    """The defining tests: rotating a defender away from a contested
    cell must reduce their influence there, increasing the attacker's
    share. This is the diagonality hypothesis at the pixel level."""

    def test_defender_facing_away_increases_att_share(self):
        """Attacker and defender at equal distance from a single contested
        target. With defender facing the target -> defender wins. With
        defender facing away -> attacker wins."""
        p = default_params()
        targets = np.array([[0.0, 0.0]])

        att_pos = np.array([[-1.0, 0.0]])
        att_vel = np.zeros_like(att_pos)
        att_facing = np.array([0.0])
        def_pos = np.array([[1.0, 0.0]])
        def_vel = np.zeros_like(def_pos)

        I_att = _player_influence(att_pos, att_vel, att_facing, targets, p).sum(axis=0)

        # Defender facing pi -> facing the target (at -X from pos 1,0)
        I_def_facing = _player_influence(def_pos, def_vel, np.array([np.pi]), targets, p).sum(axis=0)
        att_share_A, _ = _aggregate_team_shares(I_att, I_def_facing)

        # Defender facing 0.0 -> facing away from the target
        I_def_away = _player_influence(def_pos, def_vel, np.array([0.0]), targets, p).sum(axis=0)
        att_share_B, _ = _aggregate_team_shares(I_att, I_def_away)

        assert att_share_B > att_share_A, (
            f"Attacker share should be higher when defender faces away. "
            f"facing={att_share_A:.3f}, away={att_share_B:.3f}"
        )

    def test_blob_has_blind_spot_compression(self):
        """At 15m, the +/-30% sigma modulation should produce a visible
        front-to-back ratio (both still have influence, but front > back)."""
        p = default_params()
        pos = np.array([[0.0, 0.0]])
        vel = np.zeros_like(pos)
        facing = np.array([0.0])
        front = np.array([[10.0, 0.0]])
        back = np.array([[-10.0, 0.0]])
        i_front = _player_influence(pos, vel, facing, front, p)[0, 0]
        i_back = _player_influence(pos, vel, facing, back, p)[0, 0]
        assert i_back > 0.01
        assert i_front > 1.3 * i_back


# ── PPCF at targets ─────────────────────────────────────────────────────

class TestPpcfAtTargets:
    def test_returns_tuple_shape(self, trivial_frame):
        att, def_ = ppcf_at_targets(
            trivial_frame,
            np.array([[-5.0, 0.0], [0.0, 0.0], [5.0, 10.0]]),
            attacking_team=1,
        )
        assert att.shape == (3,)
        assert def_.shape == (3,)

    def test_accepts_1d_input(self, trivial_frame):
        att, def_ = ppcf_at_targets(
            trivial_frame,
            np.array([0.0, 0.0]),
            attacking_team=1,
        )
        assert att.shape == (1,)
        assert def_.shape == (1,)

    def test_consistent_with_grid(self, trivial_frame):
        """Point-wise evaluation at a grid cell should match the grid."""
        att_grid, def_grid, xg, yg = compute_ppcf_surfaces(
            trivial_frame, attacking_team=1, n_grid_x=100,
        )
        ix, iy = 40, 15
        att_pt, def_pt = ppcf_at_targets(
            trivial_frame,
            np.array([[xg[ix], yg[iy]]]),
            attacking_team=1,
        )
        assert abs(att_pt[0] - att_grid[iy, ix]) < 1e-09
        assert abs(def_pt[0] - def_grid[iy, ix]) < 1e-09


# ── Realistic scene ──────────────────────────────────────────────────────

class TestRealisticScene:
    @pytest.fixture
    def scene(self):
        """11 attackers + 11 defenders in a typical attacking phase."""
        players = []
        att_positions = (
            (-40, 0), (-30, 15), (-30, -15), (-15, 10), (-15, -10),
            (0, 0), (5, 5), (5, -5), (15, 8), (15, -8), (20, 0),
        )
        for i, (x, y) in enumerate(att_positions):
            players.append(dict(
                team=1, jersey=i, x=float(x), y=float(y),
                shoulder_angle=0.0, vx=2.0, vy=0.0,
            ))
        def_positions = (
            (40, 0), (35, 12), (35, -12), (25, 8), (25, -8),
            (15, 0), (10, 10), (10, -10), (0, 5), (0, -5), (-10, 0),
        )
        for i, (x, y) in enumerate(def_positions):
            players.append(dict(
                team=0, jersey=i, x=float(x), y=float(y),
                shoulder_angle=np.pi, vx=0.0, vy=0.0,
            ))
        return pd.DataFrame(players)

    def test_scene_has_nontrivial_surfaces(self, scene):
        att, def_, _, _ = compute_ppcf_surfaces(scene, attacking_team=1, n_grid_x=60)
        assert att.sum() > 0 and def_.sum() > 0
        assert att.max() > 0.2

    def test_attacker_side_dominates_left(self, scene):
        att, def_, xg, _ = compute_ppcf_surfaces(scene, attacking_team=1, n_grid_x=60)
        left_mask = xg < -15
        assert att[:, left_mask].mean() > def_[:, left_mask].mean()

    def test_contested_dense_scene_rotation(self):
        """Dense contested micro-scene: attacker and defender 2m apart,
        cell between them. Rotating the defender to face away must
        increase attacker share at that cell. This is the scene-level
        version of the orientation signature thesis, with distances
        small enough to fall within the 1s reach-field window."""
        frame = _make_frame([
            dict(team=1, jersey=9, x=0.0, y=0.0, shoulder_angle=0.0, vx=0.0, vy=0.0),
            dict(team=0, jersey=4, x=2.0, y=0.0, shoulder_angle=np.pi, vx=0.0, vy=0.0),
        ])
        target = np.array([[1.0, 0.0]])

        att_normal, _ = ppcf_at_targets(frame, target, attacking_team=1)

        rotated = frame.copy()
        rotated.loc[rotated["team"] == 0, "shoulder_angle"] = 0.0
        att_rotated, _ = ppcf_at_targets(rotated, target, attacking_team=1)

        assert att_rotated[0] > att_normal[0], (
            f"att share should rise when defender flips to blind spot. "
            f"normal={att_normal[0]:.4f}, rotated={att_rotated[0]:.4f}"
        )


# ── Determinism ──────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_input_same_output(self, trivial_frame):
        a1, d1, _, _ = compute_ppcf_surfaces(trivial_frame, attacking_team=1, n_grid_x=30)
        a2, d2, _, _ = compute_ppcf_surfaces(trivial_frame, attacking_team=1, n_grid_x=30)
        assert np.array_equal(a1, a2)
        assert np.array_equal(d1, d2)
