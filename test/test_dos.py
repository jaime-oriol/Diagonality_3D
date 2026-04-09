"""
test_dos — Tests for the Diagonal Opportunity Surface module.

Run: cd ~/TFG/DIAG && source ~/anaconda3/bin/activate footballdecoded \
     && python3 -m pytest test/test_dos.py -v
"""
import numpy as np
import pandas as pd
import pytest

from src.dos import (
    classify_attack_angle,
    _build_vision_grids, _query_vision,
    _compute_awareness, _detection_delay,
    compute_dos_surface, dos_at_targets, dos_for_event,
)
from src.ppcf import default_params


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_frame(players):
    """Build a minimal orientations frame from a list of player dicts."""
    return pd.DataFrame(players)


def _full_player(team, jersey, x, y, shoulder_angle=0.0, head_angle=None,
                 speed=0.0, vx=0.0, vy=0.0, shoulder_width=0.45):
    """Create a complete player dict with all DOS-relevant columns."""
    return dict(
        team=team, jersey=jersey, x=x, y=y,
        shoulder_angle=shoulder_angle,
        head_angle=head_angle if head_angle is not None else shoulder_angle,
        speed=speed, vx=vx, vy=vy, shoulder_width=shoulder_width,
    )


@pytest.fixture
def trivial_frame():
    """1 attacker (team 1) at (-5, 0) facing +x, 1 defender (team 0) at
    (5, 0) facing +x. Defender can't see anything behind him."""
    return _make_frame([
        _full_player(1, 10, -5.0, 0.0, shoulder_angle=0.0),
        _full_player(0, 5,  5.0,  0.0, shoulder_angle=0.0),
    ])


@pytest.fixture
def blind_spot_frame():
    """Attacker at (0, 5) in defender's blind spot. Defender at (0, 0)
    facing +x, attacker directly to his left (+y = 90 degrees)."""
    return _make_frame([
        _full_player(1, 9, 0.0, 5.0, shoulder_angle=np.pi / 2),
        _full_player(0, 4, 0.0, 0.0, shoulder_angle=0.0),
    ])


# ── TestClassifyAttackAngle ─────────────────────────────────────────────

class TestClassifyAttackAngle:
    def test_forward(self):
        assert classify_attack_angle(0.0) == "forward"
        assert classify_attack_angle(np.radians(10)) == "forward"
        assert classify_attack_angle(np.radians(22)) == "forward"

    def test_diagonal(self):
        assert classify_attack_angle(np.radians(30)) == "diagonal"
        assert classify_attack_angle(np.radians(45)) == "diagonal"
        assert classify_attack_angle(np.radians(60)) == "diagonal"

    def test_sideways(self):
        assert classify_attack_angle(np.radians(70)) == "sideways"
        assert classify_attack_angle(np.radians(90)) == "sideways"

    def test_boundary_forward_diagonal(self):
        assert classify_attack_angle(np.radians(22.4)) == "forward"
        assert classify_attack_angle(np.radians(22.6)) == "diagonal"

    def test_boundary_diagonal_sideways(self):
        assert classify_attack_angle(np.radians(67.4)) == "diagonal"
        assert classify_attack_angle(np.radians(67.6)) == "sideways"


# ── TestVisionGrids ─────────────────────────────────────────────────────

class TestVisionGrids:
    def test_builds_grids_for_team(self, trivial_frame):
        grids = _build_vision_grids(trivial_frame, team_id=0, smoothing=1.0)
        assert len(grids) == 1  # 1 defender in team 0

    def test_empty_team_returns_empty(self, trivial_frame):
        grids = _build_vision_grids(trivial_frame, team_id=99, smoothing=1.0)
        assert len(grids) == 0

    def test_query_returns_correct_shape(self, trivial_frame):
        grids = _build_vision_grids(trivial_frame, team_id=0, smoothing=1.0)
        points = np.array([[0.0, 0.0], [5.0, 5.0], [-10.0, 3.0]])
        V = _query_vision(grids, points)
        assert V.shape == (1, 3)

    def test_vision_in_front_higher_than_behind(self, trivial_frame):
        """Defender at (5, 0) facing +x. Point at (10, 0) is in front,
        point at (0, 0) is behind."""
        grids = _build_vision_grids(trivial_frame, team_id=0, smoothing=1.0)
        points = np.array([[10.0, 0.0], [0.0, 0.0]])
        V = _query_vision(grids, points)
        assert V[0, 0] > V[0, 1]  # front > behind

    def test_vision_values_in_unit_interval(self, trivial_frame):
        grids = _build_vision_grids(trivial_frame, team_id=0, smoothing=1.0)
        points = np.array([[0.0, 0.0], [10.0, 0.0], [-30.0, 20.0]])
        V = _query_vision(grids, points)
        assert np.all(V >= 0.0)
        assert np.all(V <= 1.0)


# ── TestAwareness ────────────────────────────────────────────────────────

class TestAwareness:
    def test_fully_visible_gives_high_awareness(self):
        aw = _compute_awareness(
            V_attacker=np.array([1.0]),
            V_passer=np.array([1.0]),
        )
        assert aw[0] > 0.9

    def test_fully_blind_gives_zero(self):
        aw = _compute_awareness(
            V_attacker=np.array([0.0]),
            V_passer=np.array([0.0]),
        )
        assert aw[0] == 0.0

    def test_attacker_invisible_dominates(self):
        """Even if passer is visible, not seeing attacker → low awareness."""
        aw = _compute_awareness(
            V_attacker=np.array([0.0]),
            V_passer=np.array([1.0]),
        )
        assert aw[0] < 0.4  # low despite seeing passer

    def test_passer_invisible_still_decent_if_attacker_visible(self):
        """Seeing attacker but not passer → decent awareness (you know
        where the threat is, just don't know when ball comes)."""
        aw = _compute_awareness(
            V_attacker=np.array([1.0]),
            V_passer=np.array([0.0]),
        )
        assert aw[0] > 0.5

    def test_speed_penalty_reduces_awareness(self):
        aw_static = _compute_awareness(
            np.array([0.8]), np.array([0.8]), attacker_speed=0.0)
        aw_sprint = _compute_awareness(
            np.array([0.8]), np.array([0.8]), attacker_speed=8.0)
        assert aw_sprint < aw_static

    def test_diagonal_penalty_reduces_awareness(self):
        aw_straight = _compute_awareness(
            np.array([0.5]), np.array([0.5]),
            attacker_angle_from_fwd=np.array([0.0]))  # forward = no penalty
        aw_diag = _compute_awareness(
            np.array([0.5]), np.array([0.5]),
            attacker_angle_from_fwd=np.array([np.pi / 4]))  # 45° = max penalty
        assert aw_diag < aw_straight

    def test_awareness_in_unit_interval(self):
        rng = np.random.default_rng(42)
        V_a = rng.uniform(0, 1, 20)
        V_p = rng.uniform(0, 1, 20)
        aw = _compute_awareness(V_a, V_p, attacker_speed=5.0 * np.ones(20),
                                attacker_angle_from_fwd=np.full(20, np.pi / 4))
        assert np.all(aw >= 0.0) and np.all(aw <= 1.0)


# ── TestDetectionDelay ───────────────────────────────────────────────────

class TestDetectionDelay:
    def test_zero_delay_when_fully_aware(self):
        p = default_params()
        d = _detection_delay(
            theta_to_threat=np.array([np.pi]),
            awareness=np.array([1.0]),
            params=p,
        )
        assert d[0] == 0.0

    def test_max_delay_when_fully_blind(self):
        p = default_params()
        d = _detection_delay(
            theta_to_threat=np.array([np.pi]),
            awareness=np.array([0.0]),
            params=p,
        )
        assert d[0] > 0.3  # significant extra delay

    def test_delay_zero_when_threat_in_front(self):
        """Even if awareness is 0, theta=0 means threat is in front →
        rt(0) ≈ rt_base → extra = rt-rt_base ≈ 0."""
        p = default_params()
        d = _detection_delay(
            theta_to_threat=np.array([0.0]),
            awareness=np.array([0.0]),
            params=p,
        )
        assert d[0] < 0.01  # negligible

    def test_delay_monotonic_with_theta(self):
        p = default_params()
        thetas = np.linspace(0, np.pi, 20)
        d = _detection_delay(thetas, np.zeros(20), p)
        assert np.all(np.diff(d) >= 0)

    def test_delay_nonnegative(self):
        p = default_params()
        rng = np.random.default_rng(42)
        d = _detection_delay(
            rng.uniform(0, np.pi, 50),
            rng.uniform(0, 1, 50),
            p,
        )
        assert np.all(d >= 0.0)


# ── TestComputeDosSurface ────────────────────────────────────────────────

class TestComputeDosSurface:
    def test_returns_correct_shapes(self, trivial_frame):
        dos, base, bdir, xg, yg = compute_dos_surface(
            trivial_frame, attacking_team=1, ball_xy=(-5, 0),
            attacking_right=True, n_grid_x=20, n_directions=8,
            vision_smoothing=1.0,
        )
        n_y = int(round(20 * 68.0 / 105.0))
        assert dos.shape == (n_y, 20)
        assert base.shape == (n_y, 20)
        assert bdir.shape == (n_y, 20)
        assert xg.shape == (20,)
        assert yg.shape == (n_y,)

    def test_dos_values_bounded(self, trivial_frame):
        dos, _, _, _, _ = compute_dos_surface(
            trivial_frame, attacking_team=1, ball_xy=(-5, 0),
            attacking_right=True, n_grid_x=20, n_directions=8,
            vision_smoothing=1.0,
        )
        assert np.all(dos >= -1.0)
        assert np.all(dos <= 1.0)

    def test_dos_nonnegative_simple_case(self, trivial_frame):
        """In a trivial 1v1 case, diagonal delivery can only help or be
        neutral — never worse than forward."""
        dos, _, _, _, _ = compute_dos_surface(
            trivial_frame, attacking_team=1, ball_xy=(-5, 0),
            attacking_right=True, n_grid_x=20, n_directions=8,
            vision_smoothing=1.0,
        )
        assert np.all(dos >= -0.01)  # allow tiny float noise

    def test_dos_has_nonzero_values(self, trivial_frame):
        """DOS should be nonzero somewhere — diagonal must provide
        some advantage over forward in this setup."""
        dos, _, _, _, _ = compute_dos_surface(
            trivial_frame, attacking_team=1, ball_xy=(-5, 0),
            attacking_right=True, n_grid_x=20, n_directions=8,
            vision_smoothing=1.0,
        )
        assert dos.max() > 0.001

    def test_baseline_ppcf_valid(self, trivial_frame):
        _, base, _, _, _ = compute_dos_surface(
            trivial_frame, attacking_team=1, ball_xy=(-5, 0),
            attacking_right=True, n_grid_x=20, n_directions=8,
            vision_smoothing=1.0,
        )
        assert np.all(base >= 0.0) and np.all(base <= 1.0)

    def test_blind_spot_has_higher_dos(self):
        """Defender facing +x with attacker behind him at (-5, 0).
        Ball at (20, 0) far right. Diagonal delivery from far right
        toward (-5, 0) passes through defender's blind spot."""
        frame = _make_frame([
            _full_player(1, 9, -5.0, 5.0, shoulder_angle=-np.pi/4),
            _full_player(0, 4, 5.0, 0.0, shoulder_angle=0.0),
        ])
        dos, _, _, xg, yg = compute_dos_surface(
            frame, attacking_team=1, ball_xy=(-20, -10),
            attacking_right=True, n_grid_x=20, n_directions=12,
            vision_smoothing=1.0,
        )
        # DOS should be positive somewhere (diagonal exploits blind spot)
        assert dos.max() > 0


# ── TestDosAtTargets ─────────────────────────────────────────────────────

class TestDosAtTargets:
    def test_returns_correct_shapes(self, trivial_frame):
        dos, base, bdir = dos_at_targets(
            trivial_frame,
            np.array([[0.0, 0.0], [5.0, 5.0], [-10.0, 3.0]]),
            attacking_team=1, ball_xy=(-5, 0), attacking_right=True,
            n_directions=8, vision_smoothing=1.0,
        )
        assert dos.shape == (3,)
        assert base.shape == (3,)
        assert bdir.shape == (3,)

    def test_accepts_1d_input(self, trivial_frame):
        dos, base, bdir = dos_at_targets(
            trivial_frame, np.array([0.0, 0.0]),
            attacking_team=1, ball_xy=(-5, 0), attacking_right=True,
            n_directions=8, vision_smoothing=1.0,
        )
        assert dos.shape == (1,)

    def test_consistent_with_surface(self, trivial_frame):
        """Point-wise DOS should approximately match grid DOS at same cell."""
        dos_grid, _, _, xg, yg = compute_dos_surface(
            trivial_frame, attacking_team=1, ball_xy=(-5, 0),
            attacking_right=True, n_grid_x=20, n_directions=8,
            vision_smoothing=1.0,
        )
        ix, iy = 10, 6
        dos_pt, _, _ = dos_at_targets(
            trivial_frame, np.array([[xg[ix], yg[iy]]]),
            attacking_team=1, ball_xy=(-5, 0), attacking_right=True,
            n_directions=8, vision_smoothing=1.0,
        )
        assert abs(dos_pt[0] - dos_grid[iy, ix]) < 0.01


# ── TestDosForEvent ──────────────────────────────────────────────────────

class TestDosForEvent:
    def test_returns_expected_keys(self, trivial_frame):
        result = dos_for_event(
            trivial_frame,
            dict(x=-5, y=0, x_receiver=0, y_receiver=5, event_type="pass"),
            attacking_team=1, attacking_right=True, vision_smoothing=1.0,
        )
        expected_keys = {
            "dos", "ppcf_att_baseline", "ppcf_att_with_awareness",
            "direction_class", "event_direction", "awareness_mean",
            "detection_delay_mean",
        }
        assert set(result.keys()) == expected_keys

    def test_diagonal_pass_classified_correctly(self, trivial_frame):
        result = dos_for_event(
            trivial_frame,
            dict(x=-5, y=0, x_receiver=0, y_receiver=5, event_type="pass"),
            attacking_team=1, attacking_right=True, vision_smoothing=1.0,
        )
        assert result["direction_class"] == "diagonal"

    def test_forward_pass_classified_correctly(self, trivial_frame):
        result = dos_for_event(
            trivial_frame,
            dict(x=-5, y=0, x_receiver=5, y_receiver=0, event_type="pass"),
            attacking_team=1, attacking_right=True, vision_smoothing=1.0,
        )
        assert result["direction_class"] == "forward"

    def test_carry_event_works(self, trivial_frame):
        result = dos_for_event(
            trivial_frame,
            dict(x=-5, y=0, carry_end_x=0, carry_end_y=5, event_type="carry"),
            attacking_team=1, attacking_right=True, vision_smoothing=1.0,
        )
        assert result["direction_class"] == "diagonal"
        assert "dos" in result

    def test_dos_value_bounded(self, trivial_frame):
        result = dos_for_event(
            trivial_frame,
            dict(x=-5, y=0, x_receiver=0, y_receiver=5, event_type="pass"),
            attacking_team=1, attacking_right=True, vision_smoothing=1.0,
        )
        assert -1.0 <= result["dos"] <= 1.0
        assert 0.0 <= result["ppcf_att_baseline"] <= 1.0
        assert 0.0 <= result["ppcf_att_with_awareness"] <= 1.0

    def test_awareness_mean_in_unit_interval(self, trivial_frame):
        result = dos_for_event(
            trivial_frame,
            dict(x=-5, y=0, x_receiver=0, y_receiver=5, event_type="pass"),
            attacking_team=1, attacking_right=True, vision_smoothing=1.0,
        )
        assert 0.0 <= result["awareness_mean"] <= 1.0

    def test_detection_delay_nonnegative(self, trivial_frame):
        result = dos_for_event(
            trivial_frame,
            dict(x=-5, y=0, x_receiver=0, y_receiver=5, event_type="pass"),
            attacking_team=1, attacking_right=True, vision_smoothing=1.0,
        )
        assert result["detection_delay_mean"] >= 0.0


# ── TestDeterminism ──────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_input_same_output(self, trivial_frame):
        kw = dict(attacking_team=1, ball_xy=(-5, 0), attacking_right=True,
                  n_grid_x=15, n_directions=8, vision_smoothing=1.0)
        dos1, _, _, _, _ = compute_dos_surface(trivial_frame, **kw)
        dos2, _, _, _, _ = compute_dos_surface(trivial_frame, **kw)
        assert np.array_equal(dos1, dos2)


# ── TestEmptyFrames ──────────────────────────────────────────────────────

class TestEmptyFrames:
    def test_no_defenders(self):
        frame = _make_frame([_full_player(1, 9, 0.0, 0.0)])
        dos, base, _, _, _ = compute_dos_surface(
            frame, attacking_team=1, ball_xy=(0, 0),
            attacking_right=True, n_grid_x=15, n_directions=8,
            vision_smoothing=1.0,
        )
        # No defenders → no detection delay → DOS = 0 everywhere
        assert np.allclose(dos, 0.0, atol=1e-9)

    def test_no_attackers(self):
        frame = _make_frame([_full_player(0, 4, 0.0, 0.0)])
        dos, base, _, _, _ = compute_dos_surface(
            frame, attacking_team=1, ball_xy=(0, 0),
            attacking_right=True, n_grid_x=15, n_directions=8,
            vision_smoothing=1.0,
        )
        assert np.allclose(base, 0.0, atol=1e-9)
