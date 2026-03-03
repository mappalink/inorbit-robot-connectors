# SPDX-FileCopyrightText: 2025 InOrbit, Inc.
#
# SPDX-License-Identifier: MIT

"""Tests for the MiR mission translator.

Covers:
- Step grouping (consecutive waypoints → single native mission)
- Frame conversion (map frame → native frame via inverse transform)
- Inverse transform math (SpatialTransform)
- Edge cases (empty missions, single waypoint, all non-waypoint, etc.)
"""

import math
import pytest

from inorbit_edge_executor.datatypes import (
    MissionDefinition,
    MissionStepPoseWaypoint,
    MissionStepWait,
    MissionStepRunAction,
    MissionStepSetData,
    Pose,
)
from inorbit_edge_executor.mission import Mission

from inorbit_mir_connector.src.mission.translator import (
    SpatialTransform,
    InOrbitToMirTranslator,
)
from inorbit_mir_connector.src.mission.datatypes import (
    MissionStepExecuteMirNativeMission,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pose_waypoint(x, y, theta=0.0, frame_id="map", label=None):
    """Create a MissionStepPoseWaypoint for testing."""
    return MissionStepPoseWaypoint(
        label=label or f"WP ({x}, {y})",
        waypoint=Pose(x=x, y=y, theta=theta, frameId=frame_id, waypointId=f"wp-{x}-{y}"),
    )


def _make_wait(secs=5.0):
    return MissionStepWait(label=f"Wait {secs}s", timeoutSecs=secs)


def _make_set_data(data=None):
    return MissionStepSetData(label="Set data", data=data or {"key": "value"})


def _make_run_action(action_id="someAction"):
    return MissionStepRunAction(
        label=f"Run {action_id}",
        runAction={"actionId": action_id, "arguments": {}},
    )


def _make_mission(steps, mission_id="test-mission", robot_id="mir-1"):
    definition = MissionDefinition(label="Test mission", steps=steps)
    return Mission(id=mission_id, robot_id=robot_id, definition=definition, arguments={})


# ---------------------------------------------------------------------------
# SpatialTransform tests
# ---------------------------------------------------------------------------

class TestSpatialTransform:
    """Test inverse affine math."""

    def test_identity_transform(self):
        """Identity matrix → no coordinate change."""
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        t = SpatialTransform(identity)
        x, y, theta = t.map_to_native(5.0, 3.0, 1.0)
        assert x == pytest.approx(5.0)
        assert y == pytest.approx(3.0)
        assert theta == pytest.approx(1.0)

    def test_pure_translation(self):
        """Translation-only matrix → subtract offset."""
        # Forward: native + (10, 20) = map
        # Inverse: map - (10, 20) = native
        matrix = [[1, 0, 10], [0, 1, 20], [0, 0, 1]]
        t = SpatialTransform(matrix)
        x, y, theta = t.map_to_native(15.0, 25.0, 0.5)
        assert x == pytest.approx(5.0)
        assert y == pytest.approx(5.0)
        assert theta == pytest.approx(0.5)  # no rotation → theta unchanged

    def test_pure_rotation_90deg(self):
        """90° rotation matrix → inverse rotates back."""
        # Forward: rotate 90° CCW
        # cos(90°) = 0, sin(90°) = 1
        # R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        matrix = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        t = SpatialTransform(matrix)

        # A point at (1, 0) in map frame
        # Forward: (x_native, y_native) → (−y_native, x_native) in map
        # Inverse: (x_map, y_map) → R^T @ (x_map, y_map)
        # R^T = [[0, 1], [-1, 0]], so (1, 0) → (0, -1) in native
        x, y, theta = t.map_to_native(1.0, 0.0, math.pi / 2)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(-1.0, abs=1e-10)
        assert theta == pytest.approx(0.0, abs=1e-10)  # π/2 - π/2 = 0

    def test_rotation_plus_translation(self):
        """Combined rotation + translation."""
        # Forward: 45° rotation + translation (5, 3)
        angle = math.pi / 4
        c, s = math.cos(angle), math.sin(angle)
        matrix = [[c, -s, 5], [s, c, 3], [0, 0, 1]]
        t = SpatialTransform(matrix)

        # Round-trip: apply forward then inverse should give back original
        # Forward: (x_n, y_n) → (c*xn - s*yn + 5, s*xn + c*yn + 3)
        xn, yn = 2.0, 1.0
        xm = c * xn - s * yn + 5
        ym = s * xn + c * yn + 3
        theta_n = 0.3
        theta_m = theta_n + angle

        xr, yr, tr = t.map_to_native(xm, ym, theta_m)
        assert xr == pytest.approx(xn, abs=1e-10)
        assert yr == pytest.approx(yn, abs=1e-10)
        assert tr == pytest.approx(theta_n, abs=1e-10)

    def test_negative_rotation(self):
        """Negative rotation angle."""
        angle = -math.pi / 6  # -30°
        c, s = math.cos(angle), math.sin(angle)
        matrix = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        t = SpatialTransform(matrix)

        # Point (1, 0) in native → rotated to map → inverse should give (1, 0) back
        xm = c * 1.0
        ym = s * 1.0
        x, y, theta = t.map_to_native(xm, ym, angle)
        assert x == pytest.approx(1.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)
        assert theta == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Translator grouping tests
# ---------------------------------------------------------------------------

class TestTranslatorGrouping:
    """Test that consecutive waypoint steps are grouped correctly."""

    def test_all_waypoints_grouped(self):
        """All waypoint steps → single MirNativeMission."""
        steps = [
            _make_pose_waypoint(1.0, 2.0, frame_id="native-map-guid"),
            _make_pose_waypoint(3.0, 4.0, frame_id="native-map-guid"),
            _make_pose_waypoint(5.0, 6.0, frame_id="native-map-guid"),
        ]
        mission = _make_mission(steps)
        result = InOrbitToMirTranslator.translate(mission)

        assert len(result.definition.steps) == 1
        step = result.definition.steps[0]
        assert isinstance(step, MissionStepExecuteMirNativeMission)
        assert len(step.waypoints) == 3
        assert step.waypoints[0].x == 1.0
        assert step.waypoints[1].x == 3.0
        assert step.waypoints[2].x == 5.0
        assert step.label == "Navigate 3 waypoints"

    def test_waypoints_split_by_wait(self):
        """Wait step splits waypoint groups."""
        steps = [
            _make_pose_waypoint(1.0, 2.0, frame_id="guid"),
            _make_pose_waypoint(3.0, 4.0, frame_id="guid"),
            _make_wait(5.0),
            _make_pose_waypoint(5.0, 6.0, frame_id="guid"),
            _make_pose_waypoint(7.0, 8.0, frame_id="guid"),
        ]
        mission = _make_mission(steps)
        result = InOrbitToMirTranslator.translate(mission)

        assert len(result.definition.steps) == 3
        assert isinstance(result.definition.steps[0], MissionStepExecuteMirNativeMission)
        assert len(result.definition.steps[0].waypoints) == 2
        assert isinstance(result.definition.steps[1], MissionStepWait)
        assert isinstance(result.definition.steps[2], MissionStepExecuteMirNativeMission)
        assert len(result.definition.steps[2].waypoints) == 2

    def test_waypoints_split_by_run_action(self):
        """RunAction step splits waypoint groups."""
        steps = [
            _make_pose_waypoint(1.0, 2.0, frame_id="guid"),
            _make_run_action("pickUp"),
            _make_pose_waypoint(3.0, 4.0, frame_id="guid"),
        ]
        mission = _make_mission(steps)
        result = InOrbitToMirTranslator.translate(mission)

        assert len(result.definition.steps) == 3
        assert isinstance(result.definition.steps[0], MissionStepExecuteMirNativeMission)
        assert len(result.definition.steps[0].waypoints) == 1
        assert isinstance(result.definition.steps[1], MissionStepRunAction)
        assert isinstance(result.definition.steps[2], MissionStepExecuteMirNativeMission)
        assert len(result.definition.steps[2].waypoints) == 1

    def test_no_waypoints_passthrough(self):
        """All non-waypoint steps pass through unchanged."""
        steps = [
            _make_wait(3.0),
            _make_set_data({"key": "val"}),
            _make_run_action("doThing"),
        ]
        mission = _make_mission(steps)
        result = InOrbitToMirTranslator.translate(mission)

        assert len(result.definition.steps) == 3
        assert isinstance(result.definition.steps[0], MissionStepWait)
        assert isinstance(result.definition.steps[1], MissionStepSetData)
        assert isinstance(result.definition.steps[2], MissionStepRunAction)

    def test_single_waypoint(self):
        """Single waypoint → single MirNativeMission with 1 waypoint."""
        steps = [_make_pose_waypoint(10.0, 20.0, frame_id="guid")]
        mission = _make_mission(steps)
        result = InOrbitToMirTranslator.translate(mission)

        assert len(result.definition.steps) == 1
        step = result.definition.steps[0]
        assert isinstance(step, MissionStepExecuteMirNativeMission)
        assert len(step.waypoints) == 1
        assert step.waypoints[0].x == 10.0
        assert step.waypoints[0].y == 20.0

    def test_empty_mission_raises(self):
        """Empty mission raises ValueError."""
        mission = _make_mission([])
        with pytest.raises(ValueError, match="no steps"):
            InOrbitToMirTranslator.translate(mission)

    def test_mission_metadata_preserved(self):
        """Mission ID, robot_id, arguments are preserved."""
        steps = [_make_pose_waypoint(1.0, 2.0, frame_id="guid")]
        mission = _make_mission(steps, mission_id="m-42", robot_id="mir-500")
        result = InOrbitToMirTranslator.translate(mission)

        assert result.id == "m-42"
        assert result.robot_id == "mir-500"
        assert result.definition.label == "Test mission"

    def test_robot_id_on_compiled_step(self):
        """Compiled step carries the robot_id."""
        steps = [_make_pose_waypoint(1.0, 2.0, frame_id="guid")]
        mission = _make_mission(steps, robot_id="mir-200")
        result = InOrbitToMirTranslator.translate(mission)

        step = result.definition.steps[0]
        assert isinstance(step, MissionStepExecuteMirNativeMission)
        assert step.robot_id == "mir-200"


# ---------------------------------------------------------------------------
# Frame conversion tests
# ---------------------------------------------------------------------------

class TestTranslatorFrameConversion:
    """Test coordinate frame handling during translation."""

    def test_native_frame_passthrough(self):
        """Native frame waypoints pass through without transform."""
        steps = [
            _make_pose_waypoint(10.0, 20.0, theta=1.5, frame_id="native-guid"),
        ]
        mission = _make_mission(steps)
        # Provide a transform, but frame_id != "map" so it won't be applied
        transform = SpatialTransform([[1, 0, 100], [0, 1, 200], [0, 0, 1]])
        result = InOrbitToMirTranslator.translate(mission, spatial_transform=transform)

        wp = result.definition.steps[0].waypoints[0]
        assert wp.x == pytest.approx(10.0)
        assert wp.y == pytest.approx(20.0)
        assert wp.orientation == pytest.approx(math.degrees(1.5))

    def test_map_frame_transformed(self):
        """Map frame waypoints are inverse-transformed to native."""
        # Forward transform: translate by (10, 20)
        transform = SpatialTransform([[1, 0, 10], [0, 1, 20], [0, 0, 1]])
        steps = [
            _make_pose_waypoint(15.0, 25.0, theta=0.5, frame_id="map"),
        ]
        mission = _make_mission(steps)
        result = InOrbitToMirTranslator.translate(mission, spatial_transform=transform)

        wp = result.definition.steps[0].waypoints[0]
        assert wp.x == pytest.approx(5.0)
        assert wp.y == pytest.approx(5.0)
        assert wp.orientation == pytest.approx(math.degrees(0.5))

    def test_map_frame_without_transform_passthrough(self):
        """Map frame waypoints without a transform pass through as-is."""
        steps = [_make_pose_waypoint(15.0, 25.0, theta=0.5, frame_id="map")]
        mission = _make_mission(steps)
        result = InOrbitToMirTranslator.translate(mission, spatial_transform=None)

        wp = result.definition.steps[0].waypoints[0]
        assert wp.x == pytest.approx(15.0)
        assert wp.y == pytest.approx(25.0)

    def test_mixed_frames_in_mission(self):
        """Waypoints with different frames are handled independently."""
        transform = SpatialTransform([[1, 0, 10], [0, 1, 20], [0, 0, 1]])
        steps = [
            _make_pose_waypoint(15.0, 25.0, frame_id="map"),       # → (5, 5)
            _make_pose_waypoint(1.0, 2.0, frame_id="native-guid"),  # → (1, 2)
        ]
        mission = _make_mission(steps)
        result = InOrbitToMirTranslator.translate(mission, spatial_transform=transform)

        # Both are in the same group since they're consecutive waypoints
        step = result.definition.steps[0]
        assert isinstance(step, MissionStepExecuteMirNativeMission)
        assert len(step.waypoints) == 2
        assert step.waypoints[0].x == pytest.approx(5.0)
        assert step.waypoints[1].x == pytest.approx(1.0)

    def test_theta_converted_to_degrees(self):
        """Theta in radians is converted to MiR degrees."""
        steps = [_make_pose_waypoint(0, 0, theta=math.pi / 2, frame_id="guid")]
        mission = _make_mission(steps)
        result = InOrbitToMirTranslator.translate(mission)

        wp = result.definition.steps[0].waypoints[0]
        assert wp.orientation == pytest.approx(90.0)

    def test_rotated_transform_applied(self):
        """A rotated SpatialTransform correctly inverse-transforms."""
        angle = math.pi / 4  # 45°
        c, s = math.cos(angle), math.sin(angle)
        transform = SpatialTransform([[c, -s, 5], [s, c, 3], [0, 0, 1]])

        # Known native point: (2, 1)
        # Forward: map_x = c*2 - s*1 + 5, map_y = s*2 + c*1 + 3
        map_x = c * 2 - s * 1 + 5
        map_y = s * 2 + c * 1 + 3
        theta_native = 0.3
        theta_map = theta_native + angle

        steps = [_make_pose_waypoint(map_x, map_y, theta=theta_map, frame_id="map")]
        mission = _make_mission(steps)
        result = InOrbitToMirTranslator.translate(mission, spatial_transform=transform)

        wp = result.definition.steps[0].waypoints[0]
        assert wp.x == pytest.approx(2.0, abs=1e-9)
        assert wp.y == pytest.approx(1.0, abs=1e-9)
        assert wp.orientation == pytest.approx(math.degrees(theta_native), abs=1e-9)


# ---------------------------------------------------------------------------
# Complex scenario tests
# ---------------------------------------------------------------------------

class TestTranslatorComplexScenarios:
    """Test more complex mission patterns."""

    def test_alternating_waypoints_and_actions(self):
        """wp, action, wp, wait, wp, action → 3 groups + 2 passthrough."""
        steps = [
            _make_pose_waypoint(1, 1, frame_id="g"),
            _make_run_action("a1"),
            _make_pose_waypoint(2, 2, frame_id="g"),
            _make_wait(2.0),
            _make_pose_waypoint(3, 3, frame_id="g"),
            _make_run_action("a2"),
        ]
        mission = _make_mission(steps)
        result = InOrbitToMirTranslator.translate(mission)

        assert len(result.definition.steps) == 6
        assert isinstance(result.definition.steps[0], MissionStepExecuteMirNativeMission)
        assert isinstance(result.definition.steps[1], MissionStepRunAction)
        assert isinstance(result.definition.steps[2], MissionStepExecuteMirNativeMission)
        assert isinstance(result.definition.steps[3], MissionStepWait)
        assert isinstance(result.definition.steps[4], MissionStepExecuteMirNativeMission)
        assert isinstance(result.definition.steps[5], MissionStepRunAction)

    def test_leading_non_waypoint_steps(self):
        """Non-waypoint steps before first waypoint are preserved."""
        steps = [
            _make_set_data({"mode": "fast"}),
            _make_wait(1.0),
            _make_pose_waypoint(1, 2, frame_id="g"),
            _make_pose_waypoint(3, 4, frame_id="g"),
        ]
        mission = _make_mission(steps)
        result = InOrbitToMirTranslator.translate(mission)

        assert len(result.definition.steps) == 3
        assert isinstance(result.definition.steps[0], MissionStepSetData)
        assert isinstance(result.definition.steps[1], MissionStepWait)
        assert isinstance(result.definition.steps[2], MissionStepExecuteMirNativeMission)
        assert len(result.definition.steps[2].waypoints) == 2
