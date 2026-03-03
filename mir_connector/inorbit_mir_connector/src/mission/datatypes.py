# SPDX-FileCopyrightText: 2025 InOrbit, Inc.
#
# SPDX-License-Identifier: MIT

"""MiR-specific mission datatypes for mission translation.

Defines custom step types and mission classes used when consecutive
waypoint steps are compiled into a single native MiR mission.
"""

from __future__ import annotations

from typing import List, Optional, Union, override

from pydantic import Field

from inorbit_edge_executor.datatypes import (
    MissionDefinition,
    MissionStep,
    MissionStepPoseWaypoint,
    MissionStepRunAction,
    MissionStepSetData,
    MissionStepWait,
    MissionStepWaitUntil,
)
from inorbit_edge_executor.mission import Mission


class MirWaypoint(MissionStep):
    """A single waypoint extracted from a MissionStepPoseWaypoint.

    Carries the MiR-native coordinates (x, y, orientation in degrees)
    ready to be sent as a ``move_to_position`` action.
    """

    x: float = Field(description="X coordinate in MiR native frame (meters)")
    y: float = Field(description="Y coordinate in MiR native frame (meters)")
    orientation: float = Field(description="Orientation in degrees (MiR convention)")


class MissionStepExecuteMirNativeMission(MissionStep):
    """Custom mission step that executes a compiled native MiR mission.

    Produced by the translator when consecutive waypoint steps are grouped.
    The behavior tree node will create a MiR mission definition, add one
    ``move_to_position`` action per waypoint, and queue it.
    """

    waypoints: List[MirWaypoint] = Field(
        description="Ordered waypoints for the native MiR mission"
    )
    robot_id: str = Field(description="InOrbit robot ID")

    @override
    def accept(self, visitor):
        """Visitor pattern for behavior tree construction."""
        if hasattr(visitor, "visit_execute_mir_native_mission"):
            return visitor.visit_execute_mir_native_mission(self)
        if hasattr(visitor, "collect_step"):
            return visitor.collect_step(self)
        return None


# Type alias for MiR-specific steps list
MirStepsList = List[
    Union[
        MissionStepSetData,
        MissionStepPoseWaypoint,
        MissionStepRunAction,
        MissionStepWait,
        MissionStepWaitUntil,
        MissionStepExecuteMirNativeMission,
    ]
]


class MissionDefinitionMir(MissionDefinition):
    """Mission definition that supports MiR-specific step types.

    Extends the base MissionDefinition to include
    MissionStepExecuteMirNativeMission which is produced during translation.
    """

    steps: MirStepsList  # type: ignore[assignment]


class MirInOrbitMission(Mission):
    """Mission subclass for MiR that uses MiR-specific definition.

    Used after translation to hold the converted mission with
    MiR-specific step types (compiled native missions).
    """

    definition: MissionDefinitionMir  # type: ignore[assignment]
