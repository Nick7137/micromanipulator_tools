# THIS IS ALL TERRIBLE CODE BUT I WAS UNDER TIME PRESSURE THIS ALL NEEDS REDOING ESSENTIALLY - BUT IT WORKS

import os
import sys
import time
import math
from typing import Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micromanipulator_tools import NanoControl, Turntable, Vision


SPEED_PRESETS = {
    1: {
        "base_joint": "c16",
        "elbow_joint": "c16",
        "prismatic_joint": "c64",
        "tweezer_joint": "c64",
    },
}

# Note: life-hack type `mode` in powershell to get the COM ports

# This is the time drive elbow executes to get to the default height
# without holding a rock
RISE_TIME_NO_ROCK_SEC = 3

TOL_PX = 2
ROCK_POS_TOL_ANGLE = 0.5
TWEEZER_MOVE_TIME_SEC = 6
TIME_TO_DISK_FROM_LEVEL_SEC = 3.5
DROP_OFF_ANGLE_THRESHOLD_DEG = 26
DROP_OFF_EXTRA_TIME_SEC = 3

# -----------------------------------------------------------------------------
# Setup Functions
# -----------------------------------------------------------------------------


def set_robot_speed(nc: NanoControl):
    active_speed_profile = 1
    nc.set_speed_profile(active_speed_profile, SPEED_PRESETS[1])
    nc.change_speed_profile_to(active_speed_profile)


# -----------------------------------------------------------------------------
# Detection Functions
# -----------------------------------------------------------------------------


def get_disk_centre(vis: Vision):
    # Extract disk center in pixels
    disk_data = vis.detect_disk()
    disk_center, disk_radius = disk_data
    disk_center_x, disk_center_y = disk_center

    return disk_center_x, disk_center_y


def get_workspace_arc_radius(vis: Vision):
    # Extract end effector arc radius
    workspace_arcs = vis.detect_workspace_arcs()
    end_effector_radius = workspace_arcs["end_effector_arc"]["radius"]
    return end_effector_radius


def get_current_robot_radius_px(vis: Vision):
    polar_coords, centroid_px, body_contour = vis.detect_robot()
    radius, angle = polar_coords
    return radius


def get_current_robot_theta_deg(vis: Vision):
    polar_coords, centroid_px, body_contour = vis.detect_robot()
    radius, angle = polar_coords
    return angle


def is_rock_available(vis: Vision) -> bool:
    """
    Check if there are any rocks available for pickup.
    Returns True if rocks are detected, False otherwise.
    """

    rocks_data = vis.detect_rocks()
    return rocks_data and len(rocks_data) > 0


def get_first_rock_coords(vis: Vision):
    # Extract first rock coordinates in pixels
    rocks_data = vis.detect_rocks()
    if rocks_data and len(rocks_data) > 0:
        first_rock = rocks_data[0]
        first_rock_centroid = first_rock["pixel_centroid"]
        first_rock_x, first_rock_y = first_rock_centroid

        print(f"First rock at: ({first_rock_x}, {first_rock_y}) pixels")
    else:
        print("No rocks detected")
        first_rock_x = None
        first_rock_y = None

    return first_rock_x, first_rock_y


def get_disk_rotation_angle(
    rock_position: Tuple[float, float],
    disk_center: Tuple[float, float],
    end_effector_radius: float,
) -> Tuple[float, bool]:
    """
    Calculate disk rotation angle using the second geometric solution.
    Returns (angle, reverse) where reverse=True means rotate counterclockwise.
    """

    # Vector from disk center to rock
    dx = rock_position[0] - disk_center[0]
    dy = rock_position[1] - disk_center[1]

    # Distance from disk center to rock
    distance_to_rock = math.hypot(dx, dy)

    # Current angle of rock relative to disk center
    rock_angle = math.atan2(dy, dx)

    # Compute geometric parameter
    S = distance_to_rock / (2 * end_effector_radius)

    if abs(S) > 1:
        raise ValueError(
            f"Rock unreachable: distance {distance_to_rock:.1f} > maximum reach {2 * end_effector_radius:.1f}"
        )

    # Use only the second solution
    arcsin_S = math.asin(S)
    rotation = math.pi - arcsin_S - rock_angle

    # Normalize to [0, 2π) and convert to degrees
    rotation_deg = (rotation % (2 * math.pi)) * 180 / math.pi

    # If rotation is more than 180°, go the other way
    if rotation_deg > 180:
        # Calculate the opposite direction
        rotation_deg = 360 - rotation_deg
        reverse = True  # Rotate counterclockwise
    else:
        reverse = False  # Rotate clockwise

    return rotation_deg, reverse


# -----------------------------------------------------------------------------
# Movement Functions
# -----------------------------------------------------------------------------


def open_tweezers(nc: NanoControl):
    nc.drive_tweezers(True)
    time.sleep(TWEEZER_MOVE_TIME_SEC)
    nc.stop()


def close_tweezers(nc: NanoControl):
    nc.drive_tweezers(False)
    time.sleep(TWEEZER_MOVE_TIME_SEC)
    nc.stop()


def level_robot(nc: NanoControl, vis: Vision):
    """
    TODO
    """

    CLOSE_THRESHOLD_PX = TOL_PX * 2  # Define "close enough" to switch modes

    while True:
        # Get current robot radius and target radius
        current_radius = get_current_robot_radius_px(vis)
        workspace_arcs = vis.detect_workspace_arcs()
        target_radius = workspace_arcs["robot_arc"]["radius"]

        # Calculate the error
        radius_error = abs(current_radius - target_radius)

        print(
            f"Current radius: {current_radius:.1f}px, Target: {target_radius:.1f}px, Error: {radius_error:.1f}px"
        )

        # Check if we've reached the target
        if radius_error <= TOL_PX:
            nc.stop()  # Make sure we stop
            print("Robot leveled successfully!")
            break

        # Determine direction
        if current_radius < target_radius:
            direction_text = "Moving robot down (away from center)"
            reverse = True
        else:
            direction_text = "Moving robot up (toward center)"
            reverse = False

        # Choose movement strategy based on distance to target
        if radius_error > CLOSE_THRESHOLD_PX:
            # Far from target - move continuously
            print(f"{direction_text} - CONTINUOUS MODE")
            nc.drive_elbow_joint(reverse=reverse)
            time.sleep(0.01)  # Short check interval while moving
            # Don't stop here - keep moving until next iteration
        else:
            # Close to target - use precise stop-and-go movements
            print(f"{direction_text} - PRECISION MODE")
            nc.drive_elbow_joint(reverse=reverse)
            time.sleep(0.1)  # Short movement duration for precision
            nc.stop()
            time.sleep(0.1)  # Brief pause between movements


def touch_disk_from_level(nc: NanoControl):
    nc.drive_elbow_joint(True)
    time.sleep(TIME_TO_DISK_FROM_LEVEL_SEC)
    nc.stop()


def move_to_rock_mapped_theta(nc: NanoControl, vis: Vision):
    # get the angle of the rock wrt to its frame of ref.
    # map that to the corresponding robot angle
    level_robot(nc, vis)
    # move base until that angle is satisfied. need to move both ways


def pickup_rock(nc: NanoControl, vis: Vision):
    move_to_rock_mapped_theta(nc, vis)
    open_tweezers(nc)
    touch_disk_from_level(nc)


def deposit_rock(nc: NanoControl, vis: Vision):
    level_robot(nc, vis)

    # Get current robot angle
    current_angle = get_current_robot_theta_deg(vis)

    # Rotate base joint counter clockwise until reaching threshold
    while current_angle < DROP_OFF_ANGLE_THRESHOLD_DEG:
        current_angle = get_current_robot_theta_deg(vis)
        print(f"current angle: {current_angle}")
        nc.drive_base_joint()  # Counter clockwise

    # Continue moving in same direction for some extra time
    nc.drive_base_joint()
    time.sleep(DROP_OFF_EXTRA_TIME_SEC)
    nc.stop()

    open_tweezers(nc)
    close_tweezers(nc)


def rotate_rock_to_pickup_pos(nc: NanoControl, tt: Turntable, vis: Vision):
    while True:
        # Get current rock position and calculate rotation needed
        rock_position = get_first_rock_coords(vis)

        # Check if rock was found
        if rock_position[0] is None or rock_position[1] is None:
            print("No rock found to rotate to")
            break

        disk_center = get_disk_centre(vis)
        arc_radius = get_workspace_arc_radius(vis)

        angle_to_rotate, reverse_direction = get_disk_rotation_angle(
            rock_position, disk_center, arc_radius
        )

        # Check if we're close enough to target
        if angle_to_rotate <= ROCK_POS_TOL_ANGLE:
            break

        # Rotate toward target
        tt.rotate(angle_to_rotate, reverse=reverse_direction)


def main():
    with (
        NanoControl("COM19") as nc,
        Turntable("COM17") as tt,
        Vision(
            enable_visualization=True,
            frame_scale_factor=1,
            calibration_debug=False,
        ) as vis,
    ):
        set_robot_speed(nc)
        while is_rock_available(vis):
            rotate_rock_to_pickup_pos(nc, tt, vis)
            pickup_rock(nc, vis)
            deposit_rock(nc, vis)


if __name__ == "__main__":
    main()


# while True:
#     level_robot(nc, vis)
#     time.sleep(3)
# time.sleep(1)
# touch_disk_from_level(nc)
# time.sleep(1)
# level_robot(nc, vis)
# time.sleep(1)

# while True:
#     time.sleep(1)
#     rotate_rock_to_pickup_pos()
#     time.sleep(2)
