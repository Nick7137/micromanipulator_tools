# THIS IS ALL TERRIBLE CODE BUT I WAS UNDER TIME PRESSURE THIS ALL NEEDS REDOING ESSENTIALLY - BUT IT WORKS
# This really should be a class
import os
import sys
import csv
import time
import math
import keyboard
from typing import Tuple
from datetime import datetime

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
TWEEZER_PARTIAL_OPEN_SEC = 5.5
TWEEZER_PARTIAL_CLOSE_SEC = 6
TWEEZER_FULL_CLOSE_SEC = 6
TWEEZER_FULL_OPEN_SEC = 6
TIME_TO_DISK_FROM_LEVEL_SEC = 3.5
DROP_OFF_ANGLE_THRESHOLD_DEG = 26
DROP_OFF_EXTRA_TIME_SEC = 3

ANGLE_OFFSET = 1

# -----------------------------------------------------------------------------
# Setup Functions
# -----------------------------------------------------------------------------


def set_robot_speed(nc: NanoControl):
    active_speed_profile = 1
    nc.set_speed_profile(active_speed_profile, SPEED_PRESETS[1])
    nc.change_speed_profile_to(active_speed_profile)


def run_angle_calibration(nc: NanoControl, vis: Vision):
    calibration_data = []

    print(
        "Move the robot to each marker and press 'Space' to capture the position."
    )

    space_pressed = False

    while True:
        # Check for space press
        if keyboard.is_pressed("space"):
            if not space_pressed:
                space_pressed = True

                # Get current centre arc angle
                workspace_arcs = vis.detect_workspace_arcs()
                centre_arc_markers = workspace_arcs["calibration_markers"]
                current_marker_index = len(calibration_data)

                if current_marker_index >= len(centre_arc_markers["angles"]):
                    print("All markers calibrated. Press 'Q' to finish.")
                    continue

                centre_angle = centre_arc_markers["angles"][
                    current_marker_index
                ]

                # Level the robot
                level_robot(nc, vis)

                # Get robot angle
                robot_angle = get_current_robot_theta_deg(vis)

                # Store calibration point
                calibration_data.append((centre_angle, robot_angle))
                print(
                    f"Calibration point captured: Centre angle {centre_angle}, Robot angle {robot_angle}"
                )

                # Touch down then raise ever so slightly so we can rotate base
                touch_disk_from_level(nc)
                nc.drive_elbow_joint()
                time.sleep(0.001)
                nc.stop()

                # Check if we've collected all calibration points
                if len(calibration_data) >= len(centre_arc_markers["angles"]):
                    print("All calibration points collected!")
                    break  # Exit the loop when all points are collected

        else:
            space_pressed = False

        if keyboard.is_pressed("q"):
            print("Calibration cancelled. Discarding data.")
            return None

        time.sleep(0.1)

    print("Calibration finished. Calibration data:")
    for centre, robot in calibration_data:
        print(f"Centre: {centre:.2f}, Robot: {robot:.2f}")

    return calibration_data


def save_calibration_data(calibration_data, filename="calibration_data.csv"):
    """Save calibration data to CSV file in the same directory as the script"""
    if calibration_data is None:
        print("No calibration data to save.")
        return

    try:
        # Get the directory where the current script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Create the full path to the calibration file
        full_path = os.path.join(script_dir, filename)

        with open(full_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(["centre_angle", "robot_angle", "timestamp"])
            # Write data
            timestamp = datetime.now().isoformat()
            for centre_angle, robot_angle in calibration_data:
                writer.writerow([centre_angle, robot_angle, timestamp])
        print(f"Calibration data saved to {full_path}")
    except Exception as e:
        print(f"Error saving calibration data: {e}")


def load_calibration_data(filename="calibration_data.csv"):
    """Load calibration data from CSV file in the same directory as the script"""
    try:
        # Get the directory where the current script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Create the full path to the calibration file
        full_path = os.path.join(script_dir, filename)

        calibration_data = []
        with open(full_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                centre_angle = float(row["centre_angle"])
                robot_angle = float(row["robot_angle"])
                calibration_data.append((centre_angle, robot_angle))
        print(
            f"Loaded {len(calibration_data)} calibration points from {full_path}"
        )
        return calibration_data
    except FileNotFoundError:
        print(f"Calibration file {filename} not found in script directory.")
        return None
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        return None


def run_angle_mapping(nc: NanoControl, vis: Vision):
    pass


def run_calibration(nc: NanoControl, vis: Vision, force_recalibrate=False):
    """
    Run calibration procedure.

    Args:
        nc: NanoControl instance
        vis: Vision instance
        force_recalibrate: If True, skip loading existing data and run new calibration
    """
    if not force_recalibrate:
        # Try to load existing calibration data
        existing_data = load_calibration_data()

        if existing_data:
            print(
                f"Found existing calibration with {len(existing_data)} points."
            )
            print("Using existing calibration data.")
            return existing_data
        else:
            print("No existing calibration data found.")
    else:
        print("Force recalibration requested.")

    # Run new calibration
    print("Starting new calibration...")
    data = run_angle_calibration(nc, vis)

    # Save the new calibration data
    if data is not None:
        save_calibration_data(data)
        print("Calibration completed and saved!")
    else:
        print("Calibration was cancelled.")

    return data


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


def get_rock_angle_relative_to_centre_arc_deg(
    vis: Vision, rock_index: int = 0
):
    """
    Get the angle of a rock relative to the centre arc center in degrees.

    Args:
        vis: Vision instance
        rock_index: Index of the rock (0 = first/closest rock)

    Returns:
        float: Angle in degrees, or None if no rock found
    """
    # Get rocks data
    rocks_data = vis.detect_rocks()
    if not rocks_data or rock_index >= len(rocks_data):
        return None

    # Get the specific rock centroid
    rock_pixel_pos = rocks_data[rock_index]["pixel_centroid"]

    # Get workspace arcs to find centre arc center
    workspace_arcs = vis.detect_workspace_arcs()
    centre_arc_center = workspace_arcs["end_effector_arc"]["center"]

    # Calculate displacement from centre arc center
    dx = centre_arc_center[0] - rock_pixel_pos[0]
    dy = rock_pixel_pos[1] - centre_arc_center[1]

    # Calculate angle (0° pointing up, positive clockwise) - same as robot angle calculation
    angle_rad = math.atan2(dx, -dy)  # Note: -dy to make 0° point up
    angle_deg = math.degrees(angle_rad)

    print(f"rock angle: {angle_deg}")

    return angle_deg


# -----------------------------------------------------------------------------
# Movement Functions
# -----------------------------------------------------------------------------


def open_tweezers(nc: NanoControl, duration_sec: float):
    nc.drive_tweezers(True)
    time.sleep(duration_sec)
    nc.stop()


def close_tweezers(nc: NanoControl, duration_sec: float):
    nc.drive_tweezers(False)
    time.sleep(duration_sec)
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

        # Check if we've reached the target
        if radius_error <= TOL_PX:
            nc.stop()
            break

        # Determine direction
        if current_radius < target_radius:
            reverse = True
        else:
            reverse = False

        # Choose movement strategy based on distance to target
        if radius_error > CLOSE_THRESHOLD_PX:
            # Far from target - move continuously
            nc.drive_elbow_joint(reverse=reverse)
            time.sleep(0.01)  # Short check interval while moving
            # Don't stop here - keep moving until next iteration
        else:
            # Close to target - use precise stop-and-go movements
            nc.drive_elbow_joint(reverse=reverse)
            time.sleep(0.1)  # Short movement duration for precision
            nc.stop()
            time.sleep(0.1)  # Brief pause between movements


def touch_disk_from_level(nc: NanoControl):
    nc.drive_elbow_joint(True)
    time.sleep(TIME_TO_DISK_FROM_LEVEL_SEC)
    nc.stop()


def move_robot_to_angle(nc: NanoControl, vis: Vision, target_angle: float):
    """
    Move robot base joint to achieve target angle.
    Similar to level_robot but controls base joint instead of elbow joint.
    """

    level_robot(nc, vis)

    CLOSE_THRESHOLD_DEG = (
        ROCK_POS_TOL_ANGLE * 2
    )  # Define "close enough" to switch modes

    while True:
        # Get current robot angle
        current_angle = get_current_robot_theta_deg(vis)

        # Calculate the error
        angle_error = target_angle - current_angle

        # Normalize angle error to [-180, 180] range
        while angle_error > 180:
            angle_error -= 360
        while angle_error < -180:
            angle_error += 360

        angle_error_abs = abs(angle_error)

        # Check if we've reached the target
        if angle_error_abs <= ROCK_POS_TOL_ANGLE:
            nc.stop()
            print("Robot positioned at target angle!")
            break

        # Determine direction
        if angle_error > 0:
            reverse = False  # Rotate clockwise (positive direction)
        else:
            reverse = True  # Rotate counter-clockwise (negative direction)

        # Choose movement strategy based on distance to target
        if angle_error_abs > CLOSE_THRESHOLD_DEG:
            # Far from target - move continuously
            nc.drive_base_joint(reverse=reverse)
            time.sleep(0.01)  # Short check interval while moving
            # Don't stop here - keep moving until next iteration
        else:
            # Close to target - use precise stop-and-go movements
            nc.drive_base_joint(reverse=reverse)
            time.sleep(0.1)  # Short movement duration for precision
            nc.stop()
            time.sleep(0.1)  # Brief pause between movements


def move_robot_to_rock(nc: NanoControl, vis: Vision):
    # Get the angle of the rock (right edge) wrt to its frame of ref.
    angle_deg = get_rock_angle_relative_to_centre_arc_deg(vis)

    # map that to the corresponding robot angle then add tolerance

    # move base until that angle is satisfied. need to move both ways
    move_robot_to_angle(nc, vis, angle_deg - ANGLE_OFFSET)


def grab_rock(nc: NanoControl, vis: Vision):
    # Undo the motion of the depositing
    nc.drive_base_joint(reverse=True)
    time.sleep(DROP_OFF_EXTRA_TIME_SEC + 3)
    nc.stop()

    move_robot_to_rock(nc, vis)
    open_tweezers(nc, TWEEZER_PARTIAL_OPEN_SEC)
    touch_disk_from_level(nc)
    # nc.drive_elbow_joint(reverse=True)
    # time.sleep(0.4)
    # nc.stop()
    close_tweezers(nc, TWEEZER_FULL_CLOSE_SEC)


def move_offscreen(nc: NanoControl, vis: Vision):
    level_robot(nc, vis)

    # Get current robot angle
    current_angle = get_current_robot_theta_deg(vis)

    # Rotate base joint counter clockwise until reaching threshold
    while current_angle < DROP_OFF_ANGLE_THRESHOLD_DEG:
        current_angle = get_current_robot_theta_deg(vis)
        nc.drive_base_joint()  # Counter clockwise

    # Continue moving in same direction for some extra time
    nc.drive_base_joint()
    time.sleep(DROP_OFF_EXTRA_TIME_SEC)
    nc.stop()


def deposit_rock(nc: NanoControl, vis: Vision):
    move_offscreen(nc, vis)
    open_tweezers(nc, TWEEZER_PARTIAL_OPEN_SEC)
    close_tweezers(nc, TWEEZER_FULL_CLOSE_SEC)


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

        time.sleep(1)

        # Essentially move the robot offscreen in a controlled manner.
        move_offscreen(nc, vis)

        while is_rock_available(vis):
            rotate_rock_to_pickup_pos(nc, tt, vis)
            grab_rock(nc, vis)
            deposit_rock(nc, vis)

        # TODO make robot keep running maybe a pause function

        # TODO the calibration data should be saved and only overwritten when a calibration procedure is run.


if __name__ == "__main__":
    main()
