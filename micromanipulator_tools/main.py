# TODO only want to rotate one direction to stop backlash?
# TODO implement big brain algorithm that decides which rock to go for.

# TODO algorithm for homing the device.
# TODO algorithm, pick rock that is closest to the purple arc, if far put it on purple line, otherwise go grab it near enough go grab it

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

TOL_PX = 0


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

        def get_current_robot_radius():
            polar_coords, centroid_px, body_contour = vis.detect_robot()
            radius, angle = polar_coords
            return radius

        def level_robot():
            radius = get_current_robot_radius()

            # Get the robot arc radius from vision
            workspace_arcs = vis.detect_workspace_arcs()
            robot_arc_radius = workspace_arcs["robot_arc"]["radius"]

            if radius < (robot_arc_radius - TOL_PX):
                while radius < robot_arc_radius:
                    radius = get_current_robot_radius()
                    nc.drive_elbow_joint(reverse=True)
            elif radius > (robot_arc_radius + TOL_PX):
                while radius > robot_arc_radius:
                    radius = get_current_robot_radius()
                    nc.drive_elbow_joint(reverse=False)
            else:
                pass

            nc.stop()

        def go_down_from_level():
            nc.drive_elbow_joint(True)
            time.sleep(3.5)
            nc.stop()

        def open_tweezers():
            nc.drive_tweezers(True)
            time.sleep(18)
            nc.stop()

        def close_tweezers():
            nc.drive_tweezers(False)
            time.sleep(18)
            nc.stop()

        def move_off_disk():
            # get the desired angle for the robot to move to and then
            # move there

            pass

        ########################################################

        def get_first_rock_coords():
            # Extract first rock coordinates in pixels
            rocks_data = vis.detect_rocks()
            if rocks_data and len(rocks_data) > 0:
                first_rock = rocks_data[0]
                first_rock_centroid = first_rock["pixel_centroid"]
                first_rock_x, first_rock_y = first_rock_centroid

                print(
                    f"First rock at: ({first_rock_x}, {first_rock_y}) pixels"
                )
            else:
                print("No rocks detected")
                first_rock_x = None
                first_rock_y = None

            return first_rock_x, first_rock_y

        def get_disk_centre():
            # Extract disk center in pixels
            disk_data = vis.detect_disk()
            disk_center, disk_radius = disk_data
            disk_center_x, disk_center_y = disk_center

            return disk_center_x, disk_center_y

        def get_workspace_arc_radius():
            # Extract end effector arc radius
            workspace_arcs = vis.detect_workspace_arcs()
            end_effector_radius = workspace_arcs["end_effector_arc"]["radius"]
            return end_effector_radius

        def get_disk_rotation_angle(
            rock_position: Tuple[float, float],
            disk_center: Tuple[float, float],
            end_effector_radius: float,
        ) -> float:
            """
            TODO
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

            # Calculate both possible rotations
            arcsin_S = math.asin(S)
            rotation_1 = arcsin_S - rock_angle
            rotation_2 = math.pi - arcsin_S - rock_angle

            # Normalize to [0, 2π) and convert to degrees
            rotation_1_deg = (rotation_1 % (2 * math.pi)) * 180 / math.pi
            rotation_2_deg = (rotation_2 % (2 * math.pi)) * 180 / math.pi

            # Choose the smaller clockwise rotation
            if rotation_1_deg <= rotation_2_deg:
                return rotation_1_deg
            else:
                return rotation_2_deg

        def rotate_to_reach_rock():
            rock_position = get_first_rock_coords()
            disk_center = get_disk_centre()
            arc_radius = get_workspace_arc_radius()

            angle_to_rotate = get_disk_rotation_angle(
                rock_position, disk_center, arc_radius
            )

            print(f"angle {angle_to_rotate}")

            tt.rotate(angle_to_rotate)

        active_speed_profile = 1
        nc.set_speed_profile(active_speed_profile, SPEED_PRESETS[1])
        nc.change_speed_profile_to(active_speed_profile)

        # while True:
        #     time.sleep(1)
        #     rotate_to_reach_rock()
        #     time.sleep(2)

        # time.sleep(1)
        # level_robot()
        # time.sleep(1)
        # go_down_from_level()
        # time.sleep(1)
        # level_robot()
        # time.sleep(1)

        # open_tweezers()
        # close_tweezers()

        # while True:
        #     # Get the latest pre-computed frame.
        #     frame, results = vis.get_latest_output()

        #     if keyboard.is_pressed("e"):
        #         print("'e' key detected, exiting...\n")
        #         break

        #     # Small delay to prevent excessive CPU usage
        #     time.sleep(0.01)

        # print("Program ended.")


if __name__ == "__main__":
    main()
