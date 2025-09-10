# TODO implement broad move
# TODO implement fine adjustments function
# TODO is robot infocus function and switch the calibration data also
# TODO only want to rotate one direction to stop backlash?
# TODO implement big brain algorithm that decides which rock to go for.

import os
import sys
import time
import keyboard

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micromanipulator_tools import NanoControl, Turntable, Vision


SPEED_PRESETS = {
    1: {
        "base_joint": "c32",
        "elbow_joint": "c64",
        "prismatic_joint": "c64",
        "tweezer_joint": "c64",
    },
}

# Note: life-hack type `mode` in powershell to get the COM ports

# This is the time drive elbow executes to get to the default height
# without holding a rock
RISE_TIME_NO_ROCK_SEC = 3


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
        active_speed_profile = 1
        nc.set_speed_profile(active_speed_profile, SPEED_PRESETS[1])
        nc.change_speed_profile_to(active_speed_profile)

        nc.drive_elbow_joint()
        time.sleep(RISE_TIME_NO_ROCK_SEC)
        nc.stop()

        # nc.stopnack()

        # tt.rotate(60, reverse=False)

        # nc.drive_base_joint()
        # nc.drive_elbow_joint()
        # nc.drive_prismatic_joint(reverse=True)
        # nc.drive_tweezers()

        # nc.drive_base_joint(reverse=True)
        # nc.drive_elbow_joint(reverse=True)
        # nc.drive_tweezers(reverse=True)

        # for i in range(3):
        #     nc.drive_base_joint(reverse=True)
        #     time.sleep(5)
        #     nc.drive_base_joint(reverse=False)
        #     time.sleep(5)

        # nc.stop()

        # vis.detect_disk(True)
        # vis.detect_robot(True)
        # vis.detect_rocks(True)
        # vis.detect_workspace(True)
        # vis.set_camera_settings()
        # vis.dump_calibration_data()
        # print(vis)

        # print("\nStarting main program. Press 'e' to quit.")

        # TODO homing function moves the robot to the middle (theta is 0) and then adjusts the elbow until the r is the same.
        # if robot not detected, try moving left for 10 sec if not detected move right for 10 seconds if not detected crash.

        while True:
            # Get the latest pre-computed frame.
            frame, results = vis.get_latest_output()

            if keyboard.is_pressed("e"):
                print("'e' key detected, exiting...\n")
                break

            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)

        print("Program ended.")


if __name__ == "__main__":
    main()
