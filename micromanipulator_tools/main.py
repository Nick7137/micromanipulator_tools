# TODO implement broad move
# TODO implement fine adjustments function
# TODO is robot infocus function and switch the calibration data also
# TODO only want to rotate one direction to stop backlash?
# TODO implement big brain algorithm that decides which rock to go for.

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micromanipulator_tools import NanoControl, Turntable, Vision


SPEED_PRESETS = {
    1: {
        "base_joint": "c64",
        "elbow_joint": "c64",
        "prismatic_joint": "c64",
        "tweezer_joint": "c64",
    },
    2: {
        "base_joint": "c01",
        "elbow_joint": "c01",
        "prismatic_joint": "c01",
        "tweezer_joint": "c01",
    },
}


def main():
    with (
        NanoControl("COM19") as nc,
        Turntable("COM17") as tt,
        Vision(frame_scale_factor=0.6, calibration_debug=False) as vis,
    ):
        # tt.rotate(60, reverse=False)

        active_speed_profile = 1

        nc.set_speed_profile(active_speed_profile, SPEED_PRESETS[1])
        nc.change_speed_profile_to(active_speed_profile)

        # nc.drive_base_joint()
        nc.drive_elbow_joint()
        # nc.drive_tweezers()

        # nc.drive_base_joint(reverse=True)
        # nc.drive_elbow_joint(reverse=True)
        # nc.drive_tweezers(reverse=True)

        # for i in range(3):
        #     nc.drive_base_joint(reverse=True)
        #     time.sleep(5)
        #     nc.drive_base_joint(reverse=False)
        #     time.sleep(5)

        nc.stop()


if __name__ == "__main__":
    main()
