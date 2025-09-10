# TODO implement broad move
# TODO implement fine adjustments function
# TODO is robot infocus function and switch the calibration data also
# TODO only want to rotate one direction to stop backlash?
# TODO implement big brain algorithm that decides which rock to go for.

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micromanipulator_tools import NanoControl, Turntable


SPEED_PRESETS = {
    1: {
        "base_joint": "c32",
        "elbow_joint": "c64",
        "prismatic_joint": "c64",
        "tweezer_joint": "c64",
    },
}

# Note: life-hack type `mode` in powershell to get the COM ports


def precise_wait_ms(duration: float):
    """
    Very precise wait using busy waiting.
    Duration in milliseconds (can be fractional, e.g., 1.5 for 1.5ms)
    """
    start = time.perf_counter()
    while (time.perf_counter() - start) < duration / 1000:
        pass  # Busy wait - uses 100% CPU but very precise


def main():
    with (
        NanoControl("COM19") as nc,
        Turntable("COM17") as tt,
        # Vision(frame_scale_factor=0.6, calibration_debug=False) as vis,
    ):
        active_speed_profile = 1
        nc.set_speed_profile(active_speed_profile, SPEED_PRESETS[1])
        nc.change_speed_profile_to(active_speed_profile)

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

        nc.stop()


if __name__ == "__main__":
    main()
