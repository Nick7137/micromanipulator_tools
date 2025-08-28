import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micromanipulator_tools import (
    NanoControl,
    NanoControlError,
    NanoControlCommandError,
    NanoControlConnectionError,
    Turntable,
    TurntableError,
    TurntableConnectionError,
    TurntableCommandError,
    TurntableTimeoutError,
)

__all__ = [
    "time",
    "sys",
    "os",
    "NanoControl",
    "NanoControlError",
    "NanoControlCommandError",
    "NanoControlConnectionError",
    "Turntable",
    "TurntableError",
    "TurntableConnectionError",
    "TurntableCommandError",
    "TurntableTimeoutError",
]


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
    try:
        with NanoControl("COM19") as nc:
            with Turntable("COM17") as tt:
                active_speed_profile = 1

                nc.set_speed_profile(active_speed_profile, SPEED_PRESETS[1])
                nc.change_speed_profile_to(active_speed_profile)

                # nc.drive_base_joint()
                # nc.drive_elbow_joint()
                # nc.drive_prismatic_joint()
                # nc.drive_tweezer_joint()

                nc.drive_base_joint(reverse=True)
                # nc.drive_elbow_joint(reverse=True)
                # nc.drive_prismatic_joint(reverse=True)
                # nc.drive_tweezer_joint(reverse=True)

                tt.rotate(300, reverse=False)
                time.sleep(5)
                tt.rotate(300, reverse=False)
                time.sleep(5)

                nc.stop()

    except Exception as e:
        print(f"{e}")


if __name__ == "__main__":
    main()


# aruco markers
# https://www.youtube.com/watch?v=bS00Vs09Upw&t=325s


# only want to rotate one direction to stop backlash?

# depthanything v2


# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
