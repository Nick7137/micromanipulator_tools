import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micromanipulator_tools import NanoControl


def emergency_stop():
    """Standalone emergency stop script"""

    try:
        with NanoControl("COM19") as nc:
            nc.stop()
            print("✓ NanoControl stopped")
    except Exception as e:
        print(f"✗ Error stopping NanoControl: {e}")


if __name__ == "__main__":
    emergency_stop()
