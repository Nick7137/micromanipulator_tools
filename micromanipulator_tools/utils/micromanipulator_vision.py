# TODO turn into class later.
# TODO IMPORTANT must set the camera 90 mm above the disk
# only want to rotate one direction to stop backlash?
# depthanything v2

# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# implement big brain algorithm that decides which rock to go for.

# ==============================================================================

import cv2 as cv
import numpy as np
from typing import Optional, Tuple, List


def tested(func):
    """
    Decorator that marks a function or method as tested by adding a
    'tested' attribute set to True.

    This can be used to programmatically check whether a function has
    been confirmed to work as expected.

    Args:
        func (callable): The function or method to mark as tested.

    Returns:
        callable: The original function with a 'tested' attribute.

    Usage:
        @tested
        def my_function():
            pass

        # Later, check if the method is tested:
        if hasattr(obj.my_function, 'tested'):
            print("Method is tested!")
    """

    func.tested = True
    return func


class MicromanipulatorVisionError(Exception):
    """
    Base exception for all MicromanipulatorVision-related errors.

    This is the parent class for all MicromanipulatorVision-specific exceptions.
    Catch this to handle any MicromanipulatorVision error generically.

    Example:
        try:
            with MicromanipulatorVision(0) as camera:
                camera.calibrate_camera()
        except MicromanipulatorVisionError as e:
            print(f"Micromanipulator vision error: {e}")
    """

    @tested
    def __init__(self, message: str) -> None:
        """
        Initialize the MicromanipulatorVisionError with a descriptive message.

        Args:
            message (str): Error description
        """
        super().__init__(message)


class MicromanipulatorVisionConnectionError(MicromanipulatorVisionError):
    """
    Exception for camera connection and hardware issues. Derived from
    MicromanipulatorVisionError class.
    """

    @tested
    def __init__(self, message: str) -> None:
        super().__init__(message)


class MicromanipulatorVisionCalibrationError(MicromanipulatorVisionError):
    """
    Exception for camera calibration processing issues. Derived from
    MicromanipulatorVisionError class.
    """

    @tested
    def __init__(self, message: str) -> None:
        super().__init__(message)


class MicromanipulatorVision:
    """
    TODO
    """

    # -------------------------------------------------------------------------
    # Class constants----------------------------------------------------------
    # -------------------------------------------------------------------------

    # Camera resolution constants
    DEFAULT_CAMERA_WIDTH = 1920
    DEFAULT_CAMERA_HEIGHT = 1080
    DEFAULT_CHECKERBOARD_SIZE = (9, 6)  # (width, height) in inner corners

    def __init__(
        self,
        camera_index: int = 0,
        checkerboard_size: Tuple[int, int] = DEFAULT_CHECKERBOARD_SIZE,
        target_width: int = DEFAULT_CAMERA_WIDTH,
        target_height: int = DEFAULT_CAMERA_HEIGHT,
    ) -> None:
        """
        Initialize MicromanipulatorVision interface.

        Args:
            camera_index (int): Camera device index (0 for default camera).
            checkerboard_size (Tuple[int, int]): Size of the chessboard pattern
                as (width, height) in inner corners.
            target_width (int): Desired camera resolution width in pixels.
            target_height (int): Desired camera resolution height in pixels.
        """

        # Store configuration parameters
        self.camera_index = camera_index
        self.checkerboard = checkerboard_size
        self.target_width = target_width
        self.target_height = target_height

        # Camera state
        self.capture: Optional[cv.VideoCapture] = None
        self.is_camera_initialized = False

        # Calibration state
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.is_calibrated = False

    def capture_calibration_images(
        self, num_images: int = 20
    ) -> List[np.ndarray]:
        """Capture a series of calibration images."""
        # TODO: Implement this method
        pass

    def find_chessboard_corners(
        self, images: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Find chessboard corners in the given images."""
        # TODO: Implement this method
        pass

    def calibrate_camera(
        self, object_points: List[np.ndarray], image_points: List[np.ndarray]
    ) -> None:
        """Perform camera calibration."""
        # TODO: Implement this method
        pass

    def save_calibration(self) -> None:
        """Save the calibration results."""
        # TODO: Implement this method
        pass

    def load_calibration(self) -> bool:
        """Load existing calibration data."""
        # TODO: Implement this method
        pass

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Apply undistortion to an image."""
        # TODO: Implement this method
        pass

    def __str__(self) -> str:
        calibration_status = (
            "calibrated" if self.is_calibrated else "not calibrated"
        )
        camera_status = (
            "connected" if self.is_camera_initialized else "not connected"
        )

        return (
            f"MicromanipulatorVision(camera_index={self.camera_index}, "
            f"checkerboard={self.checkerboard}, "
            f"resolution={self.target_width}x{self.target_height}, "
            f"camera={camera_status}, "
            f"calibration={calibration_status})"
        )
