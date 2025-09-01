# TODO turn into class later.

# class MicromanipulatorVision:
#     def __init__():
#         ...
#         # some kind of calibration - image distortion

#     def detect_rocks(self):
#         ...
#         # get coords of all the rocks and the rectangle outline shape? vector?

#     def track_robot(self):
#         ...
#         # get coords and vector of rocks


# aruco markers
# https://www.youtube.com/watch?v=bS00Vs09Upw&t=325s


# only want to rotate one direction to stop backlash?
# depthanything v2


# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# implement big brain algorithm that decides which rock to go for.


# ==============================================================================


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
