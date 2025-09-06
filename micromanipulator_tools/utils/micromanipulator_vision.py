# TODO get rid of all TODO's
# TODO make file header comment
# TODO make the docstrings good.
# TODO inline comments
# TODO turn into class later.
# TODO create a visualise everything function and add text to it.
# TODO make all spelling american, center, visualize etc.
# TODO go through all the code and check if any of it is unused or completely irrelevent
# TODO break class up into smaller classes if possible

# =============================================================================

import os
import glob
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

    This is the parent class for all MicromanipulatorVision-specific
    exceptions. Catch this to handle any MicromanipulatorVision error
    generically.

    Example:
        try:
            with MicromanipulatorVision(0) as camera:
                camera._calibration_solve_constants()
        except MicromanipulatorVisionError as e:
            print(f"Micromanipulator vision error: {e}")
    """

    @tested
    def __init__(self, message: str) -> None:
        """
        Initialise the MicromanipulatorVisionError with a descriptive
        message.

        Args:
            message (str): Error description
        """
        super().__init__(message)


class MicromanipulatorVisionConnectionError(MicromanipulatorVisionError):
    """
    Exception for camera connection and hardware issues. Derived from
    the MicromanipulatorVisionError class.
    """

    @tested
    def __init__(self, message: str) -> None:
        super().__init__(message)


class MicromanipulatorVisionCalibrationError(MicromanipulatorVisionError):
    """
    Exception for camera calibration processing issues. Derived from
    the MicromanipulatorVisionError class.
    """

    @tested
    def __init__(self, message: str) -> None:
        super().__init__(message)


class MicromanipulatorVision:
    """
    TODO talk about calibration height with tool and the type of camera logi c930e- specs and setting focus.
    """

    # -------------------------------------------------------------------------
    # Class constants----------------------------------------------------------
    # -------------------------------------------------------------------------

    DEFAULT_FRAME_SCALE_FACTOR = 1

    # Camera constants
    DEFAULT_CAMERA_CHANNEL = 0
    DEFAULT_CAMERA_WIDTH = 1920
    DEFAULT_CAMERA_HEIGHT = 1080
    DEFAULT_FOCUS_LEVEL = 45

    # Calibration specific settings
    CHESSBOARD_SIZE = (9, 6)  # (num_rows, num_cols) num inner corners
    CALIBRATION_IMAGES_FOLDER = "calibration_images"
    CALIBRATION_FILE = "calibration_data.npz"
    CALIBRATION_DISPLAY_TIME_MS = 100
    CALIBRATION_WINDOW_NAME = "Chessboard"

    # Corner refinement parameters
    CORNER_REFINEMENT_MAX_ITERATIONS = 30
    CORNER_REFINEMENT_EPSILON = 0.001
    CORNER_REFINEMENT_WINDOW_SIZE = (11, 11)
    CORNER_REFINEMENT_ZERO_ZONE = (-1, -1)

    # Calibration quality thresholds
    EXCELLENT_CALIBRATION_THRESHOLD = 0.5
    GOOD_CALIBRATION_THRESHOLD = 1.0
    ACCEPTABLE_CALIBRATION_THRESHOLD = 2.0

    # Calibration constant keys
    NPZ_KEY_CAMERA_MATRIX = "camera_matrix"
    NPZ_KEY_DIST_COEFFS = "dist_coeffs"
    NPZ_KEY_ROTATION_VECS = "rotation_vecs"
    NPZ_KEY_TRANSLATION_VECS = "translation_vecs"
    NPZ_KEY_REPROJECTION_ERROR = "reprojection_error"

    # Configure the free scaling parameter that controls how much of the
    # original frame is kept after undistortion.
    DEFAULT_ALPHA = 0
    ALPHA_MIN_VALUE = 0.0
    ALPHA_MAX_VALUE = 1.0

    # Robot head detection constants
    ROBOT_HEAD_HSV_LOWER_THRESHOLD = (40, 50, 50)
    ROBOT_HEAD_HSV_UPPER_THRESHOLD = (80, 255, 255)
    ROBOT_HEAD_MIN_AREA = 500
    ROBOT_HEAD_APPROX_EPSILON = 0.02

    # Rock detection constants
    ROCK_GRAY_UPPER_THRESHOLD = 60
    ROCK_MIN_AREA_PIXELS = 50
    ROCK_MAX_AREA_PIXELS = 5000
    ROCK_BLUR_KERNEL = (3, 3)
    ROCK_MORPH_KERNEL_SIZE = 3

    # Useful color codes
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    CYAN = (255, 255, 0)

    # -------------------------------------------------------------------------
    # Initialisation methods---------------------------------------------------
    # -------------------------------------------------------------------------

    @tested
    def __init__(
        self,
        camera_index: int = DEFAULT_CAMERA_CHANNEL,
        frame_scale_factor: float = DEFAULT_FRAME_SCALE_FACTOR,
        calibration_debug: bool = False,
    ) -> None:
        """
        Initialize MicromanipulatorVision with camera and calibration
        setup.

        Creates a new vision system instance with the specified camera
        and display settings. Automatically loads existing calibration
        data or performs new calibration if none exists.

        Args:
            camera_index (int, optional): Camera device index. Defaults
                to DEFAULT_CAMERA_CHANNEL.
            frame_scale_factor (float, optional): Display scaling factor
                for frames. Defaults to DEFAULT_FRAME_SCALE_FACTOR.
            calibration_debug (bool, optional): Enable calibration debug
                output and visualization. Defaults to False.

        Raises:
            MicromanipulatorVisionCalibrationError: If calibration
                creation or loading fails.

        Note:
            Camera connection is deferred until first use. Calibration
            data is automatically created if missing using default
            settings.
        """

        # Get important file paths.
        self._current_dir = os.path.dirname(__file__)
        self._root_dir = os.path.dirname(os.path.dirname(self._current_dir))

        # Store configuration parameters
        self._camera_index = camera_index
        self._frame_scale_factor = frame_scale_factor
        self._calibration_debug = calibration_debug

        # Camera state
        self._camera: Optional[cv.VideoCapture] = None

        # Calibration state
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None
        self._rotation_vecs: Optional[List[np.ndarray]] = None
        self._translation_vecs: Optional[List[np.ndarray]] = None
        self._reprojection_error: Optional[float] = None

        # Cached detection results
        self._orientation_result = None
        self._disk_result = None
        self._robot_head_result = None
        self._rocks_result = None

        # Check and create default calibration if missing, then load
        default_path = os.path.join(self._current_dir, self.CALIBRATION_FILE)
        if not os.path.exists(default_path):
            self._calibration_run()
        self._load_calibration_data()

    @tested
    def __enter__(self) -> "MicromanipulatorVision":
        """
        Enter the runtime context for camera resource management.

        Returns:
            self: The MicromanipulatorVision instance
        """

        return self

    @tested
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the runtime context and cleanup camera resources.

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)
        """

        if self._camera is not None:
            self._camera.release()
            self._camera = None

    @tested
    def __str__(self) -> str:
        """
        Return a string representation of the MicromanipulatorVision
        object.

        Returns:
            str: Human-readable description of the object state
        """

        return f"MicromanipulatorVision(camera_index={self._camera_index})"

    # -------------------------------------------------------------------------
    # Private camera calibration methods---------------------------------------
    # -------------------------------------------------------------------------

    @tested
    def _calibration_load_frames(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load calibration frames from the designated calibration folder.

        Searches for .jpg frames in the calibration directory and loads
        them into memory for chessboard corner detection.

        Returns:
            Tuple[List[np.ndarray], List[str]]: Loaded frames and their
                corresponding filenames.

        Raises:
            MicromanipulatorVisionCalibrationError: If no frames found
                or if no frames could be loaded.
        """

        # Find calibration frames
        calibration_dir = os.path.join(
            self._root_dir,
            "resources",
            "camera_calibration",
            self.CALIBRATION_IMAGES_FOLDER,
        )

        frame_path_list = glob.glob(os.path.join(calibration_dir, "*.jpg"))

        if len(frame_path_list) == 0:
            raise MicromanipulatorVisionCalibrationError(
                f"MicromanipulatorVision._calibration_load_frames: "
                f"No calibration frames found in {calibration_dir}. "
                f"Please add .jpg calibration frames to this directory."
            )

        if self._calibration_debug:
            print(f"Looking for calibration frames in: {calibration_dir}")
            print(f"Found {len(frame_path_list)} frames.")

        # Load all frames
        frames = []
        filenames = []

        for frame_path in frame_path_list:
            frame = cv.imread(frame_path)
            if frame is not None:
                frames.append(frame)
                filenames.append(os.path.basename(frame_path))
            else:
                print(f"WARNING: Could not load {frame_path}")

        if len(frames) == 0:
            raise MicromanipulatorVisionCalibrationError(
                "MicromanipulatorVision._calibration_load_frames: "
                "No valid calibration frames could be loaded."
            )

        return frames, filenames

    @tested
    def _calibration_find_chessboard_corners(
        self, frames: List[np.ndarray], filenames: List[str]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
        """
        Detect and refine chessboard corners in calibration frames.

        Processes each frame to find chessboard patterns, refines corner
        positions to subpixel accuracy, and optionally displays visual
        feedback during debugging.

        Args:
            frames (List[np.ndarray]): List of calibration frames.
            filenames (List[str]): Corresponding frame filenames for
                logging.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
                Object points, frame points, and actual frame size.

        Raises:
            MicromanipulatorVisionCalibrationError: If no chessboard
                patterns detected in any frames.
        """

        # Termination criteria for corner refinement
        term_criteria = (
            cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
            self.CORNER_REFINEMENT_MAX_ITERATIONS,
            self.CORNER_REFINEMENT_EPSILON,
        )

        # Create object points (3D coordinates of chessboard corners)
        num_rows, num_cols = self.CHESSBOARD_SIZE
        world_points_template = np.zeros((num_rows * num_cols, 3), np.float32)
        world_points_template[:, :2] = np.mgrid[
            0:num_rows, 0:num_cols
        ].T.reshape(-1, 2)

        object_points_list = []
        frame_points_list = []
        actual_frame_size = None

        for frame_bgr, filename in zip(frames, filenames):
            if self._calibration_debug:
                print(f"Processing: {filename}")

            frame_gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)

            # Set frame size from first successful frame
            if actual_frame_size is None:
                actual_frame_size = frame_gray.shape[::-1]

            # Find chessboard corners
            corners_found, corners_original = cv.findChessboardCorners(
                frame_gray, self.CHESSBOARD_SIZE, None
            )
            if corners_found:
                if self._calibration_debug:
                    print("✓ Found chessboard corners")

                object_points_list.append(world_points_template.copy())

                # Refine corner positions to subpixel accuracy
                corners_refined = cv.cornerSubPix(
                    frame_gray,
                    corners_original,
                    self.CORNER_REFINEMENT_WINDOW_SIZE,
                    self.CORNER_REFINEMENT_ZERO_ZONE,
                    term_criteria,
                )
                frame_points_list.append(corners_refined)

                # Show visual feedback if debug mode is enabled
                if self._calibration_debug:
                    cv.drawChessboardCorners(
                        frame_bgr,
                        self.CHESSBOARD_SIZE,
                        corners_refined,
                        corners_found,
                    )

                    resized_frame_bgr = self.scale_frame(
                        frame_bgr, self._frame_scale_factor
                    )
                    cv.imshow(self.CALIBRATION_WINDOW_NAME, resized_frame_bgr)
                    cv.waitKey(self.CALIBRATION_DISPLAY_TIME_MS)

            else:
                print(f"✗ No chessboard corners found in frame {filename}")

        cv.destroyAllWindows()

        if len(object_points_list) == 0:
            raise MicromanipulatorVisionCalibrationError(
                "MicromanipulatorVision._calibration_find_chessboard_corners: "
                "No chessboard patterns were detected in any frames."
            )

        print(
            f"Successfully processed {len(object_points_list)} out of "
            f"{len(frames)} frames for calibration. Now doing very "
            "clever maths......."
        )
        return object_points_list, frame_points_list, actual_frame_size

    @tested
    def _calibration_solve_constants(
        self,
        object_points: List[np.ndarray],
        frame_points: List[np.ndarray],
        frame_size: Tuple[int, int],
    ) -> None:
        """
        Calculate camera calibration parameters using OpenCV
        calibration.

        Performs camera calibration to determine camera matrix,
        distortion coefficients, and reprojection error from detected
        corner points.

        Args:
            object_points (List[np.ndarray]): 3D world coordinates of
                chessboard corners.
            frame_points (List[np.ndarray]): 2D frame coordinates of
                detected corners.
            frame_size (Tuple[int, int]): frame dimensions
                (width, height).

        Raises:
            MicromanipulatorVisionCalibrationError: If calibration data
                is missing or mismatched between object and frame
                points.
        """

        if len(object_points) == 0 or len(frame_points) == 0:
            raise MicromanipulatorVisionCalibrationError(
                "MicromanipulatorVision._calibration_solve_constants: Cannot "
                "calibrate: no object points or frame points provided."
            )

        if len(object_points) != len(frame_points):
            raise MicromanipulatorVisionCalibrationError(
                "MicromanipulatorVision._calibration_solve_constants: "
                f"Mismatch: {len(object_points)} object point sets vs "
                f"{len(frame_points)} frame point sets."
            )

        # Perform OpenCV camera calibration
        (
            reprojection_error,
            camera_matrix,
            dist_coeffs,
            rotation_vecs,
            translation_vecs,
        ) = cv.calibrateCamera(
            object_points,
            frame_points,
            frame_size,
            None,  # Initial camera matrix
            None,  # Initial distortion coefficients
        )

        # Store calibration results
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs
        self._rotation_vecs = rotation_vecs
        self._translation_vecs = translation_vecs
        self._reprojection_error = reprojection_error

    @tested
    def _calibration_save_constants(self) -> None:
        """
        Save computed calibration parameters to NPZ file.

        Stores camera matrix, distortion coefficients, rotation vectors,
        translation vectors, and reprojection error to disk for future
        use.

        Raises:
            MicromanipulatorVisionCalibrationError: If calibration data
                is incomplete (missing camera matrix or distortion
                coefficients).
        """

        if self._camera_matrix is None or self._dist_coeffs is None:
            raise MicromanipulatorVisionCalibrationError(
                "MicromanipulatorVision._calibration_save_constants: "
                "Calibration data is incomplete. Camera matrix or distortion "
                "coefficients missing."
            )

        # Create save path (same directory as the script) and save data.
        save_path = os.path.join(self._current_dir, self.CALIBRATION_FILE)
        np.savez(
            save_path,
            **{
                self.NPZ_KEY_CAMERA_MATRIX: self._camera_matrix,
                self.NPZ_KEY_DIST_COEFFS: self._dist_coeffs,
                self.NPZ_KEY_ROTATION_VECS: self._rotation_vecs,
                self.NPZ_KEY_TRANSLATION_VECS: self._translation_vecs,
                self.NPZ_KEY_REPROJECTION_ERROR: self._reprojection_error,
            },
        )

    @tested
    def _calibration_run(self) -> None:
        """
        Execute complete camera calibration workflow.

        Orchestrates the full calibration process: loads frames, finds
        chessboard corners, solves calibration parameters, and saves
        results to file.

        Raises:
            MicromanipulatorVisionCalibrationError: If any step in the
                calibration process fails.
        """

        try:
            frames, filenames = self._calibration_load_frames()
            object_points, frame_points, frame_size = (
                self._calibration_find_chessboard_corners(frames, filenames)
            )
            self._calibration_solve_constants(
                object_points, frame_points, frame_size
            )
            self._calibration_save_constants()

        except Exception as e:
            raise MicromanipulatorVisionCalibrationError(
                f"MicromanipulatorVision._calibration_run: "
                f"Failed to create calibration: {str(e)}"
            )

    @tested
    def _load_calibration_data(self) -> None:
        """
        Load previously saved calibration data from NPZ file.

        Reads camera matrix, distortion coefficients, and other
        calibration parameters from the calibration file and stores them
        in instance variables.

        Raises:
            MicromanipulatorVisionCalibrationError: If calibration file
                not found or data cannot be loaded/parsed.
        """

        # Build file path
        load_path = os.path.join(self._current_dir, self.CALIBRATION_FILE)

        # Check if file exists
        if not os.path.exists(load_path):
            raise MicromanipulatorVisionCalibrationError(
                f"MicromanipulatorVision._load_calibration_data: "
                f"Calibration file not found: {load_path}"
            )

        # Load calibration data from file
        try:
            data = np.load(load_path)

            # Load core calibration data
            self._camera_matrix = data[self.NPZ_KEY_CAMERA_MATRIX]
            self._dist_coeffs = data[self.NPZ_KEY_DIST_COEFFS]
            self._rotation_vecs = data[self.NPZ_KEY_ROTATION_VECS]
            self._translation_vecs = data[self.NPZ_KEY_TRANSLATION_VECS]
            self._reprojection_error = float(
                data[self.NPZ_KEY_REPROJECTION_ERROR]
            )

        except Exception as e:
            raise MicromanipulatorVisionCalibrationError(
                f"MicromanipulatorVision._load_calibration_data: "
                f"Failed to load calibration data from {load_path}: {str(e)}"
            )

    # -------------------------------------------------------------------------
    # Private vision methods---------------------------------------------------
    # -------------------------------------------------------------------------

    @tested
    def _initialise_camera(self) -> None:
        """
        Initialise camera connection and set the frame dimensions.

        Opens VideoCapture connection to the specified camera index and
        sets the camera to highest available resolution.

        Raises:
            MicromanipulatorVisionConnectionError: If camera cannot be
                opened or is not available.
        """

        # Check if already initialised.
        if self._camera is not None:
            return

        self._camera = cv.VideoCapture(self._camera_index + cv.CAP_DSHOW)

        if not self._camera.isOpened():
            self._camera = None
            raise MicromanipulatorVisionConnectionError(
                f"Failed to open camera {self._camera_index}"
            )

        # Set highest resolution
        self._camera.set(cv.CAP_PROP_FRAME_WIDTH, self.DEFAULT_CAMERA_WIDTH)
        self._camera.set(cv.CAP_PROP_FRAME_HEIGHT, self.DEFAULT_CAMERA_HEIGHT)

        # Disable autofocus and set manual focus.
        self._camera.set(cv.CAP_PROP_AUTOFOCUS, 0)
        self._camera.set(cv.CAP_PROP_FOCUS, self.DEFAULT_FOCUS_LEVEL)

        # print(f"Camera {self._camera_index} initialised")

    # TODO understand the maths here - rename vars
    @tested
    def _visualize_orientation(
        self, frame: np.ndarray, theta: float, rho: float
    ) -> np.ndarray:
        """
        Draw detected orientation line on frame for debugging.

        Args:
            frame (np.ndarray): Input frame to draw on.
            theta (float): Line angle in radians from Hough transform.
            r (float): Distance from origin to line in pixels.

        Returns:
            np.ndarray: Frame copy with red orientation line drawn.
        """

        visualization_frame = frame.copy()

        # Convert from polar to cartesian.
        x_comp = np.cos(theta)
        y_comp = np.sin(theta)

        # Get the point on the line closest to the origin
        x0 = x_comp * rho
        y0 = y_comp * rho

        # Large number to span entire window.
        span = 1000
        x1 = int(x0 + span * (-y_comp))
        y1 = int(y0 + span * (x_comp))
        x2 = int(x0 - span * (-y_comp))
        y2 = int(y0 - span * (x_comp))

        thickness = 2
        cv.line(visualization_frame, (x1, y1), (x2, y2), self.RED, thickness)

        return visualization_frame

    @tested
    def _visualize_disk(
        self, frame: np.ndarray, center: Tuple[int, int], radius: int
    ) -> np.ndarray:
        """
        Draw detected disk visualization on frame for debugging.

        Args:
            frame (np.ndarray): Input frame to draw on.
            center (Tuple[int, int]): (x, y) coordinates of disk center.
            radius (int): Radius of the disk in pixels.

        Returns:
            np.ndarray: Frame copy with cyan disk outline and red center dot.
        """

        visualization_frame = frame.copy()

        fill = -1
        outline_thickness = 2
        central_dot_radius = 4

        # Show the disk
        cv.circle(
            visualization_frame, center, radius, self.CYAN, outline_thickness
        )

        # Show the centre of the disk
        cv.circle(
            visualization_frame, center, central_dot_radius, self.RED, fill
        )

        return visualization_frame

    @tested
    def _visualize_robot_head(
        self,
        frame: np.ndarray,
        centroid: Tuple[int, int],
        largest_contour: np.ndarray,
    ) -> np.ndarray:
        """
        Draw detected robot head visualization on frame for debugging.

        Args:
            frame (np.ndarray): Input frame to draw on.
            centroid (Tuple[int, int]): (x, y) coordinates of head
                centroid.
            largest_contour (np.ndarray): Contour points of head.

        Returns:
            np.ndarray: Frame copy with green contour outline and red
                center dot.
        """

        visualization_frame = frame.copy()

        fill = -1
        contour_idx = -1
        line_thickness = 2
        central_dot_radius = 4

        cv.drawContours(
            visualization_frame,
            [largest_contour],
            contour_idx,
            self.GREEN,
            line_thickness,
        )
        cv.circle(
            visualization_frame, centroid, central_dot_radius, self.RED, fill
        )

        return visualization_frame

    @tested
    def _visualize_rocks(
        self,
        frame: np.ndarray,
        detected_rocks: List[Tuple[np.ndarray, Tuple[int, int], float]],
    ) -> np.ndarray:
        """
        Draw detected rocks visualization on frame for debugging.

        Args:
            frame (np.ndarray): Input frame to draw on.
            detected_rocks (List[Tuple]): List of (min_rect, centroid,
                area) tuples from rock detection.

        Returns:
            np.ndarray: Frame copy with cyan rock outlines and red
                center dots.
        """

        visualization_frame = frame.copy()

        fill = -1
        contour_idx = 0
        line_thickness = 2
        center_rad = 3

        for i, (min_rect, centroid, area) in enumerate(detected_rocks):
            # Get the four corner points of the rotated rectangle.
            box_points = cv.boxPoints(min_rect)
            box_points = np.int32(box_points)

            # Draw the minimum area rectangle.
            cv.drawContours(
                visualization_frame,
                [box_points],
                contour_idx,
                self.CYAN,
                line_thickness,
            )

            # Draw the centroid of the rock.
            cv.circle(
                visualization_frame, centroid, center_rad, self.RED, fill
            )

        return visualization_frame

    @tested
    def _display(self, name: str, frame: np.ndarray) -> None:
        """
        Display frame in a window and wait for user keypress. Blocks
        until user presses any key, then closes the window.

        Args:
            name (str): Window name for the display.
            frame (np.ndarray): Frame to display.

        """

        cv.imshow(name, frame)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # -------------------------------------------------------------------------
    # Public interface: calibration and settings-------------------------------
    # -------------------------------------------------------------------------

    @tested
    def dump_calibration_data(self) -> None:
        """
        Display detailed calibration information.

        Shows camera matrix, distortion coefficients, reprojection
        error, and quality assessment for the calibration.

        Raises:
            MicromanipulatorVisionCalibrationError: If calibration data
                is not available.
        """

        file_path = os.path.join(self._current_dir, self.CALIBRATION_FILE)

        if os.path.exists(file_path):
            print(f"\n{'=' * 50}")
            print(f"{' ' * 15} CALIBRATION DATA\n")

            print("Camera Matrix:")
            print(self._camera_matrix)
            print("\nDistortion Coefficients:")
            print(self._dist_coeffs)

            if self._reprojection_error is not None:
                print(
                    f"\nReprojection Error: "
                    f"{self._reprojection_error:.4f} pixels"
                )

                # Evaluate calibration quality
                if (
                    self._reprojection_error
                    < self.EXCELLENT_CALIBRATION_THRESHOLD
                ):
                    print("✓ Excellent calibration quality!")
                elif (
                    self._reprojection_error < self.GOOD_CALIBRATION_THRESHOLD
                ):
                    print("✓ Very good calibration quality")
                elif (
                    self._reprojection_error
                    < self.ACCEPTABLE_CALIBRATION_THRESHOLD
                ):
                    print("⚠ OK calibration quality (acceptable)")
                else:
                    print("⚠ Poor calibration quality - retake frames!")
            else:
                print("\nReprojection Error: Not available")

            print(f"{'=' * 50}\n")

        else:
            print(f"\n{'=' * 50}")
            print("            CALIBRATION DATA\n")
            print(f"❌ Calibration file not found: {file_path}")
            print(f"{'=' * 50}\n")

    @tested
    def set_camera_settings(self) -> None:
        """
        Open camera settings dialog and display live video feed.

        This method initialises the camera, opens the built-in camera
        settings dialog, and displays a live video feed of the
        undistorted camera output. It allows the user to adjust camera
        settings visually and see the results in real-time.

        The method runs until the user presses 'q' to quit.

        For the settings used with the logi c390e camera see
        "resources/camera_settings"

        Raises:
            MicromanipulatorVisionConnectionError: If camera
                initialisation fails or connection is lost.
            MicromanipulatorVisionCalibrationError: If undistortion
                fails due to missing calibration data.
        """

        print(
            "Entered settings configuration mode. Press 'q' to exit when done."
        )
        self._initialise_camera()
        self._camera.set(cv.CAP_PROP_SETTINGS, 1)

        while True:
            frame = self.capture_frame()
            undistorted = self.undistort_frame(frame)
            resized_frame = self.scale_frame(undistorted)

            cv.imshow("Undistorted Video Feed", resized_frame)

            # Wait 20ms for keypress; use bitwise mask (0xFF) to extract
            # only ASCII bits, exit if 'q' pressed.
            if cv.waitKey(20) & 0xFF == ord("q"):
                break

        cv.destroyAllWindows()

    # -------------------------------------------------------------------------
    # Public interface: frame manipulation-------------------------------------
    # -------------------------------------------------------------------------

    @tested
    def capture_frame(self) -> np.ndarray:
        """
        Capture a single frame from the camera. The camera must be
        initialised before calling this method.

        Returns:
            np.ndarray: Captured frame as BGR frame array.

        Raises:
            MicromanipulatorVisionConnectionError: If camera is not
                initialized, frame capture fails, or camera connection
                is lost.

        Example:
            with MicromanipulatorVision() as vision:
                frame = vision.capture_frame()
                cv.imshow("Live Frame", frame)
        """

        self._initialise_camera()

        if not self._camera.isOpened():
            raise MicromanipulatorVisionConnectionError(
                "MicromanipulatorVision.capture_frame: Camera connection lost"
            )

        ret, frame = self._camera.read()

        if not ret:
            raise MicromanipulatorVisionConnectionError(
                "MicromanipulatorVision.capture_frame: Failed to capture frame"
            )

        if frame is None:
            raise MicromanipulatorVisionConnectionError(
                "MicromanipulatorVision.capture_frame: Captured frame is None"
            )
        return frame

    @tested
    def scale_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize an input frame based on the instance's scale factor.

        Scales the input frame using the frame_scale_factor set during
        object initialisation (or uses DEFAULT_FRAME_SCALE_FACTOR).
        This is useful for resizing frames for display or processing
        while maintaining aspect ratio.

        Args:
            frame (np.ndarray): Input frame to be resized.

        Returns:
            np.ndarray: Resized frame.

        Note:
            The scaling uses cv.resize with default interpolation
            (LINEAR). frame_scale_factor > 1 enlarges the frame,
            < 1 shrinks it.
        """

        width = int(frame.shape[1] * self._frame_scale_factor)
        height = int(frame.shape[0] * self._frame_scale_factor)
        resized_frame = cv.resize(frame, (width, height))

        return resized_frame

    @tested
    def rotate_frame(
        self, frame, angle_deg, rotation_point=None
    ) -> np.ndarray:
        """
        Rotate frame by specified angle around given rotation point.

        Rotates the input frame clockwise for positive angles and
        counterclockwise for negative angles. Uses the frame center
        as rotation point if none specified.

        Args:
            frame (np.ndarray): Input frame to rotate.
            angle_deg (float): Rotation angle in degrees. Positive
                values rotate clockwise, negative counterclockwise.
            rotation_point (Tuple[int, int], optional): (x, y)
                coordinates of rotation center. Defaults to frame
                center if None.

        Returns:
            np.ndarray: Rotated frame with same dimensions as input.

        Note:
            Rotation may cause some frame content to be cropped at
            edges. Frame dimensions remain unchanged after rotation.
        """

        (height, width) = frame.shape[:2]

        if rotation_point is None:
            rotation_point = (width // 2, height // 2)

        scale_factor = 1
        rotation_matrix = cv.getRotationMatrix2D(
            rotation_point, angle_deg, scale_factor
        )

        return cv.warpAffine(frame, rotation_matrix, (width, height))

    @tested
    def correct_frame_orientation(
        self, frame: np.ndarray, visualize: bool = False
    ) -> np.ndarray:
        """
        Correct frame orientation using detected or stored angle error.

        Applies rotation correction to align the frame properly. On
        first call, automatically detects the orientation angle and
        stores it for subsequent corrections. Uses the same angle for
        all future corrections unless manually reset.

        Args:
            frame (np.ndarray): Input frame to correct.
            visualize (bool, optional): If True, display the corrected
                frame. Defaults to False.

        Returns:
            np.ndarray: Orientation-corrected frame.

        Raises:
            MicromanipulatorVisionError: If orientation angle cannot be
                detected from the frame.

        Note:
            Orientation angle is detected once and cached. To
            recalculate, delete the _orientation_angle_error attribute
            or call detect_orientation_angle_error() manually.
        """

        # Use stored angle or detect if not available
        if not hasattr(self, "_orientation_angle_error"):
            self._orientation_angle_error = (
                self.detect_orientation_angle_error(frame, visualize)
            )

        rotated = self.rotate_frame(frame, self._orientation_angle_error)

        if visualize:
            cv.imshow("Rotated Frame", rotated)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return rotated

    @tested
    def undistort_frame(
        self, frame: np.ndarray, alpha: float = DEFAULT_ALPHA
    ) -> np.ndarray:
        """
        Undistort an input frame using the camera's calibration data.

        Applies lens distortion correction to the input frame based on
        previously calculated camera matrix and distortion coefficients.

        Args:
            frame (np.ndarray): Input frame to undistort.
            alpha (float, optional): Free scaling parameter. If 0,
                undistorted frame is zoomed and cropped. If 1, all
                pixels are retained. Defaults to DEFAULT_ALPHA.

        Returns:
            np.ndarray: Undistorted frame.

        Raises:
            MicromanipulatorVisionCalibrationError: If calibration data
                is missing.
            ValueError: If alpha is not between 0.0 and 1.0.

        Note:
            Undistortion may crop the frame edges depending on the
                alpha value.
        """

        if self._camera_matrix is None or self._dist_coeffs is None:
            raise MicromanipulatorVisionCalibrationError(
                "MicromanipulatorVision.undistort_frame: "
                "Camera matrix or distortion coefficients are missing."
            )

        if not self.ALPHA_MIN_VALUE <= alpha <= self.ALPHA_MAX_VALUE:
            raise ValueError(
                "MicromanipulatorVision.undistort_frame: Alpha must be "
                f"between 0.0 and 1.0, got {alpha}"
            )

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Get optimal new camera matrix
        camera_matrix_new, _ = cv.getOptimalNewCameraMatrix(
            self._camera_matrix,
            self._dist_coeffs,
            (width, height),
            alpha,
            (width, height),
        )

        # Apply undistortion
        undistorted_frame = cv.undistort(
            frame,
            self._camera_matrix,
            self._dist_coeffs,
            None,
            camera_matrix_new,
        )

        return undistorted_frame

    # -------------------------------------------------------------------------
    # Public interface: detection functions------------------------------------
    # -------------------------------------------------------------------------

    @tested
    def detect_orientation_angle_error(
        self, frame: np.ndarray, visualize: bool = False
    ) -> float:
        """
        Detect orientation angle error from microscope bed edge.

        Uses Hough line detection to find the strongest edge line in the
        frame and calculates the rotation angle needed to correct frame
        orientation.

        Args:
            frame (np.ndarray): Input frame to analyze for orientation.
            visualize (bool, optional): If True, display detected line.
                Defaults to False.

        Returns:
            float: Angle correction in degrees. Positive values indicate
                clockwise rotation needed.

        Raises:
            MicromanipulatorVisionError: If no lines detected for
                orientation calculation.

        Note:
            Caches result in _orientation_result for visualization.
            Angle is normalized to [-90, 90] degree range.
        """

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 50, 150)
        lines = cv.HoughLines(edges, 1, np.pi / 1200, threshold=30)

        if lines is None:
            raise MicromanipulatorVisionError(
                "MicromanipulatorVision.detect_orientation_angle_error: "
                "No lines detected for orientation calculation. Check "
                "position of microscope bed in the camera view."
            )

        # Use the strongest line.
        rho, theta = lines[0][0]

        # Subtract 90 degrees to get the angle of the actual line
        angle_deg = np.degrees(theta - np.pi / 2)

        # Normalize to [-90, 90]
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180

        if visualize:
            visualization_frame = self._visualize_orientation(
                frame, theta, rho
            )
            self._display("Orientation Line", visualization_frame)

        self._orientation_result = (theta, rho)
        return angle_deg

    # TODO understand how it works
    @tested
    def detect_disk(
        self, frame: np.ndarray, visualize: bool = False
    ) -> Tuple[Tuple[int, int], int]:
        """
        Detect white disk in frame using Hough circle transform.

        Uses Gaussian blur and Hough circle detection to find the
        largest circular object, typically the microscope sample disk.
        Can detect partially obscured disks.

        Args:
            frame (np.ndarray): Input frame (BGR format) to analyze.
            visualize (bool, optional): If True, display detected disk.
                Defaults to False.

        Returns:
            Tuple[Tuple[int, int], int]: ((center_x, center_y), radius)
                where coordinates and radius are in pixels.

        Raises:
            ValueError: If no disk detected in the frame.

        Note:
            Caches result in _disk_result. Detection parameters scaled
            by frame_scale_factor for different resolutions.
        """

        # Convert to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv.GaussianBlur(gray, (15, 15), 1)

        # Use Hough Circle Transform to detect the disk.
        circles = cv.HoughCircles(
            blurred,
            cv.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=100,
            param2=100,
            minRadius=int(400 * self._frame_scale_factor),
            maxRadius=int(500 * self._frame_scale_factor),
        )

        if circles is not None:
            # Convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            # Find the largest circle (assuming it's our disk)
            largest_circle = max(circles, key=lambda c: c[2])
            center = (largest_circle[0], largest_circle[1])
            radius = largest_circle[2]

            if visualize:
                visualization_frame = self._visualize_disk(
                    frame, center, radius
                )
                self._display("Detected Disk", visualization_frame)

            self._disk_result = (center, radius)
            return (center, radius)
        else:
            raise ValueError("No disk detected in the frame")

    # TODO understand how it works
    @tested
    def detect_robot_head(
        self, frame: np.ndarray, visualize: bool = False
    ) -> Tuple[int, int]:
        """
        Detect green robot head quadrilateral and return its centroid.

        Uses HSV color masking to isolate green objects, applies
        morphological operations for noise reduction, then finds
        quadrilateral contours and selects the largest valid one.

        Args:
            frame (np.ndarray): Input frame (BGR format) to analyze.
            visualize (bool, optional): If True, display detected head
                with green outline and red centroid. Defaults to False.

        Returns:
            Tuple[int, int]: (centroid_x, centroid_y) coordinates of the
                robot head center in pixels.

        Raises:
            ValueError: If no green objects or quadrilaterals detected.

        Note:
            Caches result in _robot_head_result. Uses HSV thresholds
            defined in class constants for green color detection.
        """

        # Convert BGR to HSV for better color detection
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Create mask for green color
        mask = cv.inRange(
            hsv,
            self.ROBOT_HEAD_HSV_LOWER_THRESHOLD,
            self.ROBOT_HEAD_HSV_UPPER_THRESHOLD,
        )

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            raise ValueError("No green objects detected in the frame")

        # Filter contours by area and find quadrilaterals
        valid_contours = []
        for contour in contours:
            area = cv.contourArea(contour)
            if area > self.ROBOT_HEAD_MIN_AREA:
                # Approximate contour to polygon
                epsilon = self.ROBOT_HEAD_APPROX_EPSILON * cv.arcLength(
                    contour, True
                )
                approx = cv.approxPolyDP(contour, epsilon, True)

                # Check if it's approximately a quadrilateral (4 vertices)
                if len(approx) == 4:
                    valid_contours.append((contour, area))

        if not valid_contours:
            raise ValueError("No green quadrilateral detected in the frame")

        # Select the largest valid quadrilateral
        largest_contour = max(valid_contours, key=lambda x: x[1])[0]

        # Calculate centroid using moments
        moments = cv.moments(largest_contour)
        if moments["m00"] != 0:
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
        else:
            # Fallback: use bounding box center
            x, y, w, h = cv.boundingRect(largest_contour)
            centroid_x = x + w // 2
            centroid_y = y + h // 2

        centroid = (centroid_x, centroid_y)

        if visualize:
            visualization_frame = self._visualize_robot_head(
                frame, centroid, largest_contour
            )
            self._display("Robot Head", visualization_frame)

        self._robot_head_result = (centroid, largest_contour)
        return (centroid_x, centroid_y)

    # TODO test the square size of the rocks
    def detect_rocks(
        self,
        frame: np.ndarray,
        visualize: bool = False,
    ) -> List[Tuple[np.ndarray, Tuple[int, int], float]]:
        """
        Detect dark rock objects within the microscope sample disk.

        Uses grayscale thresholding and contour detection to identify
        dark objects within the disk boundary. Applies morphological
        operations and area filtering to reduce noise.

        Args:
            frame (np.ndarray): Input BGR frame to analyze.
            visualize (bool, optional): If True, display detected rocks
                with cyan rectangles and red centers. Defaults to False.

        Returns:
            List[Tuple[np.ndarray, Tuple[int, int], float]]: List of
                detected rocks. Each tuple contains (min_rect, centroid,
                area).

        Note:
            Automatically calls detect_disk if needed. Returns empty
            list if no rocks detected. Caches results in _rocks_result.
        """

        # Get disk info from cache or detect it
        if self._disk_result is None:
            self.detect_disk(frame, visualize=False)
        disk_center, disk_radius = self._disk_result

        # Create circular mask for the disk area
        mask_disk = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv.circle(mask_disk, disk_center, disk_radius, 255, -1)

        # Convert to grayscale and mask so we look at only the disk then
        # blur to eliminate noise
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_masked = cv.bitwise_and(gray, mask_disk)
        blurred = cv.GaussianBlur(gray_masked, self.ROCK_BLUR_KERNEL, 0)

        # Create binary mask for rock colors (dark grays/blacks). We
        # invert the threshold to make the rocks (objects of interest)
        # white - this is convention.
        _, rock_mask = cv.threshold(
            blurred, self.ROCK_GRAY_UPPER_THRESHOLD, 255, cv.THRESH_BINARY_INV
        )

        # Apply morphological operations to clean up the mask
        kernel = np.ones(
            (self.ROCK_MORPH_KERNEL_SIZE, self.ROCK_MORPH_KERNEL_SIZE),
            np.uint8,
        )
        processed_mask = cv.morphologyEx(rock_mask, cv.MORPH_OPEN, kernel)
        processed_mask = cv.morphologyEx(
            processed_mask, cv.MORPH_CLOSE, kernel
        )

        # Create an even mask to make all pixels outside the disk appear
        # black and consequently make sure there are no disk edge
        # defects. This is done by using a slightly smaller disk
        cleanup_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv.circle(cleanup_mask, disk_center, disk_radius - 4, 255, -1)
        processed_mask = cv.bitwise_and(processed_mask, cleanup_mask)

        cv.imshow("processed mask", processed_mask)

        # Find contours of potential rocks
        contours, _ = cv.findContours(
            processed_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area.
        detected_rocks = []
        for contour in contours:
            area = cv.contourArea(contour)

            # The number of pixels that are
            if (
                int(self.ROCK_MIN_AREA_PIXELS * self._frame_scale_factor**2)
                <= area
                <= int(self.ROCK_MAX_AREA_PIXELS * self._frame_scale_factor**2)
            ):
                # Calculate centroid
                moments = cv.moments(contour)
                if moments["m00"] != 0:
                    centroid_x = int(moments["m10"] / moments["m00"])
                    centroid_y = int(moments["m01"] / moments["m00"])

                    # Get minimum area rectangle (can be rotated)
                    min_rect = cv.minAreaRect(contour)
                    detected_rocks.append(
                        (min_rect, (centroid_x, centroid_y), area)
                    )

        # Return empty list if no rocks detected.
        if not detected_rocks:
            return []

        if visualize:
            visualization_frame = self._visualize_rocks(frame, detected_rocks)
            self._display("Rocks", visualization_frame)

        self._rocks_result = detected_rocks
        return detected_rocks

    def detect_central_arc():
        """
        TODO
        """

        pass

    # TODO need to be very careful with the order of things.
    # TODO add a has been rotated flag
    def visualize_all_detections(
        self, frame: np.ndarray, window_name: str = "All Detections"
    ) -> np.ndarray:
        """
        Visualize all detections that have been called.

        Only shows detections for methods that have been called.
        Uses cached results, no re-computation.
        """
        result_frame = frame.copy()

        # Draw disk if it was detected
        if self._disk_result is not None:
            center, radius = self._disk_result
            cv.circle(result_frame, center, radius, (255, 255, 0), 2)  # Yellow
            cv.circle(result_frame, center, 4, (255, 255, 0), -1)

        # Draw robot head if it was detected
        if self._robot_head_result is not None:
            centroid = self._robot_head_result
            cv.circle(result_frame, centroid, 8, (0, 255, 0), -1)  # Green

        # Draw rocks if they were detected
        if self._rocks_result is not None:
            for min_rect, centroid, area in self._rocks_result:
                box_points = cv.boxPoints(min_rect)
                box_points = np.int32(box_points)
                cv.drawContours(
                    result_frame, [box_points], 0, (0, 0, 255), 2
                )  # Red
                cv.circle(result_frame, centroid, 3, (0, 0, 255), -1)

        cv.imshow(window_name, result_frame)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return result_frame

    def clear_detection_cache(self) -> None:
        """Reset all cached detection results."""
        self._orientation_result = None
        self._disk_result = None
        self._robot_head_result = None
        self._rocks_result = None

    # TODO
    def detect_end_effector_tip():
        pass


# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================


with MicromanipulatorVision(
    frame_scale_factor=0.8, calibration_debug=False
) as vis:
    # while True:
    #     frame = vis.capture_frame()
    #     # undistorted = vis.undistort_frame(frame)
    #     resized_frame = vis.scale_frame(frame)

    #     cv.imshow("Undistorted Video Feed", resized_frame)

    #     # Wait 20ms for keypress; use bitwise mask (0xFF) to extract
    #     # only ASCII bits, exit if 'q' pressed.
    #     if cv.waitKey(1) & 0xFF == ord("q"):
    #         break

    # cv.destroyAllWindows()
    # vis.set_camera_settings()
    frame = vis.capture_frame()
    resized_frame = vis.scale_frame(frame)
    corrected_orientation = vis.correct_frame_orientation(resized_frame, False)
    center, radius = vis.detect_disk(corrected_orientation, False)
    centroid = vis.detect_robot_head(corrected_orientation, False)
    detect_rocks = vis.detect_rocks(corrected_orientation, True)
    print(vis._orientation_angle_error)
