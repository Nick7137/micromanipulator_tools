# TODO get rid of all TODO's
# TODO make file header comment
# TODO make the docstrings good.
# TODO create a visualize everything function and add text to it.
# TODO make all errors have the name of the class in them.
# TODO check for any inconsistencies at all
# TODO refactor the class so that the logical flow is clean - use subclasses
# TODO go through all the code and check if any of it is unused or completely irrelevent
# TODO not focussing properly
# TODO go through the logical flow and ask see if the correcting and resizing should be done automatically
# TODO why is it so slow on startup - optimise for efficiency

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


class VisionError(Exception):
    """
    Base exception for all Vision-related errors.

    This is the parent class for all Vision-specific
    exceptions. Catch this to handle any Vision error
    generically.

    Example:
        try:
            with Vision(0) as camera:
                camera._calibration_solve_constants()
        except VisionError as e:
            print(f"Micromanipulator vision error: {e}")
    """

    @tested
    def __init__(self, message: str) -> None:
        """
        Initialize the VisionError with a descriptive
        message.

        Args:
            message (str): Error description
        """
        super().__init__(message)


class VisionConnectionError(VisionError):
    """
    Exception for camera connection and hardware issues. Derived from
    the VisionError class.
    """

    @tested
    def __init__(self, message: str) -> None:
        super().__init__(message)


class VisionCalibrationError(VisionError):
    """
    Exception for camera calibration processing issues. Derived from
    the VisionError class.
    """

    @tested
    def __init__(self, message: str) -> None:
        super().__init__(message)


class Vision:
    """
    TODO talk about calibration height with tool and the type of camera logi c930e- specs and setting focus.
    """

    # -------------------------------------------------------------------------
    # Class constants----------------------------------------------------------
    # -------------------------------------------------------------------------

    DEFAULT_FRAME_SCALE_FACTOR = 1

    # Hardware measurements
    BASE_TO_DISK_CENTER = 76.4  # mm
    DISK_RADIUS = 74.5

    # Camera constants
    DEFAULT_CAMERA_CHANNEL = 0
    DEFAULT_CAMERA_WIDTH = 1920
    DEFAULT_CAMERA_HEIGHT = 1080
    DEFAULT_FOCUS_LEVEL = 45
    NUM_INIT_FRAMES = 10

    # Calibration specific settings
    CHESSBOARD_SIZE = (9, 6)
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

    # Rock detection constants.
    ROCK_GRAY_UPPER_THRESHOLD = 80
    ROCK_MIN_AREA_PIXELS = 20
    ROCK_MAX_AREA_PIXELS = 5000
    ROCK_BLUR_KERNEL = (3, 3)
    ROCK_MORPH_KERNEL_SIZE = 3

    # Useful color codes
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    CYAN = (255, 255, 0)

    # -------------------------------------------------------------------------
    # Initialization methods---------------------------------------------------
    # -------------------------------------------------------------------------

    @tested
    def __init__(
        self,
        camera_index: int = DEFAULT_CAMERA_CHANNEL,
        frame_scale_factor: float = DEFAULT_FRAME_SCALE_FACTOR,
        calibration_debug: bool = False,
    ) -> None:
        """
        Initialize Vision with camera and calibration
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
            VisionCalibrationError: If calibration
                creation or loading fails.
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

        # Check and create default calibration if missing, then load
        default_path = os.path.join(self._current_dir, self.CALIBRATION_FILE)
        if not os.path.exists(default_path):
            self._calibration_run()
        self._load_calibration_data()

        self._initialize_camera()

    @tested
    def __enter__(self) -> "Vision":
        """
        Enter the runtime context for camera resource management.

        Returns:
            self: The Vision instance
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
        Return a string representation of the Vision
        object.

        Returns:
            str: Human-readable description of the object state
        """

        return f"Vision(camera_index={self._camera_index})"

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
            VisionCalibrationError: If no frames found
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
            raise VisionCalibrationError(
                f"Vision._calibration_load_frames: "
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
            raise VisionCalibrationError(
                "Vision._calibration_load_frames: "
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
            VisionCalibrationError: If no chessboard
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
            raise VisionCalibrationError(
                "Vision._calibration_find_chessboard_corners: "
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
            VisionCalibrationError: If calibration data
                is missing or mismatched between object and frame
                points.
        """

        if len(object_points) == 0 or len(frame_points) == 0:
            raise VisionCalibrationError(
                "Vision._calibration_solve_constants: Cannot "
                "calibrate: no object points or frame points provided."
            )

        if len(object_points) != len(frame_points):
            raise VisionCalibrationError(
                "Vision._calibration_solve_constants: "
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
            object_points, frame_points, frame_size, None, None
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
            VisionCalibrationError: If calibration data
                is incomplete (missing camera matrix or distortion
                coefficients).
        """

        if self._camera_matrix is None or self._dist_coeffs is None:
            raise VisionCalibrationError(
                "Vision._calibration_save_constants: "
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
            VisionCalibrationError: If any step in the
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
            raise VisionCalibrationError(
                f"Vision._calibration_run: "
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
            VisionCalibrationError: If calibration file
                not found or data cannot be loaded/parsed.
        """

        # Build file path
        load_path = os.path.join(self._current_dir, self.CALIBRATION_FILE)

        # Check if file exists
        if not os.path.exists(load_path):
            raise VisionCalibrationError(
                f"Vision._load_calibration_data: "
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
            raise VisionCalibrationError(
                f"Vision._load_calibration_data: "
                f"Failed to load calibration data from {load_path}: {str(e)}"
            )

    # -------------------------------------------------------------------------
    # Private vision methods---------------------------------------------------
    # -------------------------------------------------------------------------

    @tested
    def _initialize_camera(self) -> None:
        """
        Initialize camera connection and set the frame dimensions.

        Opens VideoCapture connection to the specified camera index and
        sets the camera to highest available resolution.

        Raises:
            VisionConnectionError: If camera cannot be
                opened or is not available.
        """

        # Check if already initialized.
        if self._camera is not None:
            return

        self._camera = cv.VideoCapture(self._camera_index + cv.CAP_DSHOW)

        if not self._camera.isOpened():
            self._camera = None
            raise VisionConnectionError(
                f"Failed to open camera {self._camera_index}"
            )

        # Set highest resolution
        self._camera.set(cv.CAP_PROP_FRAME_WIDTH, self.DEFAULT_CAMERA_WIDTH)
        self._camera.set(cv.CAP_PROP_FRAME_HEIGHT, self.DEFAULT_CAMERA_HEIGHT)

        # Disable autofocus and set manual focus.
        self._camera.set(cv.CAP_PROP_AUTOFOCUS, 0)
        self._camera.set(cv.CAP_PROP_FOCUS, self.DEFAULT_FOCUS_LEVEL)

        # Capture a few frames to warm the camera up and allow the focus
        # to set, this is important otherwise it will be fuzzy.
        for i in range(self.NUM_INIT_FRAMES):
            self.capture_frame()

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
        a = np.cos(theta)
        b = np.sin(theta)

        # Get the point on the line closest to the origin
        x0 = a * rho
        y0 = b * rho

        # Large number to span entire window.
        span = 1000
        x1 = int(x0 + span * (-b))
        y1 = int(y0 + span * (a))
        x2 = int(x0 - span * (-b))
        y2 = int(y0 - span * (a))

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

        # Show the center of the disk
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
    # Private helper functions for detection methods---------------------------
    # -------------------------------------------------------------------------

    # TODO
    def _convert_coords_to_polar(
        self,
        frame: np.ndarray,
        coords: Tuple[int, int],
    ) -> Tuple[float, float]:
        """
        Convert Cartesian coordinates to polar coordinates relative to
        robot base.

        Establishes polar origin below the detected disk center using
        physical scaling ratios. Converts coordinates to robot-centric
        polar form with 0° pointing up and positive angles
        counterclockwise.

        Args:
            frame (np.ndarray): Input frame containing disk.
            coords (Tuple[int, int]): (x, y) coordinates to convert.

        Returns:
            Tuple[float, float]: (radius, angle_deg) where 0° = up,
                positive = counterclockwise, range [-180°, 180°].
        """

        # Detect disk to establish reference point
        disk_center, disk_radius = self.detect_disk(frame, visualize=False)
        disk_center_x, disk_center_y = disk_center

        disk_to_origin_scaling = self.BASE_TO_DISK_CENTER / self.DISK_RADIUS

        # Calculate polar origin (vertically below disk center)
        origin_x = disk_center_x
        origin_y = disk_center_y + disk_radius * disk_to_origin_scaling

        # Convert input coordinates to polar
        input_x, input_y = coords

        # Calculate displacement from origin, note, we are also
        # changing the coordinate system with these operations to make
        # y point up from the origin and x point to the right.
        delta_x = input_x - origin_x
        delta_y = origin_y - input_y

        # Calculate radius (distance from origin)
        radius = np.sqrt(delta_x**2 + delta_y**2)

        # Calculate angle in radians, then convert to degrees
        standard_angle_rad = np.arctan2(delta_y, delta_x)
        standard_angle_deg = np.degrees(standard_angle_rad)

        # We need to convert the angle to the coordinate system defined
        # in the schematic.
        # Standard 90° (up) -> Our 0°
        # Standard 180° (left) -> Our +90°
        # Standard 270° (down) -> Our ±180°
        # Standard 0° (right) -> Our -90°
        angle_deg = standard_angle_deg - 90

        return (radius, angle_deg)

    @tested
    def _create_robot_head_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Create binary mask for robot head detection using HSV color
        filtering.

        Converts the input frame to HSV color space and creates a binary
        mask where green robot head regions appear white and all other
        regions appear black. Applies morphological operations to clean
        noise and fill gaps.

        Args:
            frame (np.ndarray): Input BGR frame to process.

        Returns:
            np.ndarray: Binary mask with robot head regions in white
                (255) and background in black (0).
        """

        # Convert BGR to HSV for better color detection
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Create binary mask, green color becomes white and non green
        # becomes black.
        mask = cv.inRange(
            hsv,
            self.ROBOT_HEAD_HSV_LOWER_THRESHOLD,
            self.ROBOT_HEAD_HSV_UPPER_THRESHOLD,
        )

        # Apply morphological operations to remove noise and clean up
        # the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        return mask

    @tested
    def _find_robot_head_contour(
        self, mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Find the largest quadrilateral contour in the robot head mask.

        Analyzes the binary mask to locate contours, filters them by
        area and quadrilateral shape, then returns the largest valid
        contour representing the robot head.

        Args:
            mask (np.ndarray): Binary mask with robot head regions in
                white and background in black.

        Returns:
            Optional[np.ndarray]: Largest quadrilateral contour if
                found, None if no valid contours detected.
        """

        # Find outlines of white regions of mask
        contours, _ = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Filter contours by area and find quadrilaterals
        valid_contours = []
        for contour in contours:
            area = cv.contourArea(contour)
            if area > self.ROBOT_HEAD_MIN_AREA:
                # Approximate contour to polygon
                epsilon = self.ROBOT_HEAD_APPROX_EPSILON * cv.arcLength(
                    contour, True
                )
                # Check if it's approximately a quadrilateral
                approx = cv.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    valid_contours.append((contour, area))

        if not valid_contours:
            return None

        # Return the largest valid contour
        return max(valid_contours, key=lambda x: x[1])[0]

    @tested
    def _find_prismatic_link(
        self, head_rect
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """
        Calculate prismatic link position and dimensions from robot head
        rectangle.

        Uses the robot head's minimum area rectangle to determine the
        position, size, and orientation of the attached prismatic link.
        The link is positioned adjacent to the head based on head angle.

        Args:
            head_rect: Minimum area rectangle from cv.minAreaRect()
                containing ((center_x, center_y), (width, height),
                angle).

        Returns:
            Tuple[Tuple[float, float], Tuple[float, float], float]: Link
                rectangle as ((center_x, center_y), (width, height),
                angle) in same format as cv.minAreaRect().
        """

        head_center, (head_width, head_height), head_angle_deg = head_rect

        # Convert from what cv returns to angles that range from -90 to 90
        # See: https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
        if head_width < head_height:
            head_angle_deg -= 90

        # Calculate link position
        link_offset = head_height
        head_angle_rad = abs(np.radians(head_angle_deg))

        # Determine link center based on head angle
        if head_angle_deg <= 0:
            link_center_x = head_center[0] + link_offset * np.sin(
                head_angle_rad
            )
        else:
            link_center_x = head_center[0] - link_offset * np.sin(
                head_angle_rad
            )

        link_center_y = head_center[1] + link_offset * np.cos(head_angle_rad)
        link_center = (link_center_x, link_center_y)

        # Calculate link dimensions
        link_width = head_width * 0.4
        link_length = head_height * 2

        return (link_center, (link_width, link_length), head_angle_deg)

    # TODO
    def _create_disk_mask(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int], int]:
        """
        Create circular mask for disk area detection.

        Detects the disk and creates a binary mask where only the disk
        area is white (255) and everything else is black (0).

        Args:
            frame (np.ndarray): Input frame to process.

        Returns:
            Tuple[np.ndarray, Tuple[int, int], int]: (mask, disk_center,
                disk_radius) where mask is the binary disk mask,
                disk_center is (x, y) coordinates, and disk_radius is
                radius in pixels.
        """

        disk_center, disk_radius = self.detect_disk(frame, visualize=False)
        mask_disk = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv.circle(mask_disk, disk_center, disk_radius, 255, -1)

        return mask_disk, disk_center, disk_radius

    # TODO
    def _create_rock_detection_mask(
        self,
        frame: np.ndarray,
        disk_mask: np.ndarray,
        disk_center: Tuple[int, int],
        disk_radius: int,
    ) -> np.ndarray:
        """
        Create processed mask for rock detection within disk area.

        Converts frame to grayscale, applies disk mask, blurs to reduce
        noise, thresholds for dark objects, applies morphological
        operations, and creates cleanup mask to remove edge artifacts.

        Args:
            frame (np.ndarray): Input BGR frame.
            disk_mask (np.ndarray): Binary mask of disk area.
            disk_center (Tuple[int, int]): Disk center coordinates.
            disk_radius (int): Disk radius in pixels.

        Returns:
            np.ndarray: Processed binary mask with rock candidates in
                white and background in black.
        """

        # Convert to grayscale and mask to look at only the disk
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_masked = cv.bitwise_and(gray, disk_mask)
        blurred = cv.GaussianBlur(gray_masked, self.ROCK_BLUR_KERNEL, 0)

        # Create binary mask for rock colors (dark grays/blacks)
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

        # Create cleanup mask to remove disk edge defects
        cleanup_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv.circle(cleanup_mask, disk_center, disk_radius - 4, 255, -1)
        processed_mask = cv.bitwise_and(processed_mask, cleanup_mask)

        return processed_mask

    # TODO
    def _filter_rock_contours(
        self, contours: List[np.ndarray], robot_result: Optional[Tuple]
    ) -> List[Tuple[np.ndarray, Tuple[int, int], float]]:
        """
        Filter contours by area and exclude those inside robot area.

        Processes detected contours to find valid rock candidates by
        filtering on area constraints and excluding any rocks that fall
        within the robot's detected area.

        Args:
            contours (List[np.ndarray]): List of detected contours from
                cv.findContours.
            robot_result (Optional[Tuple]): Robot detection result from
                detect_robot_head, or None if no robot detected.

        Returns:
            List[Tuple[np.ndarray, Tuple[int, int], float]]: List of
                tuple rock candidates as (min_rect, centroid, area).
        """

        detected_rocks = []

        for contour in contours:
            area = cv.contourArea(contour)

            # Check if contour area is within valid rock size range
            min_area = int(
                self.ROCK_MIN_AREA_PIXELS * self._frame_scale_factor**2
            )
            max_area = int(
                self.ROCK_MAX_AREA_PIXELS * self._frame_scale_factor**2
            )

            if min_area <= area <= max_area:
                # Calculate centroid
                moments = cv.moments(contour)
                if moments["m00"] != 0:
                    centroid_x = int(moments["m10"] / moments["m00"])
                    centroid_y = int(moments["m01"] / moments["m00"])

                    # Check if rock centroid is NOT inside robot area
                    is_valid_rock = True
                    if robot_result is not None:
                        _, robot_contour = robot_result
                        # Check if centroid is inside robot contour
                        if (
                            cv.pointPolygonTest(
                                robot_contour, (centroid_x, centroid_y), False
                            )
                            >= 0
                        ):
                            # Rock is inside robot area, exclude it
                            is_valid_rock = False

                    if is_valid_rock:
                        # Get minimum area rectangle (can be rotated)
                        min_rect = cv.minAreaRect(contour)
                        detected_rocks.append(
                            (min_rect, (centroid_x, centroid_y), area)
                        )

        return detected_rocks

    # TODO
    def _convert_rocks_to_polar(
        self,
        frame: np.ndarray,
        detected_rocks: List[Tuple[np.ndarray, Tuple[int, int], float]],
    ) -> List[Tuple[Tuple[float, float], float, float]]:
        """
        Convert rock positions to polar coordinates relative to robot
        base.

        Converts each detected rock's centroid from Cartesian to polar
        coordinates using the robot coordinate system where 0° points up
        and positive angles are counterclockwise.

        Args:
            frame (np.ndarray): Input frame containing disk for
                reference.
            detected_rocks (List[Tuple]): List of detected rocks as
                (min_rect, centroid, area) tuples.

        Returns:
            List[Tuple[Tuple[float, float], float, float]]: List of
                rocks as ((radius, angle_deg), orientation_deg, area)
                where radius and angle_deg are polar coordinates,
                orientation_deg is rock orientation, and area is rock
                area in pixels.
        """

        polar_rocks = []

        for min_rect, centroid, area in detected_rocks:
            # Convert centroid to polar coordinates
            polar_coords = self._convert_coords_to_polar(frame, centroid)

            # Extract rock orientation from minimum area rectangle
            _, _, orientation_deg = min_rect

            polar_rocks.append((polar_coords, orientation_deg, area))

        return polar_rocks

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
            VisionCalibrationError: If calibration data
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

        This method initializes the camera, opens the built-in camera
        settings dialog, and displays a live video feed of the
        undistorted camera output. It allows the user to adjust camera
        settings visually and see the results in real-time.

        The method runs until the user presses 'q' to quit.

        For the settings used with the logi c390e camera see
        "resources/camera_settings"

        Raises:
            VisionConnectionError: If camera
                Initialization fails or connection is lost.
            VisionCalibrationError: If undistortion
                fails due to missing calibration data.
        """

        print(
            "Entered settings configuration mode. Press 'q' to exit when done."
        )
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
        initialized before calling this method.

        Returns:
            np.ndarray: Captured frame as BGR frame array.


        Example:
            with Vision() as vision:
                frame = vision.capture_frame()
                cv.imshow("Live Frame", frame)
        """

        ret, frame = self._camera.read()
        return frame

    @tested
    def scale_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize an input frame based on the instance's scale factor.

        Scales the input frame using the frame_scale_factor set during
        object Initialization (or uses DEFAULT_FRAME_SCALE_FACTOR).
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
        TODO
        """

        orientation_angle_error = self.detect_orientation_angle_error(
            frame, visualize
        )

        rotated = self.rotate_frame(frame, orientation_angle_error)

        if visualize:
            self._display("Rotated Frame", rotated)

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
            VisionCalibrationError: If calibration data
                is missing.
            ValueError: If alpha is not between 0.0 and 1.0.

        Note:
            Undistortion may crop the frame edges depending on the
                alpha value.
        """

        if self._camera_matrix is None or self._dist_coeffs is None:
            raise VisionCalibrationError(
                "Vision.undistort_frame: "
                "Camera matrix or distortion coefficients are missing."
            )

        if not self.ALPHA_MIN_VALUE <= alpha <= self.ALPHA_MAX_VALUE:
            raise ValueError(
                "Vision.undistort_frame: Alpha must be "
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
            VisionError: If no lines detected for
                orientation calculation.
        """

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 50, 150)
        lines = cv.HoughLines(edges, 1, np.pi / 3600, threshold=30)

        if lines is None:
            raise VisionError(
                "Vision.detect_orientation_angle_error: "
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

        return angle_deg

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

            return (center, radius)
        else:
            raise ValueError("No disk detected in the frame.")

    # TODO frame has to be oriented correctly before calling this method. either check or bake in.
    def detect_robot_head(
        self, frame: np.ndarray, visualize: bool = False
    ) -> Optional[Tuple[Tuple[float, float], np.ndarray]]:
        """
        TODO
        """

        robot_head_mask = self._create_robot_head_mask(frame)
        robot_head_contour = self._find_robot_head_contour(robot_head_mask)

        if robot_head_contour is None:
            print("Robot head not detected.")
            return None

        # Get the minimum area rectangle to find robot head orientation.
        head_rect = cv.minAreaRect(robot_head_contour)

        # Create link rectangle
        link_rect = self._find_prismatic_link(head_rect)
        link_points = cv.boxPoints(link_rect)
        link_points = np.int32(link_points)

        # Combine head mask and link mask into one contour.
        head_points = cv.boxPoints(head_rect)
        head_points = np.int32(head_points)

        # Create a new mask with both rectangles
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv.fillPoly(combined_mask, [head_points], 255)
        cv.fillPoly(combined_mask, [link_points], 255)

        # Find the combined contour
        combined_contours, _ = cv.findContours(
            combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        combined_contour = max(combined_contours, key=cv.contourArea)

        # Calculate centroid using moments of the head ONLY (not the
        # combined shape which includes the last link).
        moments = cv.moments(robot_head_contour)
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])

        centroid = (centroid_x, centroid_y)

        polar_centroid = self._convert_coords_to_polar(frame, centroid)

        if visualize:
            visualization_frame = self._visualize_robot_head(
                frame, centroid, combined_contour
            )
            self._display("Robot Head and prismatic link", visualization_frame)

        return (polar_centroid, combined_contour)

    # TODO frame has to be oriented correctly before calling this method. either check or bake in.
    # TODO return the box outline of each particle?
    def detect_rocks(
        self,
        frame: np.ndarray,
        visualize: bool = False,
    ) -> Optional[List[Tuple[np.ndarray, Tuple[int, int], float]]]:
        """
        TODO
        """

        disk_mask, disk_center, disk_radius = self._create_disk_mask(frame)

        processed_mask = self._create_rock_detection_mask(
            frame, disk_mask, disk_center, disk_radius
        )

        # Find contours of potential rocks
        contours, _ = cv.findContours(
            processed_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area and detect robot for exclusion
        robot_result = self.detect_robot_head(frame, visualize=False)

        detected_rocks = self._filter_rock_contours(contours, robot_result)

        # Return None if no rocks detected.
        if not detected_rocks:
            return None

        if visualize:
            visualization_frame = self._visualize_rocks(frame, detected_rocks)
            self._display("Rocks", visualization_frame)

        polar_rocks = self._convert_rocks_to_polar(frame, detected_rocks)

        return polar_rocks

    # TODO make this return the centerline arc equation and the max and min scaled by the size of the disk.
    # TODO must ensure the orientation has been corrected before using this function, raise an error if the orientation has not been corrected and this function is called.
    def detect_workspace():
        """
        TODO
        """

        pass

    # TODO add the text to all the detections.
    def visualize_all_detections(
        self, frame: np.ndarray, window_name: str = "All Detections"
    ) -> np.ndarray:
        pass


# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================

if __name__ == "__main__":
    with Vision(frame_scale_factor=0.6, calibration_debug=False) as vis:
        # vis.set_camera_settings()
        frame = vis.capture_frame()
        resized_frame = vis.scale_frame(frame)
        corrected_orientation = vis.correct_frame_orientation(
            resized_frame, False
        )
        center, radius = vis.detect_disk(corrected_orientation, False)
        centroid = vis.detect_robot_head(corrected_orientation, False)
        detect_rocks = vis.detect_rocks(corrected_orientation, False)

        while True:
            frame = vis.capture_frame()
            resized_frame = vis.scale_frame(frame)
            corrected_orientation = vis.correct_frame_orientation(
                resized_frame, False
            )
            cv.imshow("Video", corrected_orientation)

            # Wait 20ms for keypress; use bitwise mask (0xFF) to extract only
            # ASCII bits, exit if 'd' pressed.
            if cv.waitKey(20) & 0xFF == ord("q"):
                break

        cv.destroyAllWindows()
