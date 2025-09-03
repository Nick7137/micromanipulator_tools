# TODO get rid of all TODO's
# TODO make file header comment
# TODO make the docstrings good.
# TODO turn into class later.

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
    TODO talk about calibration height with tool and the type of camera
    """

    # -------------------------------------------------------------------------
    # Class constants----------------------------------------------------------
    # -------------------------------------------------------------------------

    DEFAULT_CAMERA_CHANNEL = 0

    # Camera resolution constants
    DEFAULT_CAMERA_WIDTH = 1920
    DEFAULT_CAMERA_HEIGHT = 1080

    # (num_rows, num_cols) num inner corners
    DEFAULT_CHESSBOARD_SIZE = (9, 6)

    # Camera focus specific settings
    DEFAULT_FOCUS_FOLDER = "focus_20%"
    SECONDARY_FOCUS_FOLDER = "focus_30%"
    DEFAULT_FOCUS_FILENAME = "camera_calibration_20%.npz"
    SECONDARY_FOCUS_FILENAME = "camera_calibration_30%.npz"
    DEFAULT_FOCUS_LEVEL = 20
    SECONDARY_FOCUS_LEVEL = 30

    # Calibration quality thresholds
    EXCELLENT_CALIBRATION_THRESHOLD = 0.5
    GOOD_CALIBRATION_THRESHOLD = 1.0
    ACCEPTABLE_CALIBRATION_THRESHOLD = 2.0

    # Corner refinement parameters
    CORNER_REFINEMENT_MAX_ITERATIONS = 30
    CORNER_REFINEMENT_EPSILON = 0.001
    CORNER_REFINEMENT_WINDOW_SIZE = (11, 11)
    CORNER_REFINEMENT_ZERO_ZONE = (-1, -1)

    # Configure the free scaling parameter that controls how much of the
    # original image is kept after undistortion.
    DEFAULT_ALPHA = 1.0
    ALPHA_MIN_VALUE = 0.0
    ALPHA_MAX_VALUE = 1.0

    # Visual calibration constants
    CALIBRATION_DISPLAY_TIME_MS = 300
    CALIBRATION_WINDOW_NAME = "Chessboard"

    # Calibration constant keys
    NPZ_KEY_CAMERA_MATRIX = "camera_matrix"
    NPZ_KEY_DIST_COEFFS = "dist_coeffs"
    NPZ_KEY_ROTATION_VECS = "rotation_vecs"
    NPZ_KEY_TRANSLATION_VECS = "translation_vecs"
    NPZ_KEY_REPROJECTION_ERROR = "reprojection_error"
    NPZ_KEY_CHESSBOARD_SIZE = "chessboard_size"
    NPZ_KEY_TARGET_WIDTH = "target_width"
    NPZ_KEY_TARGET_HEIGHT = "target_height"

    # -------------------------------------------------------------------------
    # Initialisation methods---------------------------------------------------
    # -------------------------------------------------------------------------

    def __init__(
        self,
        camera_index: int = DEFAULT_CAMERA_CHANNEL,
        calibration_debug: bool = False,
    ) -> None:
        """
        TODO
        """

        # Store configuration parameters
        self._camera_index = camera_index
        self._calibration_debug = calibration_debug
        self._chessboard = self.DEFAULT_CHESSBOARD_SIZE
        self._target_width = self.DEFAULT_CAMERA_WIDTH
        self._target_height = self.DEFAULT_CAMERA_HEIGHT

        # Camera state
        self._capture: Optional[cv.VideoCapture] = None
        self._is_camera_initialised = False
        self._current_focus_level = self.DEFAULT_FOCUS_LEVEL

        # Calibration state
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None
        self._rotation_vecs: Optional[List[np.ndarray]] = None
        self._translation_vecs: Optional[List[np.ndarray]] = None
        self._reprojection_error: Optional[float] = None
        self._is_calibrated = False

        # Ensure both calibration files exist, create if missing, and
        # then load the default data.
        self._ensure_calibration_files_exist()
        self._load_calibration_data()

    @tested
    def __enter__(self) -> "MicromanipulatorVision":
        """
        Enter the runtime context for camera resource management.

        Returns:
            self: The MicromanipulatorVision instance
        """

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the runtime context and cleanup camera resources.

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)
        """
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._is_camera_initialised = False

    def __str__(self) -> str:
        """
        Return a string representation of the MicromanipulatorVision
        object.

        Returns:
            str: Human-readable description of the object state
        """

        return f"MicromanipulatorVision(camera_index={self._camera_index})"

    # -------------------------------------------------------------------------
    # Private Methods----------------------------------------------------------
    # -------------------------------------------------------------------------

    def _calibration_load_images(self, subfolder: str) -> List[np.ndarray]:
        """
        Load calibration images from the expected directory structure.

        Args:
            subfolder: Subfolder within calibration_images to use

        Returns:
            List of loaded images (BGR format)

        Raises:
            MicromanipulatorVisionCalibrationError: If no images found
        """

        # Find calibration images
        current_dir = os.path.dirname(__file__)
        root_dir = os.path.dirname(os.path.dirname(current_dir))
        calibration_dir = os.path.join(
            root_dir, "resources", "calibration_images", subfolder
        )

        image_path_list = glob.glob(os.path.join(calibration_dir, "*.jpg"))

        if len(image_path_list) == 0:
            raise MicromanipulatorVisionCalibrationError(
                f"MicromanipulatorVision._calibration_load_images: "
                f"No calibration images found in {calibration_dir}. "
                f"Please add .jpg calibration images to this directory."
            )

        if self._calibration_debug:
            print(f"Looking for calibration images in: {calibration_dir}")
            print(f"Found {len(image_path_list)} images.")

        # Load all images
        images = []
        for image_path in image_path_list:
            image = cv.imread(image_path)
            if image is not None:
                images.append(image)
            else:
                print(f"WARNING: Could not load {image_path}")

        if len(images) == 0:
            raise MicromanipulatorVisionCalibrationError(
                "MicromanipulatorVision._calibration_load_images: "
                "No valid calibration images could be loaded."
            )

        return images

    def _calibration_find_chessboard_corners(
        self, images: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Find chessboard corners in the given images.

        Args:
            images: List of input images (BGR format)

        Returns:
            Tuple of (object_points_list, image_points_list)
            - object_points_list: 3D points in real world space
            - image_points_list: 2D points in image plane

        Raises:
            MicromanipulatorVisionCalibrationError: If no corners found
                in any image.
        """

        # Termination criteria for corner refinement
        term_criteria = (
            cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
            self.CORNER_REFINEMENT_MAX_ITERATIONS,
            self.CORNER_REFINEMENT_EPSILON,
        )

        # Create object points (3D coordinates of chessboard corners)
        num_rows, num_cols = self._chessboard
        world_points_template = np.zeros((num_rows * num_cols, 3), np.float32)
        world_points_template[:, :2] = np.mgrid[
            0:num_rows, 0:num_cols
        ].T.reshape(-1, 2)

        object_points_list = []
        image_points_list = []
        self._actual_image_size = (
            None  # Store actual image size as instance variable
        )

        for i, image_bgr in enumerate(images):
            if self._calibration_debug:
                print(f"Processing: image {i + 1}")

            image_gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)

            # Set image size from first successful image
            if self._actual_image_size is None:
                self._actual_image_size = image_gray.shape[
                    ::-1
                ]  # (width, height)

            # Find chessboard corners
            corners_found, corners_original = cv.findChessboardCorners(
                image_gray, self._chessboard, None
            )
            if corners_found:
                if self._calibration_debug:
                    print("✓ Found chessboard corners")

                object_points_list.append(world_points_template.copy())

                # Refine corner positions to subpixel accuracy
                corners_refined = cv.cornerSubPix(
                    image_gray,
                    corners_original,
                    self.CORNER_REFINEMENT_WINDOW_SIZE,
                    self.CORNER_REFINEMENT_ZERO_ZONE,
                    term_criteria,
                )
                image_points_list.append(corners_refined)

                # Show visual feedback if debug mode is enabled
                if self._calibration_debug:
                    cv.drawChessboardCorners(
                        image_bgr,
                        self._chessboard,
                        corners_refined,
                        corners_found,
                    )
                    cv.imshow(self.CALIBRATION_WINDOW_NAME, image_bgr)
                    cv.waitKey(self.CALIBRATION_DISPLAY_TIME_MS)

            else:
                print(f"✗ No chessboard corners found in image {i + 1}")

        cv.destroyAllWindows()

        if len(object_points_list) == 0:
            raise MicromanipulatorVisionCalibrationError(
                "MicromanipulatorVision._calibration_find_chessboard_corners: "
                "No chessboard patterns were detected in any images."
            )

        print(
            f"Successfully processed {len(object_points_list)} out of "
            f"{len(images)} images"
        )
        return object_points_list, image_points_list

    def _calibration_solve_constants(
        self, object_points: List[np.ndarray], image_points: List[np.ndarray]
    ) -> None:
        """
        Perform camera calibration using object and image points.

        Args:
            object_points: List of 3D points in real world space
            image_points: List of 2D points in image plane

        Raises:
            MicromanipulatorVisionCalibrationError: If calibration fails
        """

        if len(object_points) == 0 or len(image_points) == 0:
            raise MicromanipulatorVisionCalibrationError(
                "MicromanipulatorVision._calibration_solve_constants: Cannot "
                "calibrate: no object points or image points provided."
            )

        if len(object_points) != len(image_points):
            raise MicromanipulatorVisionCalibrationError(
                "MicromanipulatorVision._calibration_solve_constants: "
                f"Mismatch: {len(object_points)} object point sets vs "
                f"{len(image_points)} image point sets."
            )

        # Use actual image size from the images, fallback to target size if not available
        if (
            hasattr(self, "_actual_image_size")
            and self._actual_image_size is not None
        ):
            image_size = self._actual_image_size
        else:
            image_size = (self._target_width, self._target_height)

        # Perform OpenCV camera calibration
        (
            reprojection_error,
            camera_matrix,
            dist_coeffs,
            rotation_vecs,
            translation_vecs,
        ) = cv.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,  # Initial camera matrix
            None,  # Initial distortion coefficients
        )

        # Store calibration results
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs
        self._rotation_vecs = rotation_vecs
        self._translation_vecs = translation_vecs
        self._reprojection_error = reprojection_error
        self._is_calibrated = True

        self.dump_calibration_data()

    def _calibration_save_constants(self, filename: str) -> None:
        """
        Save the calibration results to a file.

        Args:
            filename: Name of the file to save calibration data to

        Raises:
            MicromanipulatorVisionCalibrationError: If no calibration
                data to save
        """

        if not self._is_calibrated:
            raise MicromanipulatorVisionCalibrationError(
                "MicromanipulatorVision._calibration_save_constants: "
                "No calibration data to save. Run calibration first."
            )

        if self._camera_matrix is None or self._dist_coeffs is None:
            raise MicromanipulatorVisionCalibrationError(
                "MicromanipulatorVision._calibration_save_constants: "
                "Calibration data is incomplete. Camera matrix or distortion "
                "coefficients missing."
            )

        # Create save path (same directory as the script)
        current_dir = os.path.dirname(__file__)
        save_path = os.path.join(current_dir, filename)

        # Save all calibration data
        np.savez(
            save_path,
            **{
                self.NPZ_KEY_CAMERA_MATRIX: self._camera_matrix,
                self.NPZ_KEY_DIST_COEFFS: self._dist_coeffs,
                self.NPZ_KEY_ROTATION_VECS: self._rotation_vecs,
                self.NPZ_KEY_TRANSLATION_VECS: self._translation_vecs,
                self.NPZ_KEY_REPROJECTION_ERROR: self._reprojection_error,
                self.NPZ_KEY_CHESSBOARD_SIZE: np.array(self._chessboard),
                self.NPZ_KEY_TARGET_WIDTH: self._target_width,
                self.NPZ_KEY_TARGET_HEIGHT: self._target_height,
            },
        )

    def _calibration_run(
        self, folder: str, filename: str, focus_level: int
    ) -> None:
        """
        Create a calibration file by running calibration on images.

        Args:
            folder: Folder containing calibration images
            filename: Filename to save calibration results
            focus_level: Focus level being calibrated.

        Raises:
            MicromanipulatorVisionCalibrationError: If calibration fails
        """
        try:
            images = self._calibration_load_images(folder)
            object_points, image_points = (
                self._calibration_find_chessboard_corners(images)
            )
            self._calibration_solve_constants(object_points, image_points)
            self._calibration_save_constants(filename)

        except Exception as e:
            raise MicromanipulatorVisionCalibrationError(
                f"MicromanipulatorVision._calibration_run: "
                f"Failed to create {focus_level}% calibration: {str(e)}"
            )

    def _ensure_calibration_files_exist(self) -> None:
        """
        Ensure both calibration files exist, creating them if missing.

        Raises:
            MicromanipulatorVisionCalibrationError: If calibration
                cannot be completed for either focus level
        """

        current_dir = os.path.dirname(__file__)

        # Check and create default calibration if missing
        default_path = os.path.join(current_dir, self.DEFAULT_FOCUS_FILENAME)
        if not os.path.exists(default_path):
            self._calibration_run(
                self.DEFAULT_FOCUS_FOLDER,
                self.DEFAULT_FOCUS_FILENAME,
                self.DEFAULT_FOCUS_LEVEL,
            )

        # Check and create secondary calibration if missing
        secondary_path = os.path.join(
            current_dir, self.SECONDARY_FOCUS_FILENAME
        )
        if not os.path.exists(secondary_path):
            self._calibration_run(
                self.SECONDARY_FOCUS_FOLDER,
                self.SECONDARY_FOCUS_FILENAME,
                self.SECONDARY_FOCUS_LEVEL,
            )

    def _load_calibration_data(self) -> None:
        """
        Load calibration for the current focus level.

        Raises:
            MicromanipulatorVisionCalibrationError: If calibration
                cannot be loaded.
        """

        # Determine filename based on current focus level
        if self._current_focus_level == self.DEFAULT_FOCUS_LEVEL:
            filename = self.DEFAULT_FOCUS_FILENAME
        elif self._current_focus_level == self.SECONDARY_FOCUS_LEVEL:
            filename = self.SECONDARY_FOCUS_FILENAME
        else:
            raise MicromanipulatorVisionCalibrationError(
                f"MicromanipulatorVision._load_calibration_data: "
                f"Invalid focus level: {self._current_focus_level}"
            )

        # Build file path
        current_dir = os.path.dirname(__file__)
        load_path = os.path.join(current_dir, filename)

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
            self._is_calibrated = True

        except Exception as e:
            raise MicromanipulatorVisionCalibrationError(
                f"MicromanipulatorVision._load_calibration_data: "
                f"Failed to load calibration from {load_path}: {str(e)}"
            )

    # -------------------------------------------------------------------------
    # Public interface---------------------------------------------------------
    # -------------------------------------------------------------------------

    def switch_focus_level(self, focus_level: int) -> None:
        """
        Switch to a different focus level and load its calibration.

        Args:
            focus_level: The focus level to switch to, either:
                DEFAULT_FOCUS_LEVEL or SECONDARY_FOCUS_LEVEL

        Raises:
            MicromanipulatorVisionCalibrationError: If focus level is
                invalid or calibration file doesn't exist.
        """

        if focus_level not in [
            self.DEFAULT_FOCUS_LEVEL,
            self.SECONDARY_FOCUS_LEVEL,
        ]:
            raise MicromanipulatorVisionCalibrationError(
                f"MicromanipulatorVision.switch_focus_level: "
                f"Invalid focus level {focus_level}. Must be "
                f"{self.DEFAULT_FOCUS_LEVEL} or {self.SECONDARY_FOCUS_LEVEL}"
            )

        # Return early if already at this focus level
        if focus_level == self._current_focus_level:
            return

        # Check if calibration file exists
        if focus_level == self.DEFAULT_FOCUS_LEVEL:
            filename = self.DEFAULT_FOCUS_FILENAME
        else:
            filename = self.SECONDARY_FOCUS_FILENAME

        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, filename)

        if not os.path.exists(file_path):
            raise MicromanipulatorVisionCalibrationError(
                f"MicromanipulatorVision.switch_focus_level: "
                f"Calibration file for {focus_level}% focus not "
                f"found: {file_path}"
            )

        # Update focus level and load calibration
        self._current_focus_level = focus_level
        self._load_calibration_data()

    def undistort_image(
        self, image: np.ndarray, alpha: float = DEFAULT_ALPHA
    ) -> np.ndarray:
        """
        Apply undistortion to an image using the calibrated camera
        parameters.

        Args:
            image: Input image to undistort (BGR format)
            alpha: Free scaling parameter (0.0 = crop to valid pixels,
                1.0 = keep all pixels)

        Returns:
            np.ndarray: Undistorted image

        Raises:
            MicromanipulatorVisionCalibrationError: If camera is not
                calibrated
            ValueError: If alpha is not between 0.0 and 1.0
        """
        if not self._is_calibrated:
            raise MicromanipulatorVisionCalibrationError(
                "MicromanipulatorVision.undistort_image: "
                "Camera not calibrated. Run calibration or load "
                "calibration data first."
            )

        if self._camera_matrix is None or self._dist_coeffs is None:
            raise MicromanipulatorVisionCalibrationError(
                "MicromanipulatorVision.undistort_image: "
                "Camera matrix or distortion coefficients are missing."
            )

        if not self.ALPHA_MIN_VALUE <= alpha <= self.ALPHA_MAX_VALUE:
            raise ValueError(
                "MicromanipulatorVision.undistort_image: Alpha must be "
                f"between 0.0 and 1.0, got {alpha}"
            )

        # Get image dimensions
        height, width = image.shape[:2]

        # Get optimal new camera matrix
        camera_matrix_new, roi = cv.getOptimalNewCameraMatrix(
            self._camera_matrix,
            self._dist_coeffs,
            (width, height),
            alpha,
            (width, height),
        )

        # Apply undistortion
        undistorted_image = cv.undistort(
            image,
            self._camera_matrix,
            self._dist_coeffs,
            None,
            camera_matrix_new,
        )

        return undistorted_image

    def dump_calibration_data(self) -> None:
        """
        Display detailed calibration information to console.

        Shows camera matrix, distortion coefficients, reprojection
        error, and quality assessment. Used for debugging and
        verification.

        Raises:
            MicromanipulatorVisionCalibrationError: If no calibration
                data is available to display
        """

        if not self._is_calibrated:
            raise MicromanipulatorVisionCalibrationError(
                "MicromanipulatorVision.dump_calibration_data: "
                "No calibration data available to display."
            )

        if self._camera_matrix is None or self._dist_coeffs is None:
            raise MicromanipulatorVisionCalibrationError(
                "MicromanipulatorVision.dump_calibration_data: "
                "Calibration data is incomplete."
            )

        # Display calibration results
        print("==============================================================")
        print("Camera Matrix:")
        print(self._camera_matrix)
        print("\nDistortion Coefficients:")
        print(self._dist_coeffs)

        if self._reprojection_error is not None:
            print(
                f"\nReprojection Error: {self._reprojection_error:.4f} pixels"
            )
            # Evaluate calibration quality
            if self._reprojection_error < self.EXCELLENT_CALIBRATION_THRESHOLD:
                print("✓ Excellent calibration quality!")
            elif self._reprojection_error < self.GOOD_CALIBRATION_THRESHOLD:
                print("✓ Very good calibration quality")
            elif (
                self._reprojection_error
                < self.ACCEPTABLE_CALIBRATION_THRESHOLD
            ):
                print("⚠ OK calibration quality (acceptable)")
            else:
                print("⚠ Poor calibration quality - consider retaking images")
        else:
            print("\nReprojection Error: Not available")

        print("==============================================================")


with MicromanipulatorVision() as vis:
    pass
