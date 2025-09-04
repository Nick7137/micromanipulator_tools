# TODO get rid of all TODO's
# TODO make file header comment
# TODO make the docstrings good.
# TODO turn into class later.
# TODO note that the width and height of the camera in opencv may not match the size of the photos taken for the calibration

# dont need live just the ability to take a picture rapidly and process it, the big brain algorithm handles ("live")

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
    TODO talk about calibration height with tool and the type of camera logi c930e- specs
    """

    # -------------------------------------------------------------------------
    # Class constants----------------------------------------------------------
    # -------------------------------------------------------------------------

    DEFAULT_FRAME_SCALE_FACTOR = 1

    # Camera constants
    DEFAULT_CAMERA_CHANNEL = 0
    DEFAULT_CAMERA_WIDTH = 1920
    DEFAULT_CAMERA_HEIGHT = 1080

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
    NPZ_KEY_CHESSBOARD_SIZE = "chessboard_size"
    NPZ_KEY_TARGET_WIDTH = "target_width"
    NPZ_KEY_TARGET_HEIGHT = "target_height"

    # Configure the free scaling parameter that controls how much of the
    # original image is kept after undistortion.
    DEFAULT_ALPHA = 0
    ALPHA_MIN_VALUE = 0.0
    ALPHA_MAX_VALUE = 1.0

    # -------------------------------------------------------------------------
    # Initialisation methods---------------------------------------------------
    # -------------------------------------------------------------------------

    # fixed
    def __init__(
        self,
        camera_index: int = DEFAULT_CAMERA_CHANNEL,
        frame_scale_factor: float = DEFAULT_FRAME_SCALE_FACTOR,
        calibration_debug: bool = False,
    ) -> None:
        """
        TODO

                Note:
            Camera remains connected for the lifetime of the object and
            is automatically released in __exit__(). The camera is
            configured for highest quality capture at 1920x1080 resolution.
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

        if self._camera is not None:
            self._camera.release()
            self._camera = None

        print(f"Camera {self._camera_index} closed")

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
    def _calibration_load_images(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load calibration images from the designated calibration folder.

        Searches for .jpg images in the calibration directory and loads
        them into memory for chessboard corner detection.

        Returns:
            Tuple[List[np.ndarray], List[str]]: Loaded images and their
                corresponding filenames.

        Raises:
            MicromanipulatorVisionCalibrationError: If no images found
                or if no images could be loaded.
        """

        # Find calibration images
        calibration_dir = os.path.join(
            self._root_dir,
            "resources",
            "camera_calibration",
            self.CALIBRATION_IMAGES_FOLDER,
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
        filenames = []

        for image_path in image_path_list:
            image = cv.imread(image_path)
            if image is not None:
                images.append(image)
                filenames.append(os.path.basename(image_path))
            else:
                print(f"WARNING: Could not load {image_path}")

        if len(images) == 0:
            raise MicromanipulatorVisionCalibrationError(
                "MicromanipulatorVision._calibration_load_images: "
                "No valid calibration images could be loaded."
            )

        return images, filenames

    @tested
    def _calibration_find_chessboard_corners(
        self, images: List[np.ndarray], filenames: List[str]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
        """
        Detect and refine chessboard corners in calibration images.

        Processes each image to find chessboard patterns, refines corner
        positions to subpixel accuracy, and optionally displays visual
        feedback during debugging.

        Args:
            images (List[np.ndarray]): List of calibration images.
            filenames (List[str]): Corresponding image filenames for
                logging.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
                Object points, image points, and actual image size.

        Raises:
            MicromanipulatorVisionCalibrationError: If no chessboard
                patterns detected in any images.
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
        image_points_list = []
        actual_image_size = None

        for image_bgr, filename in zip(images, filenames):
            if self._calibration_debug:
                print(f"Processing: {filename}")

            image_gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)

            # Set image size from first successful image
            if actual_image_size is None:
                actual_image_size = image_gray.shape[::-1]

            # Find chessboard corners
            corners_found, corners_original = cv.findChessboardCorners(
                image_gray, self.CHESSBOARD_SIZE, None
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
                        self.CHESSBOARD_SIZE,
                        corners_refined,
                        corners_found,
                    )

                    resized_image_bgr = self.scale_frame(
                        image_bgr, self._frame_scale_factor
                    )
                    cv.imshow(self.CALIBRATION_WINDOW_NAME, resized_image_bgr)
                    cv.waitKey(self.CALIBRATION_DISPLAY_TIME_MS)

            else:
                print(f"✗ No chessboard corners found in image {filename}")

        cv.destroyAllWindows()

        if len(object_points_list) == 0:
            raise MicromanipulatorVisionCalibrationError(
                "MicromanipulatorVision._calibration_find_chessboard_corners: "
                "No chessboard patterns were detected in any images."
            )

        print(
            f"Successfully processed {len(object_points_list)} out of "
            f"{len(images)} images for calibration. Now doing very "
            "clever maths......."
        )
        return object_points_list, image_points_list, actual_image_size

    @tested
    def _calibration_solve_constants(
        self,
        object_points: List[np.ndarray],
        image_points: List[np.ndarray],
        image_size: Tuple[int, int],
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
            image_points (List[np.ndarray]): 2D image coordinates of
                detected corners.
            image_size (Tuple[int, int]): Image dimensions
                (width, height).

        Raises:
            MicromanipulatorVisionCalibrationError: If calibration data
                is missing or mismatched between object and image
                points.
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

        Orchestrates the full calibration process: loads images, finds
        chessboard corners, solves calibration parameters, and saves
        results to file.

        Raises:
            MicromanipulatorVisionCalibrationError: If any step in the
                calibration process fails.
        """

        try:
            images, filenames = self._calibration_load_images()
            object_points, image_points, image_size = (
                self._calibration_find_chessboard_corners(images, filenames)
            )
            self._calibration_solve_constants(
                object_points, image_points, image_size
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
        Initialise camera connection and set the image dimensions.

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

        print(f"Camera {self._camera_index} initialised")

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
                    print("⚠ Poor calibration quality - retake images!")
            else:
                print("\nReprojection Error: Not available")

            print(f"{'=' * 50}\n")

        else:
            print(f"\n{'=' * 50}")
            print("            CALIBRATION DATA\n")
            print(f"❌ Calibration file not found: {file_path}")
            print(f"{'=' * 50}\n")

    @tested
    def set_camera_settings(self):
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
    # Public interface: basic frame manipulation-------------------------------
    # -------------------------------------------------------------------------

    @tested
    def capture_frame(self) -> np.ndarray:
        """
        Capture a single frame from the camera. The camera must be
        initialised before calling this method.

        Returns:
            np.ndarray: Captured frame as BGR image array.

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
            (LINEAR). frame_scale_factor > 1 enlarges the image,
            < 1 shrinks it.
        """

        width = int(frame.shape[1] * self._frame_scale_factor)
        height = int(frame.shape[0] * self._frame_scale_factor)
        resized_frame = cv.resize(frame, (width, height))

        return resized_frame

    # TODO make pretty
    def rotate_frame(self, image, angle_deg, rotation_point=None):
        """
        TODO
        # +ve => clockwise
        # -ve => anticlockwise
        """

        (height, width) = image.shape[:2]

        if rotation_point is None:
            rotation_point = (width // 2, height // 2)

        rotation_matrix = cv.getRotationMatrix2D(
            rotation_point, angle_deg, 1.0
        )

        return cv.warpAffine(image, rotation_matrix, (width, height))

    # TODO make the angle an internal parameter. maybe this should be in init.
    def correct_frame_orientation(self, image):
        """
        TODO
        """

        angle_deg = self.detect_orientation_angle_error(image)
        return self.rotate_frame(image, angle_deg)

    @tested
    def undistort_frame(
        self, image: np.ndarray, alpha: float = DEFAULT_ALPHA
    ) -> np.ndarray:
        """
        Undistort an input image using the camera's calibration data.

        Applies lens distortion correction to the input image based on
        previously calculated camera matrix and distortion coefficients.

        Args:
            image (np.ndarray): Input image to undistort.
            alpha (float, optional): Free scaling parameter. If 0,
                undistorted image is zoomed and cropped. If 1, all
                pixels are retained. Defaults to DEFAULT_ALPHA.

        Returns:
            np.ndarray: Undistorted image.

        Raises:
            MicromanipulatorVisionCalibrationError: If calibration data
                is missing.
            ValueError: If alpha is not between 0.0 and 1.0.

        Note:
            Undistortion may crop the image edges depending on the
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

        # Get image dimensions
        height, width = image.shape[:2]

        # Get optimal new camera matrix
        camera_matrix_new, _ = cv.getOptimalNewCameraMatrix(
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

    # -------------------------------------------------------------------------
    # Public interface: detection functions------------------------------------
    # -------------------------------------------------------------------------

    # TODO magic numbers
    def detect_disk(
        self, image: np.ndarray, visualize: bool = False
    ) -> Tuple[Tuple[int, int], int]:
        """
        Detect a white disk in the image, even if partially obscured,
        and optionally visualise the result.

        Args:
            image (np.ndarray): Input image (BGR format) containing the white disk.
            visualise (bool, optional): If True, display the image with
                detected disk. Defaults to False.

        Returns:
            Tuple[Tuple[int, int], int]: ((center_x, center_y), radius)
                center_x, center_y: Coordinates of the disk's center in pixels.
                radius: Radius of the detected disk in pixels.

        Raises:
            ValueError: If no disk is detected in the image.

        Note:
            This method can detect the disk even if parts of it are covered.
            Adjust parameters if detection is not accurate for your setup.
        """
        # Convert to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

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
                # Draw the result on the image
                cv.circle(image, center, radius, (0, 255, 0), 2)
                cv.circle(image, center, 2, (0, 0, 255), 3)
                cv.imshow("Detected Disk", image)
                cv.waitKey(0)
                cv.destroyAllWindows()

            return (center, radius)
        else:
            raise ValueError("No disk detected in the image")

    # TODO magic numbers
    # TODO add error handling if angle makes no sense or strip not found
    # TODO on startup add code which gets the user to confirm the orientation before starting.
    # TODO check the maths is correct and works.
    # TODO get grey and edges and final image for the presentation
    def detect_orientation_angle_error(self, image, visualize=False):
        """
        TODO
        """

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 50, 150)
        lines = cv.HoughLines(edges, 1, np.pi / 10000, threshold=30)

        if lines is None:
            return None

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
            # Calculate line endpoints for visualization
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv.imshow("Detected Line", image)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return angle_deg


# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================

with MicromanipulatorVision(
    frame_scale_factor=0.6, calibration_debug=False
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

    # frame = vis.capture_frame()
    # undistorted = vis.undistort_frame(frame)
    # resized_frame = vis.scale_frame(undistorted)
    # center, radius = vis.detect_disk(resized_frame, True)

    frame = vis.capture_frame()
    resized_frame = vis.scale_frame(frame)
    # center, radius = vis.detect_disk(resized_frame, True)
    corrected_orientation = vis.correct_frame_orientation(resized_frame)

    cv.imshow("correct", corrected_orientation)
    cv.waitKey(0)

    # print(radius)
