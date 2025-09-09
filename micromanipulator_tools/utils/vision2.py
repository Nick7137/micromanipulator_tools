# TODO get rid of all TODO's
# TODO make file header comment
# TODO make the docstrings
# TODO make all errors have the name of the class in them.
# TODO check for any inconsistencies at all
# TODO why is it so slow on startup - optimise for efficiency
# TODO check the prefixing for all the errors and make them make sense.
# TODO update the init.py file
# TODO input output of each method.
# TODO maybe do the visualize orientation line.
# TODO put text on the visualizations.
# TODO correct the angle for the robot.

import os
import glob
import time
import threading
import cv2 as cv
import numpy as np
from typing import Optional, Tuple, List


def tested(func):
    """
    Decorator that marks a function or method as tested.

    Adds a 'tested' attribute set to True to indicate the function
    has been confirmed to work as expected.

    Args:
        func (callable): The function or method to mark as tested.

    Returns:
        callable: The original function with a 'tested' attribute.

    Example:
        @tested
        def my_function():
            pass

        # Later check if tested:
        if hasattr(my_function, 'tested'):
            print("Function is tested!")
    """

    func.tested = True
    return func


class VisionBase:
    """
    TODO
    """

    # -------------------------------------------------------------------------
    # Hardware and Physical Constants
    # -------------------------------------------------------------------------

    # Robot workspace measurements (in mm)
    BASE_TO_DISK_CENTER_MM = 76.4
    DISK_RADIUS_MM = 74.5

    # -------------------------------------------------------------------------
    # Camera Configuration Constants
    # -------------------------------------------------------------------------

    # Default camera settings
    DEFAULT_CAMERA_CHANNEL = 0
    DEFAULT_CAMERA_WIDTH = 1920
    DEFAULT_CAMERA_HEIGHT = 1080
    DEFAULT_FOCUS_LEVEL = 45
    NUM_INIT_FRAMES = 10

    # Frame processing
    DEFAULT_FRAME_SCALE_FACTOR = 1.0

    # -------------------------------------------------------------------------
    # Calibration Constants
    # -------------------------------------------------------------------------

    # Calibration file and folder settings
    CHESSBOARD_SIZE = (9, 6)
    RESOURCES_DIR = "resources"
    CAMERA_CALIBRATION_DIR = "camera_calibration"
    CALIBRATION_IMAGES_FOLDER = "calibration_images"
    CALIBRATION_DATA_FILE = "calibration_data.npz"
    CALIBRATION_DISPLAY_TIME_MS = 100
    CALIBRATION_WINDOW_NAME = "Chessboard Detection"

    # Corner refinement parameters for subpixel accuracy
    CORNER_REFINEMENT_MAX_ITERATIONS = 30
    CORNER_REFINEMENT_EPSILON = 0.001
    CORNER_REFINEMENT_WINDOW_SIZE = (11, 11)
    CORNER_REFINEMENT_ZERO_ZONE = (-1, -1)

    # Reprojection error in pixels quality assessment thresholds
    EXCELLENT_CALIBRATION_THRESHOLD = 0.5
    GOOD_CALIBRATION_THRESHOLD = 1.0
    ACCEPTABLE_CALIBRATION_THRESHOLD = 2.0

    # NPZ file keys for calibration data storage
    NPZ_KEY_CAMERA_MATRIX = "camera_matrix"
    NPZ_KEY_DIST_COEFFS = "dist_coeffs"
    NPZ_KEY_REPROJ_ERROR = "reproj_error"

    # Free scaling parameter for undistortion
    DEFAULT_ALPHA = 0.0
    ALPHA_MIN_VALUE = 0.0
    ALPHA_MAX_VALUE = 1.0

    # -------------------------------------------------------------------------
    # Detection Algorithm Constants
    # -------------------------------------------------------------------------

    # Robot head detection (HSV color filtering for green robot head)
    ROBOT_HEAD_HSV_LOWER_THRESHOLD = (40, 50, 50)
    ROBOT_HEAD_HSV_UPPER_THRESHOLD = (80, 255, 255)
    ROBOT_HEAD_MIN_AREA = 500
    ROBOT_HEAD_APPROX_EPSILON = 0.02

    # Rock detection (dark objects on light background)
    ROCK_GRAY_UPPER_THRESHOLD = 60
    ROCK_MIN_AREA_PIXELS = 20
    ROCK_MAX_AREA_PIXELS = 5000
    ROCK_BLUR_KERNEL = (3, 3)
    ROCK_MORPH_KERNEL_SIZE = 3

    # -------------------------------------------------------------------------
    # Visualization Color Constants (BGR format for OpenCV)
    # -------------------------------------------------------------------------

    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    CYAN = (255, 255, 0)
    YELLOW = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    TEXT_SIZE = 0.6
    TEXT_THICKNESS = 1

    # -------------------------------------------------------------------------
    # File Path Management
    # -------------------------------------------------------------------------

    @classmethod
    def _get_current_dir(cls) -> str:
        """
        Get the directory containing the current module.
        """

        return os.path.dirname(__file__)

    @classmethod
    def _get_root_dir(cls) -> str:
        """
        Get the project root directory.
        """

        current_dir = cls._get_current_dir()
        return os.path.dirname(os.path.dirname(current_dir))

    @classmethod
    def _get_calibration_images_dir(cls) -> str:
        """
        Get the full path to calibration images directory.
        """

        return os.path.join(
            cls._get_root_dir(),
            cls.RESOURCES_DIR,
            cls.CAMERA_CALIBRATION_DIR,
            cls.CALIBRATION_IMAGES_FOLDER,
        )

    @classmethod
    def _get_calibration_data_path(cls) -> str:
        """
        Get the full path to calibration data file.
        """

        return os.path.join(cls._get_current_dir(), cls.CALIBRATION_DATA_FILE)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _display_frame(window_name: str, frame: np.ndarray) -> None:
        """
        Display frame in a window and wait for user keypress.

        Shows the frame in a named window, waits for any keypress,
        then closes the window. Useful for debugging and visualization.

        Args:
            window_name (str): Name for the display window
            frame (np.ndarray): Frame to display
        """

        cv.imshow(window_name, frame)
        cv.waitKey(0)
        cv.destroyAllWindows()


class CameraManager(VisionBase):
    """
    TODO
    """

    def __init__(self, camera_index: int = None) -> None:
        """
        Initialize camera manager and connect to camera.
        """

        self._camera_index = camera_index or self.DEFAULT_CAMERA_CHANNEL
        self._initialize_camera()

    def _initialize_camera(self) -> None:
        """
        Initialize camera connection and set the frame dimensions.
        """

        self._camera = cv.VideoCapture(self._camera_index + cv.CAP_DSHOW)

        if not self._camera.isOpened():
            raise VisionConnectionError(
                f"Failed to open camera {self._camera_index}"
            )

        # Set highest resolution
        self._camera.set(cv.CAP_PROP_FRAME_WIDTH, self.DEFAULT_CAMERA_WIDTH)
        self._camera.set(cv.CAP_PROP_FRAME_HEIGHT, self.DEFAULT_CAMERA_HEIGHT)

        # Disable autofocus and set manual focus.
        self._camera.set(cv.CAP_PROP_AUTOFOCUS, 0)
        self._camera.set(cv.CAP_PROP_FOCUS, self.DEFAULT_FOCUS_LEVEL)

        for i in range(self.NUM_INIT_FRAMES):
            self._capture_frame()

    def _capture_frame(self) -> np.ndarray:
        """
        Capture a single frame from camera.
        """
        ret, frame = self._camera.read()
        if not ret:
            raise VisionConnectionError(
                f"Failed to capture frame from camera {self._camera_index}"
            )
        return frame

    def _cleanup(self) -> None:
        """
        Release camera resources.
        """
        self._camera.release()

    def _open_settings_dialog(self) -> None:
        """
        TODO
        """

        self._camera.set(cv.CAP_PROP_SETTINGS, 1)


class ThreadingCameraManager(VisionBase):
    """
    TODO
    """

    def __init__(self, camera_index: int = None):
        """
        TODO
        """

        self._camera_index = camera_index or self.DEFAULT_CAMERA_CHANNEL
        self.stopped = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        # Open the video stream using the default (fast) backend.
        self._camera = cv.VideoCapture(self._camera_index)

        if not self._camera.isOpened():
            raise VisionConnectionError(
                "ThreadingCameraManager: Failed to open camera "
                f"{self._camera_index}."
            )

        # Configure camera properties
        self._configure_camera_properties()

        # Perform an initial, blocking read to populate the first frame.
        # This is CRITICAL to prevent race conditions and startup
        # crashes.
        self.grabbed, self.frame = self._camera.read()
        if not self.grabbed:
            self._camera.release()  # Clean up before raising the error
            raise VisionConnectionError(
                "ThreadingCameraManager: Failed to grab initial frame from "
                f"camera {self._camera_index}."
            )

        # Start the background thread to continuously update the frame.
        self._start_thread()
        print("ThreadingCameraManager started successfully with fast backend.")

    def _configure_camera_properties(self):
        """
        TODO
        """

        self._camera.set(cv.CAP_PROP_FRAME_WIDTH, self.DEFAULT_CAMERA_WIDTH)
        self._camera.set(cv.CAP_PROP_FRAME_HEIGHT, self.DEFAULT_CAMERA_HEIGHT)
        self._camera.set(cv.CAP_PROP_AUTOFOCUS, 0)
        self._camera.set(cv.CAP_PROP_FOCUS, self.DEFAULT_FOCUS_LEVEL)

        # Good Practice: Verify the resolution was set correctly.
        actual_width = self._camera.get(cv.CAP_PROP_FRAME_WIDTH)
        actual_height = self._camera.get(cv.CAP_PROP_FRAME_HEIGHT)
        print(
            f"Attempted to set {self.DEFAULT_CAMERA_WIDTH}x"
            f"{self.DEFAULT_CAMERA_HEIGHT}. "
            f"Actual resolution: {int(actual_width)}x{int(actual_height)}"
        )

    def _start_thread(self):
        """
        TODO
        """

        self._thread = threading.Thread(target=self._update, args=())
        self._thread.daemon = True
        self._thread.start()

    def _update(self):
        """
        TODO
        """

        while not self.stopped:
            # Read the next frame from the stream
            (grabbed, frame) = self._camera.read()

            # If the frame could not be grabbed (e.g., camera
            # disconnected), signal the thread to stop.
            if not grabbed:
                self.stopped = True
                continue

            # Use a lock to ensure thread-safe assignment of the new
            # frame
            with self._lock:
                self.grabbed = grabbed
                self.frame = frame

        # Release the camera resource once the loop has exited
        self._camera.release()
        print("Camera hardware has been released.")

    def _capture_frame(self) -> np.ndarray:
        """
        TODO
        """

        with self._lock:
            # Check if the stream is running and a frame is available
            if not self.grabbed or self.frame is None:
                raise VisionConnectionError(
                    "ThreadingCameraManager: No frame available. "
                    "The stream may be stopped."
                )

            # Return a copy to prevent race conditions if the main
            # thread modifies the frame
            frame_copy = self.frame.copy()

        return frame_copy

    def _cleanup(self):
        """
        TODO
        """

        if self.stopped:
            return

        print("Stopping camera thread...")
        self.stopped = True

        # Wait for the thread to finish its work (which includes
        # releasing the camera)
        if self._thread is not None:
            self._thread.join()
        print("Camera thread stopped and cleaned up.")

    def _open_settings_dialog(self):
        """
        TODO
        """

        self._camera.set(cv.CAP_PROP_SETTINGS, 1)


class CalibrationManager(VisionBase):
    """
    TODO
    """

    def __init__(self, calibration_debug: bool = False) -> None:
        """
        TODO
        """

        self._calibration_debug = calibration_debug
        self._camera_matrix = None
        self._dist_coeffs = None
        self._reprojection_error = None

        # Check and create default calibration if missing, then load
        default_path = self._get_calibration_data_path()
        if not os.path.exists(default_path):
            self._calibration_run()
        self._load_calibration_data()

    def _calibration_load_frames(self):
        """
        TODO
        """

        calibration_dir = self._get_calibration_images_dir()
        frame_path_list = glob.glob(os.path.join(calibration_dir, "*.jpg"))

        if len(frame_path_list) == 0:
            raise VisionCalibrationError(
                f"CalibrationManager._calibration_load_frames: "
                f"No calibration frames found in {calibration_dir}. "
                f"Please add .jpg calibration frames to this directory."
            )

        if self._calibration_debug:
            print(f"Looking for calibration frames in: {calibration_dir}")
            print(f"Found {len(frame_path_list)} frames.")

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
                "CalibrationManager._calibration_load_frames: "
                "No valid calibration frames could be loaded."
            )

        return frames, filenames

    def _calibration_find_chessboard_corners(self, frames, filenames):
        """
        TODO
        """

        term_criteria = (
            cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
            self.CORNER_REFINEMENT_MAX_ITERATIONS,
            self.CORNER_REFINEMENT_EPSILON,
        )

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

            if actual_frame_size is None:
                actual_frame_size = frame_gray.shape[::-1]

            corners_found, corners_original = cv.findChessboardCorners(
                frame_gray, self.CHESSBOARD_SIZE, None
            )
            if corners_found:
                if self._calibration_debug:
                    print("✓ Found chessboard corners")

                object_points_list.append(world_points_template.copy())

                corners_refined = cv.cornerSubPix(
                    frame_gray,
                    corners_original,
                    self.CORNER_REFINEMENT_WINDOW_SIZE,
                    self.CORNER_REFINEMENT_ZERO_ZONE,
                    term_criteria,
                )
                frame_points_list.append(corners_refined)

                if self._calibration_debug:
                    cv.drawChessboardCorners(
                        frame_bgr,
                        self.CHESSBOARD_SIZE,
                        corners_refined,
                        corners_found,
                    )
                    cv.imshow(self.CALIBRATION_WINDOW_NAME, frame_bgr)
                    cv.waitKey(self.CALIBRATION_DISPLAY_TIME_MS)

            else:
                print(f"✗ No chessboard corners found in frame {filename}")

        cv.destroyAllWindows()

        if len(object_points_list) == 0:
            raise VisionCalibrationError(
                "CalibrationManager._calibration_find_chessboard_corners: "
                "No chessboard patterns were detected in any frames."
            )

        print(
            f"Successfully processed {len(object_points_list)} out of "
            f"{len(frames)} frames for calibration. Now doing very "
            "clever maths......."
        )
        return object_points_list, frame_points_list, actual_frame_size

    def _calibration_solve_constants(
        self, object_points, frame_points, frame_size
    ):
        """
        TODO
        """

        if len(object_points) == 0 or len(frame_points) == 0:
            raise VisionCalibrationError(
                "CalibrationManager._calibration_solve_constants: Cannot "
                "calibrate: no object points or frame points provided."
            )

        if len(object_points) != len(frame_points):
            raise VisionCalibrationError(
                "CalibrationManager._calibration_solve_constants: "
                f"Mismatch: {len(object_points)} object point sets vs "
                f"{len(frame_points)} frame point sets."
            )

        (
            reprojection_error,
            camera_matrix,
            dist_coeffs,
            rotation_vecs,
            translation_vecs,
        ) = cv.calibrateCamera(
            object_points, frame_points, frame_size, None, None
        )

        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs
        self._reprojection_error = reprojection_error

    def _calibration_save_constants(self):
        """
        TODO
        """

        if self._camera_matrix is None or self._dist_coeffs is None:
            raise VisionCalibrationError(
                "CalibrationManager._calibration_save_constants: "
                "Calibration data is incomplete. Camera matrix or distortion "
                "coefficients missing."
            )

        save_path = self._get_calibration_data_path()
        np.savez(
            save_path,
            **{
                self.NPZ_KEY_CAMERA_MATRIX: self._camera_matrix,
                self.NPZ_KEY_DIST_COEFFS: self._dist_coeffs,
                self.NPZ_KEY_REPROJ_ERROR: self._reprojection_error,
            },
        )

    def _calibration_run(self):
        """
        TODO
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
                f"CalibrationManager._calibration_run: "
                f"Failed to create calibration: {str(e)}"
            )

    def _load_calibration_data(self):
        """
        TODO
        """

        load_path = self._get_calibration_data_path()

        if not os.path.exists(load_path):
            raise VisionCalibrationError(
                f"CalibrationManager._load_calibration_data: "
                f"Calibration file not found: {load_path}"
            )

        try:
            data = np.load(load_path)

            self._camera_matrix = data[self.NPZ_KEY_CAMERA_MATRIX]
            self._dist_coeffs = data[self.NPZ_KEY_DIST_COEFFS]
            self._reprojection_error = data[self.NPZ_KEY_REPROJ_ERROR]

        except Exception as e:
            raise VisionCalibrationError(
                f"CalibrationManager._load_calibration_data: "
                f"Failed to load calibration data from {load_path}: {str(e)}"
            )

    def dump_calibration_data(self):
        """
        TODO
        """

        file_path = self._get_calibration_data_path()

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


class FrameProcessor(VisionBase):
    """
    TODO
    """

    def __init__(
        self,
        frame_scale_factor: float = None,
        camera_matrix: np.ndarray = None,
        dist_coeffs: np.ndarray = None,
    ) -> None:
        """
        TODO
        """

        if frame_scale_factor is not None:
            self._frame_scale_factor = frame_scale_factor
        else:
            self._frame_scale_factor = self.DEFAULT_FRAME_SCALE_FACTOR

        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs

    def _scale_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        TODO
        """

        if self._frame_scale_factor == 1.0:
            return frame

        width = int(frame.shape[1] * self._frame_scale_factor)
        height = int(frame.shape[0] * self._frame_scale_factor)
        return cv.resize(frame, (width, height))

    def _rotate_frame(
        self, frame: np.ndarray, angle_deg: float, rotation_point: tuple = None
    ) -> np.ndarray:
        """
        TODO
        """

        height, width = frame.shape[:2]
        if rotation_point is None:
            rotation_point = (width // 2, height // 2)

        rotation_matrix = cv.getRotationMatrix2D(
            rotation_point, angle_deg, 1.0
        )
        return cv.warpAffine(frame, rotation_matrix, (width, height))

    def _detect_orientation_angle_error(self, frame: np.ndarray) -> float:
        """
        TODO
        """

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 50, 150)
        lines = cv.HoughLines(edges, 1, np.pi / 3600, threshold=30)

        if lines is None:
            raise VisionDetectionError(
                "No lines detected for orientation calculation. Check "
                "position of microscope bed in the camera view."
            )

        rho, theta = lines[0][0]
        angle_deg = np.degrees(theta - np.pi / 2)

        # Normalize to [-90, 90]
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180

        return angle_deg

    # UNUSED
    def _undistort_frame(
        self, frame: np.ndarray, alpha: float = None
    ) -> np.ndarray:
        """
        TODO
        """

        if alpha is None:
            alpha = self.DEFAULT_ALPHA

        if self._camera_matrix is None or self._dist_coeffs is None:
            raise VisionCalibrationError(
                "CalibrationManager.undistort_frame: "
                "Camera matrix or distortion coefficients are missing."
            )

        if not self.ALPHA_MIN_VALUE <= alpha <= self.ALPHA_MAX_VALUE:
            raise ValueError(
                "CalibrationManager.undistort_frame: Alpha must be "
                f"between 0.0 and 1.0, got {alpha}"
            )

        height, width = frame.shape[:2]

        camera_matrix_new, _ = cv.getOptimalNewCameraMatrix(
            self._camera_matrix,
            self._dist_coeffs,
            (width, height),
            alpha,
            (width, height),
        )

        undistorted_frame = cv.undistort(
            frame,
            self._camera_matrix,
            self._dist_coeffs,
            None,
            camera_matrix_new,
        )

        return undistorted_frame

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        TODO
        """

        try:
            scaled = self._scale_frame(frame)
            orientation_error = self._detect_orientation_angle_error(scaled)
            oriented = self._rotate_frame(scaled, orientation_error)

            return oriented

        except Exception as e:
            raise VisionDetectionError(f"Failed to process frame: {str(e)}")


class ObjectDetector(VisionBase):
    """
    TODO
    """

    def __init__(self, frame_scale_factor: float = None):
        """
        TODO
        """

        if frame_scale_factor is None:
            self._frame_scale_factor = self.DEFAULT_FRAME_SCALE_FACTOR
        else:
            self._frame_scale_factor = frame_scale_factor

    def _detect_disk(self, frame: np.ndarray) -> Tuple[Tuple[int, int], int]:
        """
        TODO
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

            return (center, radius)
        else:
            raise VisionDetectionError("No disk detected in the frame.")

    def _detect_robot(
        self,
        frame: np.ndarray,
        disk_center: Tuple[int, int],
        disk_radius: int,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[int, int], np.ndarray]]:
        """
        TODO
        """

        robot_mask = self._create_robot_head_mask(frame)
        head_contour = self._find_robot_head_contour(robot_mask)

        if head_contour is None:
            return None

        # Calculate centroid using the head-only contour
        centroid_px = self._get_contour_centroid(head_contour)
        if centroid_px is None:
            return None

        # Generate the full body contour for visualization
        head_rect = cv.minAreaRect(head_contour)
        link_rect = self._find_prismatic_link_rect(head_rect)
        body_contour = self._get_robot_body_contour(
            frame.shape, head_rect, link_rect
        )

        # Convert the head's centroid to polar coordinates
        polar_coords = self._convert_cartesian_to_polar(
            disk_center, disk_radius, centroid_px
        )

        # Return polar coords, pixel centroid, and the full body contour
        return (polar_coords, centroid_px, body_contour)

    def _detect_rocks(
        self,
        frame: np.ndarray,
        disk_center: Tuple[int, int],
        disk_radius: int,
        robot_body_contour: Optional[np.ndarray],
    ) -> Optional[
        List[Tuple[Tuple[float, float], Tuple[float, float], float, float]]
    ]:
        """
        TODO
        """

        rock_mask = self._create_rock_mask(frame, disk_center, disk_radius)
        contours, _ = cv.findContours(
            rock_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        rock_data_pixels = self._filter_and_get_rocks_from_contours(
            contours, robot_body_contour
        )

        if not rock_data_pixels:
            return None

        polar_rocks = []
        for min_rect, centroid, area in rock_data_pixels:
            # Unpack the full data from the minimum area rectangle
            (center, (width, height), orientation_deg) = min_rect

            polar_coords = self._convert_cartesian_to_polar(
                disk_center, disk_radius, centroid
            )

            # Append all the information we are interested in regarding
            # the rocks.
            polar_rocks.append(
                (polar_coords, (width, height), orientation_deg, area)
            )

        return polar_rocks

    # _detect_robot helper functions ------------------------------------------

    def _create_robot_head_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        TODO
        """

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(
            hsv,
            self.ROBOT_HEAD_HSV_LOWER_THRESHOLD,
            self.ROBOT_HEAD_HSV_UPPER_THRESHOLD,
        )
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        return mask

    def _find_robot_head_contour(
        self, mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        TODO
        """

        contours, _ = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        valid_contours = []
        for c in contours:
            area = cv.contourArea(c)
            if area > self.ROBOT_HEAD_MIN_AREA:
                epsilon = self.ROBOT_HEAD_APPROX_EPSILON * cv.arcLength(
                    c, True
                )
                approx = cv.approxPolyDP(c, epsilon, True)
                if len(approx) == 4:
                    valid_contours.append((c, area))

        if not valid_contours:
            return None

        # Return the contour with the largest area
        return max(valid_contours, key=lambda x: x[1])[0]

    def _get_contour_centroid(
        self, contour: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """
        TODO
        """

        moments = cv.moments(contour)
        if moments["m00"] == 0:
            return None  # Avoid division by zero
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])
        return (centroid_x, centroid_y)

    def _find_prismatic_link_rect(self, head_rect: Tuple) -> Tuple:
        """
        TODO
        """

        head_center, (head_width, head_height), head_angle_deg = head_rect

        # Convert from what cv returns to angles that range from -90 to 90
        # See: https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
        if head_width < head_height:
            head_angle_deg -= 90

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

        link_width = head_width * 0.4
        link_length = head_height * 2

        return (link_center, (link_width, link_length), head_angle_deg)

    def _get_robot_body_contour(
        self, frame_shape: Tuple, head_rect: Tuple, link_rect: Tuple
    ) -> np.ndarray:
        """
        TODO
        """

        head_points = np.int32(cv.boxPoints(head_rect))
        link_points = np.int32(cv.boxPoints(link_rect))

        # Create a mask by drawing both filled polygons
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv.fillPoly(mask, [head_points], 255)
        cv.fillPoly(mask, [link_points], 255)

        # Find the single external contour of the combined shape
        contours, _ = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        return max(contours, key=cv.contourArea)

    def _convert_cartesian_to_polar(
        self,
        disk_center_px: Tuple[int, int],
        disk_radius_px: int,
        point_px: Tuple[int, int],
    ) -> Tuple[float, float]:
        """
        TODO converts pixel coords to polar coords in the frame of the
        robot
        """

        scaling = self.BASE_TO_DISK_CENTER_MM / self.DISK_RADIUS_MM

        # X coord is in the disk center.
        origin_x_px = disk_center_px[0]
        origin_y_px = disk_center_px[1] + disk_radius_px * scaling

        # # Calculate displacement from origin, note, we are also
        # # changing the coordinate system with these operations to make
        # # y point up from the origin and x point to the right.

        point_new_x_px = point_px[0] - origin_x_px
        point_new_y_px = origin_y_px - point_px[1]

        # Changing the frame to have the x axis pointing up and the y
        # axis pointing to the left.
        radius_px = np.sqrt(point_new_x_px**2 + point_new_y_px**2)

        angle_rad = np.arctan2(point_new_y_px, point_new_x_px)
        angle_deg = np.degrees(angle_rad)

        print(f"disk_radius_px, {disk_radius_px}")

        # Convert to robot's coordinate system (0 deg is up)
        robot_angle_deg = angle_deg - 90
        return (radius_px, robot_angle_deg)

    # _detect_rocks helper functions ------------------------------------------

    def _create_rock_mask(
        self,
        frame: np.ndarray,
        disk_center: Tuple[int, int],
        disk_radius: int,
    ) -> np.ndarray:
        """
        TODO
        """

        # Create a mask for the disk area itself
        disk_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv.circle(disk_mask, disk_center, disk_radius, 255, -1)

        # Convert to grayscale and apply the disk mask
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_masked = cv.bitwise_and(gray, disk_mask)
        blurred = cv.GaussianBlur(gray_masked, self.ROCK_BLUR_KERNEL, 0)

        # Threshold for dark objects (rocks)
        _, rock_mask = cv.threshold(
            blurred, self.ROCK_GRAY_UPPER_THRESHOLD, 255, cv.THRESH_BINARY_INV
        )

        # Clean up the mask with morphology
        kernel = np.ones(
            (self.ROCK_MORPH_KERNEL_SIZE, self.ROCK_MORPH_KERNEL_SIZE),
            np.uint8,
        )
        processed_mask = cv.morphologyEx(rock_mask, cv.MORPH_OPEN, kernel)
        processed_mask = cv.morphologyEx(
            processed_mask, cv.MORPH_CLOSE, kernel
        )

        # Create a cleanup mask to remove artifacts at the disk's edge
        cleanup_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv.circle(cleanup_mask, disk_center, disk_radius - 4, 255, -1)

        return cv.bitwise_and(processed_mask, cleanup_mask)

    def _filter_and_get_rocks_from_contours(
        self, contours: List[np.ndarray], robot_contour: Optional[np.ndarray]
    ) -> List[Tuple[Tuple, Tuple[int, int], float]]:
        """
        TODO
        """

        detected_rocks = []
        min_area = int(self.ROCK_MIN_AREA_PIXELS * self._frame_scale_factor**2)
        max_area = int(self.ROCK_MAX_AREA_PIXELS * self._frame_scale_factor**2)

        for contour in contours:
            area = cv.contourArea(contour)
            if not (min_area <= area <= max_area):
                continue  # Skip if not within size limits

            centroid = self._get_contour_centroid(contour)
            if centroid is None:
                continue

            # Exclude any "rock" whose centroid is inside the robot's body
            if robot_contour is not None:
                if cv.pointPolygonTest(robot_contour, centroid, False) >= 0:
                    continue

            min_rect = cv.minAreaRect(contour)
            detected_rocks.append((min_rect, centroid, area))

        return detected_rocks


class Visualizer(VisionBase):
    """
    TODO
    """

    def _visualize_disk(
        self, frame: np.ndarray, center: Tuple[int, int], radius: int
    ) -> np.ndarray:
        """
        TODO
        """

        vis_frame = frame.copy()
        # Draw the disk outline
        cv.circle(vis_frame, center, radius, self.CYAN, 2)
        # Draw the center point
        cv.circle(vis_frame, center, 4, self.RED, -1)

        # Add the text label
        label_pos = (center[0] - 30, center[1] - radius - 10)

        cv.putText(
            vis_frame,
            f"Disk (r={radius}px)",
            label_pos,
            cv.FONT_HERSHEY_SIMPLEX,
            self.TEXT_SIZE,
            self.CYAN,
            self.TEXT_THICKNESS,
        )

        return vis_frame

    def _visualize_robot(
        self,
        frame: np.ndarray,
        polar_coords: Tuple[float, float],
        centroid_px: Tuple[int, int],
        contour: np.ndarray,
    ) -> np.ndarray:
        """
        TODO
        """

        vis_frame = frame.copy()

        # Draw the contour of the robot body
        cv.drawContours(vis_frame, [contour], -1, self.GREEN, 2)

        # Draw the centroid
        cv.circle(vis_frame, centroid_px, 4, self.RED, -1)

        # Create the label text with the polar coordinates
        radius, angle = polar_coords
        label_text = "Robot"

        # Position and draw the text label
        label_pos = (centroid_px[0] + 10, centroid_px[1])
        cv.putText(
            vis_frame,
            label_text,
            label_pos,
            cv.FONT_HERSHEY_SIMPLEX,
            self.TEXT_SIZE,
            self.RED,
            self.TEXT_THICKNESS,
        )

        return vis_frame

    def _visualize_rocks(
        self,
        frame: np.ndarray,
        rocks_data: List[Tuple[Tuple, Tuple[int, int], float]],
    ) -> np.ndarray:
        """
        TODO
        """
        vis_frame = frame.copy()

        # Use enumerate to get an index 'i' for each rock
        for i, (min_rect, centroid, area) in enumerate(rocks_data):
            # Draw the rotated bounding box
            box_points = np.int32(cv.boxPoints(min_rect))
            cv.drawContours(vis_frame, [box_points], 0, self.CYAN, 2)

            # Draw the centroid
            cv.circle(vis_frame, centroid, 3, self.RED, -1)

            # Create and draw the unique label for each rock
            label_text = f"R{i + 1}"
            label_pos = (centroid[0], centroid[1] - 15)

            cv.putText(
                vis_frame,
                label_text,
                label_pos,
                cv.FONT_HERSHEY_SIMPLEX,
                self.TEXT_SIZE,
                self.RED,
                self.TEXT_THICKNESS,
            )

        return vis_frame

    def _visualize_workspace_arc(
        self,
        frame: np.ndarray,
        arc_center_px: Tuple[int, int],
        arc_radius_px: int,
        disk_center_px: Tuple[int, int],
    ) -> np.ndarray:
        """
        TODO
        """

        vis_frame = frame.copy()

        # Draw the operational arc/circle
        cv.circle(vis_frame, arc_center_px, arc_radius_px, self.MAGENTA, 2)

        # Draw the robot's base origin
        cv.circle(vis_frame, arc_center_px, 8, self.YELLOW, -1)
        cv.circle(vis_frame, arc_center_px, 8, self.BLACK, 2)

        # Draw a line connecting the origin to the disk center to show the radius
        cv.line(
            vis_frame,
            arc_center_px,
            disk_center_px,
            self.YELLOW,
            2,
            cv.LINE_AA,
        )

        # Add text labels for clarity
        cv.putText(
            vis_frame,
            "Robot Origin",
            (arc_center_px[0] + 15, arc_center_px[1]),
            cv.FONT_HERSHEY_SIMPLEX,
            self.TEXT_SIZE,
            self.YELLOW,
            self.TEXT_THICKNESS,
        )
        cv.putText(
            vis_frame,
            f"Arc (r={self.BASE_TO_DISK_CENTER_MM}mm)",
            (disk_center_px[0], disk_center_px[1] - 15),
            cv.FONT_HERSHEY_SIMPLEX,
            self.TEXT_SIZE,
            self.MAGENTA,
            self.TEXT_THICKNESS,
        )

        return vis_frame


class Vision:
    """
    TODO
    """

    def __init__(
        self,
        frame_scale_factor: float = None,
        calibration_debug: bool = False,
        camera_index: int = None,
    ):
        """
        TODO
        """

        self.camera_index = camera_index or VisionBase.DEFAULT_CAMERA_CHANNEL

        # The frame_scale_factor defaults are handled inside the components
        self.frame_scale_factor = (
            frame_scale_factor or VisionBase.DEFAULT_FRAME_SCALE_FACTOR
        )

        # self.camera_manager = CameraManager(camera_index)
        self.camera_manager = ThreadingCameraManager(camera_index)
        self.calibration_manager = CalibrationManager(calibration_debug)
        self.frame_processor = FrameProcessor(
            frame_scale_factor,
            self.calibration_manager._camera_matrix,
            self.calibration_manager._dist_coeffs,
        )
        self.object_detector = ObjectDetector(frame_scale_factor)
        self.visualizer = Visualizer()

    def __enter__(self):
        """
        TODO
        """

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        TODO
        """

        self.camera_manager._cleanup()

    def __str__(self) -> str:
        """
        TODO
        """

        # Query the hardware for the current focus value for accuracy
        try:
            focus_level = self.camera_manager._camera.get(cv.CAP_PROP_FOCUS)
        except (cv.error, AttributeError):
            # Handle cases where camera is not available
            focus_level = "N/A"

        return (
            f"Vision(camera_index={self.camera_index}, "
            f"scale_factor={self.frame_scale_factor:.2f}, "
            f"current_focus={focus_level})"
        )

    def _get_processed_frame(self) -> np.ndarray:
        """
        TODO
        """

        # Capture a new frame
        raw_frame = self.camera_manager._capture_frame()

        # # Undistort the image
        # undistorted = self.frame_processor._undistort_frame(raw_frame)

        # Scale and correct orientation
        # procesqsed_frame = self.frame_processor._process_frame(raw_frame)

        return raw_frame

    def dump_calibration_data(self) -> None:
        """
        TODO
        """

        self.calibration_manager.dump_calibration_data()

    def set_camera_settings(self) -> None:
        """
        TODO
        """

        print(
            "Entered settings configuration mode. Press 'q' to exit when done."
        )

        self.camera_manager._open_settings_dialog()

        while True:
            try:
                # Try to get and show the processed frame
                frame = self._get_processed_frame()
                cv.imshow("Camera Settings Preview", frame)

            except VisionDetectionError as e:
                # If processing fails, show a black "error" frame
                # instead of crashing.
                print(f"Preview Warning: {e}")
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv.putText(
                    error_frame,
                    "Frame Processing Error",
                    (50, 240),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv.imshow("Camera Settings Preview", error_frame)

            # This part remains the same
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

        cv.destroyAllWindows()

    def detect_disk(
        self, visualize: bool = False
    ) -> Tuple[Tuple[int, int], int]:
        """
        TODO
        """

        processed_frame = self._get_processed_frame()

        # Perform detection on the processed frame
        center_px, radius_px = self.object_detector._detect_disk(
            processed_frame
        )

        # Handle visualization if requested
        if visualize:
            vis_frame = self.visualizer._visualize_disk(
                processed_frame, center_px, radius_px
            )
            self.visualizer._display_frame("Disk Detection", vis_frame)

        return center_px, radius_px

    def detect_robot(
        self, visualize: bool = False
    ) -> Optional[Tuple[float, float]]:
        """
        TODO
        """

        processed_frame = self._get_processed_frame()

        # If disk doesn't exist then we are stuffed.
        try:
            disk_center_px, disk_radius_px = self.object_detector._detect_disk(
                processed_frame
            )
        except VisionDetectionError:
            return None

        robot_data = self.object_detector._detect_robot(
            processed_frame, disk_center_px, disk_radius_px
        )

        if robot_data is None:
            print("Robot head not detected.")
            return None

        polar_coords, centroid_px, body_contour = robot_data

        if visualize:
            vis_frame = self.visualizer._visualize_robot(
                processed_frame, polar_coords, centroid_px, body_contour
            )
            self.visualizer._display_frame("Robot Detection", vis_frame)

        # Return only the public-facing polar coordinates
        return polar_coords

    def detect_rocks(
        self, visualize: bool = False
    ) -> Optional[List[Tuple[Tuple[float, float], float, float]]]:
        """
        TODO
        """

        processed_frame = self._get_processed_frame()

        # Detect disk (mandatory dependency)
        disk_center_px, disk_radius_px = self.object_detector._detect_disk(
            processed_frame
        )

        # Detect robot (optional dependency, can be None)
        robot_data = self.object_detector._detect_robot(
            processed_frame, disk_center_px, disk_radius_px
        )

        robot_contour = robot_data[2] if robot_data else None

        # Perform rock detection
        polar_rocks = self.object_detector._detect_rocks(
            processed_frame, disk_center_px, disk_radius_px, robot_contour
        )

        if not polar_rocks:
            return None

        # Handle visualization if requested
        if visualize:
            # For visualization, we need the pixel data, not polar.
            contours, _ = cv.findContours(
                self.object_detector._create_rock_mask(
                    processed_frame, disk_center_px, disk_radius_px
                ),
                cv.RETR_EXTERNAL,
                cv.CHAIN_APPROX_SIMPLE,
            )
            pixel_rocks = (
                self.object_detector._filter_and_get_rocks_from_contours(
                    contours, robot_contour
                )
            )
            if pixel_rocks:
                vis_frame = self.visualizer._visualize_rocks(
                    processed_frame, pixel_rocks
                )
                self.visualizer._display_frame("Rock Detection", vis_frame)

        return polar_rocks

    def detect_workspace():
        pass


# -------------------------------------------------------------------------
# Exception Classes
# -------------------------------------------------------------------------


class VisionError(Exception):
    """
    Base exception for all Vision system errors.

    This is the parent class for all Vision-specific exceptions.
    Catch this to handle any Vision error generically.

    Example:
        try:
            with Vision() as camera:
                robot_pos = camera.detect_robot()
        except VisionError as e:
            print(f"Vision system error: {e}")
    """

    @tested
    def __init__(self, message: str) -> None:
        """
        Initialize the VisionError with a descriptive message.

        Args:
            message (str): Error description with class context
        """
        super().__init__(message)


class VisionConnectionError(VisionError):
    """
    Exception raised for camera connection and hardware issues.

    This includes camera initialization failures, connection timeouts,
    and hardware communication problems.
    """

    @tested
    def __init__(self, message: str) -> None:
        super().__init__(f"CameraManager: {message}")


class VisionCalibrationError(VisionError):
    """
    Exception raised for camera calibration issues.

    This includes calibration file problems, calibration computation
    failures, and calibration data validation errors.
    """

    @tested
    def __init__(self, message: str) -> None:
        super().__init__(f"CalibrationManager: {message}")


class VisionDetectionError(VisionError):
    """
    Exception raised for object detection failures.

    This includes cases where required objects cannot be detected
    or detection algorithms fail to process the input properly.
    """

    @tested
    def __init__(self, message: str) -> None:
        super().__init__(f"ObjectDetector: {message}")


if __name__ == "__main__":
    with Vision(frame_scale_factor=0.6, calibration_debug=False) as vis:
        # vis.detect_disk(True)
        # vis.detect_robot(True)
        # vis.detect_rocks(True)
        # vis.detect_workspace(True)
        # vis.set_camera_settings()
        # vis.dump_calibration_data()
        # print(vis)

        print("\nStarting lag test. Press 'q' to quit.")
        while True:
            try:
                frame = vis._get_processed_frame()
                cv.imshow("Lag Test", frame)

                # Simulate heavy work taking 0.5 seconds
                print("Main loop is 'busy'...")

            except VisionError as e:
                print(f"Error during preview: {e}")
                time.sleep(1)  # Wait a bit before retrying

            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        cv.destroyAllWindows()
