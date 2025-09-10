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
# TODO make the orientation correction work for different scale factors

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
    NUM_INIT_FRAMES = 15

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

    ROBOT_TWEEZER_HEIGHT_FACTOR = 1.15
    ROBOT_TWEEZER_BASE_WIDTH_FACTOR = 0.5

    # Rock detection (dark objects on light background)
    ROCK_GRAY_UPPER_THRESHOLD = 100
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
    # Interactive Setup Procedure Constants
    # -------------------------------------------------------------------------

    # Window name for the setup procedure
    SETUP_WINDOW_NAME = "Disk Positioning Setup"

    # Target position for the disk center as a fraction of frame
    # dimensions. Horizontally centered (0.5), and vertically tunable
    # (e.g., 0.6 means 60% from the top).
    SETUP_TARGET_Y_POSITION_FACTOR = 0.54

    # Tolerances for the setup to be considered "OK"
    SETUP_POSITION_TOLERANCE_PX = 5
    SETUP_ANGLE_TOLERANCE_DEG = 0.2

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

    def _rotate_point(
        self, point: Tuple[int, int], center: Tuple[int, int], angle_deg: float
    ) -> Tuple[int, int]:
        """
        TODO
        """

        angle_rad = np.radians(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)

        px, py = point[0] - center[0], point[1] - center[1]

        x_new = px * c - py * s
        y_new = px * s + py * c

        x_new += center[0]
        y_new += center[1]

        return (int(round(x_new)), int(round(y_new)))

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
        tweezer_points = self._get_tweezer_triangle_points(head_rect)

        body_contour = self._get_robot_body_contour(
            frame.shape, head_rect, link_rect, tweezer_points
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
    ) -> List[dict]:
        """
        TODO
        """

        # Create mask and find contours
        rock_mask = self._create_rock_mask(frame, disk_center, disk_radius)
        contours, _ = cv.findContours(
            rock_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []

        # Filter contours and get pixel-level data
        rock_data_pixels = self._filter_and_get_rocks_from_contours(
            contours, robot_body_contour
        )

        if not rock_data_pixels:
            return []

        # Process each detected rock to add polar coordinates
        rock_data = []
        px_to_mm_ratio = self.DISK_RADIUS_MM / disk_radius

        for min_rect, centroid_px, area_px in rock_data_pixels:
            # Unpack data from the minimum area rectangle
            (center, (width_px, height_px), orientation_deg) = min_rect

            # Convert pixel data to polar coordinates (using another helper)
            polar_coords = self._convert_cartesian_to_polar(
                disk_center, disk_radius, centroid_px
            )

            # Create the comprehensive dictionary for this rock
            rock_dict = {
                "pixel_centroid": centroid_px,
                "pixel_min_rect": min_rect,
                "pixel_area": area_px,
                "polar_coords": polar_coords,
                "dimensions_mm": (
                    width_px * px_to_mm_ratio,
                    height_px * px_to_mm_ratio,
                ),
                "orientation_deg": orientation_deg,
            }
            rock_data.append(rock_dict)

        return rock_data

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

    def _get_tweezer_triangle_points(self, head_rect: Tuple) -> np.ndarray:
        """
        TODO
        """
        head_center, (head_width, head_height), head_angle_deg = head_rect

        # Standardize the angle from minAreaRect to a predictable 0-360
        # range.
        if head_width < head_height:
            # If width is less than height, OpenCV's angle is along the
            # longer side. We want the angle of the shorter side
            # (the top/bottom).
            head_angle_deg += 90

        # Define tweezer geometry based on head size and constants
        tweezer_tip_height = head_height * self.ROBOT_TWEEZER_HEIGHT_FACTOR
        tweezer_base_width = head_width * self.ROBOT_TWEEZER_BASE_WIDTH_FACTOR

        # Define the triangle's points in a local, unrotated coordinate
        # system centered on the head, where +Y is down.
        # The base is on the "top" edge of the head's bounding box.
        # The tip is further "up" (negative Y).
        local_p_base_left = (-tweezer_base_width / 2, -head_height / 2)
        local_p_base_right = (tweezer_base_width / 2, -head_height / 2)
        local_p_tip = (0, -head_height / 2 - tweezer_tip_height)

        local_points = np.array(
            [local_p_base_left, local_p_base_right, local_p_tip]
        )

        # Create the 2D rotation matrix
        angle_rad = np.radians(head_angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array(((c, -s), (s, c)))

        # Rotate the local points and then translate them to the head's
        # center
        world_points = (rotation_matrix @ local_points.T).T + head_center

        return np.int32(world_points)

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
        self,
        frame_shape: Tuple,
        head_rect: Tuple,
        link_rect: Tuple,
        tweezer_points: np.ndarray,
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
        cv.fillPoly(mask, [tweezer_points], 255)

        # Find the single external contour of the combined shape
        contours, _ = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        return max(contours, key=cv.contourArea)

    # TODO fix maybe?
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
        rocks_data: List[dict],  # <-- The type hint is now List[dict]
    ) -> np.ndarray:
        """
        Visualizes detected rocks from a list of rock data dictionaries.

        Args:
            frame: The frame to draw on.
            rocks_ A list of dictionaries, where each dict is a processed rock.
        """
        vis_frame = frame.copy()

        for i, rock in enumerate(rocks_data):
            # Extract the pixel data needed for drawing from the dictionary
            min_rect = rock["pixel_min_rect"]
            centroid = rock["pixel_centroid"]

            # The rest of the drawing logic is identical
            box_points = np.int32(cv.boxPoints(min_rect))
            cv.drawContours(vis_frame, [box_points], 0, self.CYAN, 2)

            cv.circle(vis_frame, centroid, 3, self.RED, -1)

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


class Vision(VisionBase):
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

        # Thread 1 (Camera I/O) starts here
        self.camera_manager = ThreadingCameraManager(self.camera_index)

        self.calibration_manager = CalibrationManager(calibration_debug)
        self.frame_processor = FrameProcessor(
            self.frame_scale_factor,
            self.calibration_manager._camera_matrix,
            self.calibration_manager._dist_coeffs,
        )
        self.object_detector = ObjectDetector(frame_scale_factor)
        self.visualizer = Visualizer()

        # Implement threading for the interactive setup.
        self._setup_lock = threading.Lock()
        self._setup_processing_stopped = False
        self.latest_setup_frame: Optional[np.ndarray] = None
        self.latest_setup_detection: Optional[tuple] = None

        # Start the setup processing thread
        self._setup_thread = threading.Thread(
            target=self._setup_processing_loop
        )
        self._setup_thread.daemon = True
        self._setup_thread.start()

        # Start the setup display thread
        self._setup_display_thread = threading.Thread(
            target=self._setup_display_loop
        )
        self._setup_display_thread.daemon = True
        self._setup_display_thread.start()

        self._run_interactive_setup()

        # Stop both setup threads
        self._setup_processing_stopped = True
        if self._setup_thread:
            self._setup_thread.join()
        if self._setup_display_thread:
            self._setup_display_thread.join()

        self._stable_disk_center: Optional[Tuple[int, int]] = None
        self._stable_straightened_disk_center: Optional[Tuple[int, int]] = None
        self._stable_disk_radius: Optional[int] = None
        self._stable_orientation_angle: Optional[float] = None

        self._perform_post_setup_calibration()

        # Create the attributes for the processing thread.
        self._processing_lock = threading.Lock()
        self.processing_stopped = False

        # These are the shared "mailboxes" for the finished results
        self.latest_processed_frame: Optional[np.ndarray] = None
        self.latest_detection_results: dict = {}

        # Thread 2 (Processing) starts here
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def __enter__(self):
        """
        TODO
        """

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        TODO
        """

        self.processing_stopped = True
        if self.processing_thread:
            self.processing_thread.join()
        print("Processing thread stopped.")

        # Stop the camera manager
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

    def _restart_all_threads(self):
        """
        TODO
        """

        print("Restarting all background threads...")
        try:
            self.camera_manager = ThreadingCameraManager(self.camera_index)
            self.processing_stopped = False
            self.processing_thread = threading.Thread(
                target=self._processing_loop
            )
            self.processing_thread.daemon = True
            self.processing_thread.start()
            print("All threads restarted successfully.")
        except VisionConnectionError as e:
            raise VisionConnectionError(
                f"FATAL: Failed to restart threads: {e}"
            ) from e

    def _setup_processing_loop(self):
        """
        Background thread for setup detection processing.
        """
        while not self._setup_processing_stopped:
            try:
                raw_frame = self.camera_manager._capture_frame()
                scaled_frame = self.frame_processor._scale_frame(raw_frame)

                # Try to detect disk
                try:
                    center, radius = self.object_detector._detect_disk(
                        scaled_frame
                    )
                    detection_result = (center, radius)
                except (VisionDetectionError, TypeError):
                    detection_result = None

                # Update shared detection result atomically
                with self._setup_lock:
                    self.latest_setup_detection = detection_result

            except VisionConnectionError:
                time.sleep(0.1)

    def _setup_display_loop(self):
        """
        Background thread for setup frame display with color-coded feedback.
        """
        while not self._setup_processing_stopped:
            try:
                raw_frame = self.camera_manager._capture_frame()
                scaled_frame = self.frame_processor._scale_frame(raw_frame)

                # Add crosshairs
                h, w = scaled_frame.shape[:2]
                target_pos = (
                    w // 2,
                    int(h * self.SETUP_TARGET_Y_POSITION_FACTOR),
                )

                # Draw crosshairs
                cv.line(
                    scaled_frame,
                    (0, target_pos[1]),
                    (w, target_pos[1]),
                    self.YELLOW,
                    1,
                )
                cv.line(
                    scaled_frame,
                    (target_pos[0], 0),
                    (target_pos[0], h),
                    self.YELLOW,
                    1,
                )

                # Get latest detection and analyze alignment
                with self._setup_lock:
                    detection = self.latest_setup_detection

                if detection is not None:
                    center, radius = detection

                    # Calculate position error
                    pos_error = np.linalg.norm(
                        np.array(center) - np.array(target_pos)
                    )
                    is_position_ok = (
                        pos_error < self.SETUP_POSITION_TOLERANCE_PX
                    )

                    # Determine colors based on alignment
                    outline_color = self.GREEN if is_position_ok else self.RED
                    center_color = self.GREEN if is_position_ok else self.RED

                    # Draw disk with color-coded feedback
                    cv.circle(scaled_frame, center, radius, outline_color, 3)
                    cv.circle(scaled_frame, center, 7, center_color, -1)
                    cv.circle(scaled_frame, center, 7, self.BLACK, 1)

                # Update shared frame atomically
                with self._setup_lock:
                    self.latest_setup_frame = scaled_frame

            except VisionConnectionError:
                time.sleep(0.1)

    def _run_interactive_setup(self):
        """
        Simple real-time frame display for debugging.
        """
        print("\n--- Starting Simple Frame Display ---")
        print("Press 'q' to exit.")

        while not self.camera_manager.stopped:
            # Get the latest frame
            with self._setup_lock:
                if self.latest_setup_frame is not None:
                    display_frame = self.latest_setup_frame.copy()
                else:
                    continue

            cv.imshow(self.SETUP_WINDOW_NAME, display_frame)
            key = cv.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Setup aborted by user.")
                self.camera_manager._cleanup()
                cv.destroyAllWindows()
                raise SystemExit("Program exited during setup.")

        cv.destroyAllWindows()

    def _perform_post_setup_calibration(self):
        """
        TODO
        """

        print(
            "Vision: Performing startup calibration for stable world frame..."
        )

        centers_x, centers_y, radii, angles = [], [], [], []

        for i in range(self.NUM_INIT_FRAMES):
            try:
                raw_frame = self.camera_manager._capture_frame()

                if i > 5:
                    scaled_frame = self.frame_processor._scale_frame(raw_frame)

                    center, radius = self.object_detector._detect_disk(
                        scaled_frame
                    )
                    angle = (
                        self.frame_processor._detect_orientation_angle_error(
                            scaled_frame
                        )
                    )

                    centers_x.append(center[0])
                    centers_y.append(center[1])
                    radii.append(radius)
                    angles.append(angle)
            except (VisionDetectionError, VisionConnectionError) as e:
                print(
                    f"  - WARNING: Could not process frame {i + 1}: {e}. Skipping."
                )
                continue

        if len(radii) < self.NUM_INIT_FRAMES // 2:
            raise VisionCalibrationError(
                "Failed to gather enough valid frames during startup calibration."
            )

        # Calculate the average un-rotated values
        unrotated_center = (int(np.mean(centers_x)), int(np.mean(centers_y)))
        self._stable_disk_radius = int(np.mean(radii))
        self._stable_orientation_angle = np.mean(angles)

        # Find the center of the image (the pivot for rotation)
        # We can get this from the last valid scaled_frame
        h, w = scaled_frame.shape[:2]
        image_center = (w // 2, h // 2)

        # Calculate the final, rotated coordinates for the disk center
        self._stable_straightened_disk_center = (
            self.frame_processor._rotate_point(
                point=unrotated_center,
                center=image_center,
                angle_deg=-self._stable_orientation_angle,
            )
        )

        print("\nVision: Startup calibration complete.")
        print(f"  -> Stable Un-rotated Center: {unrotated_center}")
        print(
            f"  -> Stable Orientation Angle: {self._stable_orientation_angle:.2f} degrees"
        )
        print(
            f"  -> Stable STRAIGHTENED Center: {self._stable_straightened_disk_center}"
        )
        print(f"  -> Stable Radius: {self._stable_disk_radius} px\n")

    def _processing_loop(self):
        """
        TODO
        """

        while not self.processing_stopped:
            try:
                # Get the latest raw frame from the camera thread
                raw_frame = self.camera_manager._capture_frame()

                # Process the frame (scaling, etc.)
                scaled_frame = self.frame_processor._scale_frame(raw_frame)

                processed_frame = self.frame_processor._rotate_frame(
                    scaled_frame, self._stable_orientation_angle
                )

                disk_center = self._stable_straightened_disk_center
                disk_radius = self._stable_disk_radius

                robot_data = self.object_detector._detect_robot(
                    processed_frame, disk_center, disk_radius
                )

                robot_contour = robot_data[2] if robot_data else None

                rocks_data = self.object_detector._detect_rocks(
                    processed_frame, disk_center, disk_radius, robot_contour
                )

                # Visualize all detections
                vis_frame = processed_frame.copy()
                vis_frame = self.visualizer._visualize_disk(
                    vis_frame, disk_center, disk_radius
                )
                if robot_data:
                    vis_frame = self.visualizer._visualize_robot(
                        vis_frame, robot_data[0], robot_data[1], robot_data[2]
                    )
                if rocks_data:
                    vis_frame = self.visualizer._visualize_rocks(
                        vis_frame, rocks_data
                    )

                # Atomically update the shared results
                with self._processing_lock:
                    self.latest_processed_frame = vis_frame
                    self.latest_detection_results = {
                        "disk": (disk_center, disk_radius),
                        "robot": robot_data,
                        "rocks": rocks_data,
                    }

            except VisionDetectionError as e:
                # If a core object isn't found, show an error.
                with self._processing_lock:
                    try:
                        # Grab a raw frame to draw on
                        raw_frame = self.camera_manager._capture_frame()
                        error_frame = self.frame_processor._scale_frame(
                            raw_frame
                        )
                        cv.putText(
                            error_frame,
                            str(e),
                            (20, 50),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                        self.latest_processed_frame = error_frame
                        self.latest_detection_results = {}
                    except VisionConnectionError:
                        pass

            except VisionConnectionError:
                # If the camera stream stops, wait for it to come back.
                time.sleep(0.5)
                pass

    # Public Interface --------------------------------------------------------

    def set_camera_settings(self) -> None:
        """
        TODO
        """

        print("\n--- Entering Camera Settings Mode 'q' to Exit ---")
        print("Stopping main threaded stream to access hardware settings...")

        self.processing_stopped = True
        if self.processing_thread is not None:
            self.processing_thread.join()

        # Stop and clean up the current (fast) camera manager
        self.camera_manager._cleanup()

        print("Opening camera with cv.CAP_DSHOW for settings access...")

        # Create a temporary, blocking camera instance with the DSHOW backend
        settings_cam = cv.VideoCapture(self.camera_index, cv.CAP_DSHOW)

        # Restart the main manager before failing if it did not open.
        if not settings_cam.isOpened():
            print("ERROR: Could not open camera with DSHOW backend.")
            print("Restarting main stream with previous settings...")
            self._restart_camera_manager()
            return

        try:
            settings_cam.set(
                cv.CAP_PROP_FRAME_WIDTH, VisionBase.DEFAULT_CAMERA_WIDTH
            )
            settings_cam.set(
                cv.CAP_PROP_FRAME_HEIGHT, VisionBase.DEFAULT_CAMERA_HEIGHT
            )

            # Open the settings dialog.
            settings_cam.set(cv.CAP_PROP_SETTINGS, 1)

            # Run a simple, blocking preview loop to observe changes
            while True:
                ret, frame = settings_cam.read()

                if not ret:
                    print("Warning: Lost connection to settings camera.")
                    break

                scaled = self.frame_processor._scale_frame(frame)

                # Show a preview with a clear title
                cv.imshow("Camera Settings Preview (DSHOW)", scaled)

                # Wait for 'q' to be pressed to close the preview
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            # Clean up the temporary DSHOW camera and any open windows
            settings_cam.release()
            cv.destroyAllWindows()
            print("DSHOW camera released.")

        print("--- Exiting Camera Settings Mode ---")

        # Restart the main threaded camera manager with the fast backend
        self._restart_all_threads()

    def dump_calibration_data(self) -> None:
        """
        TODO
        """

        self.calibration_manager.dump_calibration_data()

    def detect_disk(self):
        """
        TODO
        """

        with self._processing_lock:
            return self.latest_detection_results.get("disk")

    def detect_robot(self):
        """
        TODO
        """

        with self._processing_lock:
            return self.latest_detection_results.get("robot")

    def detect_rocks(self):
        """
        TODO
        """

        with self._processing_lock:
            return self.latest_detection_results.get("rocks")

    def detect_workspace(self):
        pass

    def get_latest_output(self) -> Tuple[Optional[np.ndarray], dict]:
        """
        TODO
        """

        # Acquire the lock. The code inside this 'with' block is now
        # protected. No other thread can enter a
        # 'with self._processing_lock:' block until this one is done.
        with self._processing_lock:
            # Check if the processing thread has produced its first
            # frame yet.
            if self.latest_processed_frame is not None:
                # If it has, make a copy. We copy it so that once we
                # release the lock, the processing thread can't change
                # the frame we're holding.
                frame_to_return = self.latest_processed_frame.copy()
            else:
                # If no frame has been processed yet (e.g., at startup),
                # we'll return None for the frame.
                frame_to_return = None

            # Make a copy of the results dictionary for the same reason.
            # This prevents the processing thread from modifying the
            # dictionary while the main thread is trying to use it.
            results_to_return = self.latest_detection_results.copy()

        # The 'with' block has ended, so the lock is automatically
        # released. Other threads can now acquire the lock.

        # Return the copies we made.
        return frame_to_return, results_to_return


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
    with Vision(frame_scale_factor=1, calibration_debug=False) as vis:
        # vis.detect_disk(True)
        # vis.detect_robot(True)
        # vis.detect_rocks(True)
        # vis.detect_workspace(True)
        # vis.set_camera_settings()
        # vis.dump_calibration_data()
        # print(vis)

        print(
            "\nStarting lag-free test with background processing. Press 'q' to quit."
        )

        while True:
            # Get the latest pre-computed frame.
            frame, results = vis.get_latest_output()

            # Check if the processing thread has produced a frame yet
            if frame is None:
                print("Waiting for first processed frame...")
                time.sleep(0.5)
                continue

            # Display the frame.
            cv.imshow("Fully Processed Realtime Feed", frame)

            # Handle user input
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    cv.destroyAllWindows()
    print("Program finished cleanly.")
