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
    ELECTRIC_BLUE = (255, 100, 0)
    MAGENTA = (255, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    TEXT_SIZE = 0.6
    TEXT_THICKNESS = 1

    # -------------------------------------------------------------------------
    # Setup and Main Procedure Constants
    # -------------------------------------------------------------------------

    # Window name for the setup procedure
    SETUP_WINDOW_NAME = "Disk Positioning Setup"
    HEIGHT_SETUP_WINDOW_NAME = "Camera Height Setup"
    ROBOT_SETUP_WINDOW_NAME = "Robot Positioning Setup"

    TARGET_DISK_RADIUS_OPTIMAL = 454
    RADIUS_TOLERANCE = 2

    # Target position for the disk center as a fraction of frame
    # dimensions. Horizontally centered (0.5), and vertically tunable
    # (e.g., 0.6 means 60% from the top).
    SETUP_TARGET_Y_POSITION_FACTOR = 0.5

    # Tolerances for the setup to be considered "OK"
    SETUP_POSITION_TOLERANCE_PX = 5
    SETUP_ANGLE_TOLERANCE_DEG = 0.3

    MAIN_DISPLAY_WINDOW_NAME = "Vision Real-time Detection"
    DISPLAY_REFRESH_RATE_MS = 30

    # The scale factor of the radius of the disk that tracks the tip of
    # the tweezers when they go through the centre of the disk
    CENTRAL_ARC_SCALE = 2.2

    # When the robot is raised from the floor (elbow) for 3 seconds,
    # at speed c32, the centroid of the robot head follows this arc.
    ROBOT_ARC_MAX_SCALE = CENTRAL_ARC_SCALE - 1.61
    ROBOT_ARC_RADIUS_SCALE = 2.1

    ROBOT_ARC_START_ANGLE = 244
    ROBOT_ARC_END_ANGLE = 270
    END_EFFECTOR_ARC_START_ANGLE = 243.7
    END_EFFECTOR_ARC_END_ANGLE = 270
    END_EFFECTOR_ARC_THICKNESS = 2

    ROBOT_PRISM_LINK_HEIGHT_FACTOR = 2
    ROBOT_PRISM_LINK_WIDTH_FACTOR = 0.5
    ROBOT_TWEEZER_HEIGHT_FACTOR = 1.23
    ROBOT_TWEEZER_TIP_WIDTH_FACTOR = 0.7
    ROBOT_HEAD_MASK_OFFSET_PX = 10

    CALIBRATION_MARKER_ANGLES = [0, 3, 6, 9, 12, 15, 18, 20, 22, 24, 26]
    CALIBRATION_MARKER_RADIUS = 2
    CALIBRATION_MARKER_THICKNESS = -1

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

    @property
    def SCALED_DISK_RADIUS(self):
        return int(self.TARGET_DISK_RADIUS_OPTIMAL * self.frame_scale_factor)


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

    def _configure_camera_properties(self):
        """
        TODO
        """

        self._camera.set(cv.CAP_PROP_FRAME_WIDTH, self.DEFAULT_CAMERA_WIDTH)
        self._camera.set(cv.CAP_PROP_FRAME_HEIGHT, self.DEFAULT_CAMERA_HEIGHT)
        self._camera.set(cv.CAP_PROP_AUTOFOCUS, 0)
        self._camera.set(cv.CAP_PROP_FOCUS, self.DEFAULT_FOCUS_LEVEL)

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

        self.stopped = True

        # Wait for the thread to finish its work (which includes
        # releasing the camera)
        if self._thread is not None:
            self._thread.join()
            print("Vision: Camera thread terminated.")

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
        tweezer_points = self._get_tweezer_trapezium_points(head_rect)

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

        # Sort rocks by distance from disk center (closest = Rock 1)
        rock_data.sort(
            key=lambda rock: np.linalg.norm(
                np.array(rock["pixel_centroid"]) - np.array(disk_center)
            )
        )

        return rock_data

    def _detect_workspace_arcs(
        self, disk_center_px: Tuple[int, int], disk_radius_px: int
    ) -> dict:
        """
        TODO
        """

        # Calculate magenta arc (central workspace)
        end_effector_radius_px = int(disk_radius_px * self.CENTRAL_ARC_SCALE)
        end_effector_center = (
            disk_center_px[0],
            disk_center_px[1] + end_effector_radius_px,
        )

        # Calculate red arc (robot workspace)
        robot_top_y = int(
            disk_center_px[1] + disk_radius_px * self.ROBOT_ARC_MAX_SCALE
        )
        robot_radius_px = int(disk_radius_px * self.ROBOT_ARC_RADIUS_SCALE)
        robot_center = (disk_center_px[0], robot_top_y + robot_radius_px)

        # Calculate endpoint coordinates for the robot arc
        robot_end_start_x = int(
            robot_center[0]
            + robot_radius_px * np.cos(np.radians(self.ROBOT_ARC_START_ANGLE))
        )
        robot_end_start_y = int(
            robot_center[1]
            + robot_radius_px * np.sin(np.radians(self.ROBOT_ARC_START_ANGLE))
        )
        robot_end_end_x = int(
            robot_center[0]
            + robot_radius_px * np.cos(np.radians(self.ROBOT_ARC_END_ANGLE))
        )
        robot_end_end_y = int(
            robot_center[1]
            + robot_radius_px * np.sin(np.radians(self.ROBOT_ARC_END_ANGLE))
        )

        # Calculate endpoint coordinates for the end effector arc
        end_effector_end_start_x = int(
            end_effector_center[0]
            + end_effector_radius_px
            * np.cos(np.radians(self.END_EFFECTOR_ARC_START_ANGLE))
        )
        end_effector_end_start_y = int(
            end_effector_center[1]
            + end_effector_radius_px
            * np.sin(np.radians(self.END_EFFECTOR_ARC_START_ANGLE))
        )
        end_effector_end_end_x = int(
            end_effector_center[0]
            + end_effector_radius_px
            * np.cos(np.radians(self.END_EFFECTOR_ARC_END_ANGLE))
        )
        end_effector_end_end_y = int(
            end_effector_center[1]
            + end_effector_radius_px
            * np.sin(np.radians(self.END_EFFECTOR_ARC_END_ANGLE))
        )

        marker_positions = self._detect_centre_arc_markers(
            disk_center_px, disk_radius_px, self.CALIBRATION_MARKER_ANGLES
        )

        return {
            "end_effector_arc": {
                "center": end_effector_center,
                "radius": end_effector_radius_px,
                "start_angle": self.END_EFFECTOR_ARC_START_ANGLE,
                "end_angle": self.END_EFFECTOR_ARC_END_ANGLE,
            },
            "robot_arc": {
                "center": robot_center,
                "radius": robot_radius_px,
                "start_angle": self.ROBOT_ARC_START_ANGLE,
                "end_angle": self.ROBOT_ARC_END_ANGLE,
            },
            "robot_endpoint_dots": [
                (
                    robot_end_start_x,
                    robot_end_start_y,
                ),
                (
                    robot_end_end_x,
                    robot_end_end_y,
                ),
            ],
            "end_effector_endpoint_dots": [
                (
                    end_effector_end_start_x,
                    end_effector_end_start_y,
                ),
                (
                    end_effector_end_end_x,
                    end_effector_end_end_y,
                ),
            ],
            "calibration_markers": {
                "positions": marker_positions,
                "angles": self.CALIBRATION_MARKER_ANGLES,
            },
        }

    def _detect_centre_arc_markers(
        self,
        disk_center_px: Tuple[int, int],
        disk_radius_px: int,
        marker_angles: List[float],
    ) -> List[Tuple[int, int]]:
        """
        Calculate pixel positions for angle markers on the centre arc.

        Args:
            disk_center_px: Center of the disk in pixels
            disk_radius_px: Radius of the disk in pixels
            marker_angles: List of angles in degrees (0° = up, clockwise positive)

        Returns:
            List of (x, y) pixel coordinates for each marker
        """

        # Calculate centre arc (end effector) geometry
        end_effector_radius_px = int(disk_radius_px * self.CENTRAL_ARC_SCALE)
        end_effector_center = (
            disk_center_px[0],
            disk_center_px[1] + end_effector_radius_px,
        )

        marker_positions = []

        for angle_deg in marker_angles:
            # Convert angle to radians (0° = up, clockwise positive)
            angle_rad = np.radians(
                -angle_deg - 90
            )  # Subtract 90 to make 0° point up

            # Calculate marker position on the centre arc
            marker_x = int(
                end_effector_center[0]
                + end_effector_radius_px * np.cos(angle_rad)
            )
            marker_y = int(
                end_effector_center[1]
                + end_effector_radius_px * np.sin(angle_rad)
            )

            marker_positions.append((marker_x, marker_y))

        return marker_positions

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

    def _get_tweezer_trapezium_points(self, head_rect: Tuple) -> np.ndarray:
        """
        TODO
        """
        head_center, (head_width, head_height), head_angle_deg = head_rect

        # This normalization MUST be identical to the link function's logic.
        normalized_angle = head_angle_deg
        width_for_calc = head_width
        height_for_calc = head_height

        if head_width < head_height:
            normalized_angle = head_angle_deg - 90
            # When we adjust the angle, we also need to swap dimensions
            width_for_calc = head_height
            height_for_calc = head_width

        # Define geometry relative to the NORMALIZED dimensions
        tweezer_h = height_for_calc * self.ROBOT_TWEEZER_HEIGHT_FACTOR
        base_w = width_for_calc
        tip_w = width_for_calc * self.ROBOT_TWEEZER_TIP_WIDTH_FACTOR

        # Place tweezers at the "front" of the robot
        y_base = -height_for_calc / 2
        y_tip = y_base - tweezer_h

        local_points = np.array(
            [
                [-base_w / 2, y_base],
                [base_w / 2, y_base],
                [tip_w / 2, y_tip],
                [-tip_w / 2, y_tip],
            ],
            dtype=np.float32,
        )

        # Use the NORMALIZED angle for rotation, not the original
        rad = np.radians(normalized_angle)
        c, s = np.cos(rad), np.sin(rad)
        rot_matrix = np.array([[c, -s], [s, c]])

        world_points = (rot_matrix @ local_points.T).T + head_center

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

        # Apply the same normalization logic as the tweezers function
        normalized_angle = head_angle_deg
        width_for_calc = head_width
        height_for_calc = head_height

        if head_width < head_height:
            normalized_angle = head_angle_deg - 90
            # When we adjust the angle, we also need to swap dimensions
            width_for_calc = head_height
            height_for_calc = head_width

        # Calculate link position using normalized values
        link_offset = height_for_calc
        head_angle_rad = abs(np.radians(normalized_angle))

        # Determine link center based on normalized angle
        if normalized_angle <= 0:
            link_center_x = head_center[0] + link_offset * np.sin(
                head_angle_rad
            )
        else:
            link_center_x = head_center[0] - link_offset * np.sin(
                head_angle_rad
            )

        link_center_y = head_center[1] + link_offset * np.cos(head_angle_rad)
        link_center = (link_center_x, link_center_y)

        # Calculate link dimensions using normalized values
        link_width = width_for_calc * self.ROBOT_PRISM_LINK_WIDTH_FACTOR
        link_length = height_for_calc * self.ROBOT_PRISM_LINK_HEIGHT_FACTOR

        # Return with the normalized angle
        return (link_center, (link_width, link_length), normalized_angle)

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

        # Create a mask by drawing filled shapes
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)

        # Deconstruct the original head rectangle
        head_center, (head_width, head_height), head_angle = head_rect

        # Create a new, larger rectangle for the mask by adding the offset
        offset = self.ROBOT_HEAD_MASK_OFFSET_PX
        offset_head_width = head_width + (offset * 2)
        offset_head_height = head_height + (offset * 2)

        # The new rectangle for the mask has the same center and angle
        offset_head_rect = (
            head_center,
            (offset_head_width, offset_head_height),
            head_angle,
        )

        # Get the 4 corner points of this new, larger rectangle
        offset_head_points = np.int32(cv.boxPoints(offset_head_rect))

        # Draw the head as a filled rectangle
        cv.fillPoly(mask, [offset_head_points], 255)

        # Draw the prismatic link as a filled rectangle (this is the key part)
        link_points = np.int32(cv.boxPoints(link_rect))
        cv.fillPoly(mask, [link_points], 255)

        # Draw tweezers
        cv.fillPoly(mask, [tweezer_points], 255)

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
        Convert pixel coordinates to polar coordinates with origin at robot arc center.

        Returns:
            Tuple[float, float]: (radius_pixels, angle_deg) where:
                - radius_pixels: distance from robot arc center
                - angle_deg: angle in degrees (0° pointing up, positive clockwise)
        """

        # Calculate robot arc center (this should match your workspace arc calculation)
        robot_top_y = int(
            disk_center_px[1] + disk_radius_px * self.ROBOT_ARC_MAX_SCALE
        )
        robot_radius_px = int(disk_radius_px * self.ROBOT_ARC_RADIUS_SCALE)
        robot_arc_center = (disk_center_px[0], robot_top_y + robot_radius_px)

        # Calculate displacement from robot arc center
        dx_px = robot_arc_center[0] - point_px[0]
        dy_px = point_px[1] - robot_arc_center[1]

        # Calculate radius (distance from origin)
        radius_px = np.sqrt(dx_px**2 + dy_px**2)

        # Calculate angle (0° pointing up, positive clockwise)
        angle_rad = np.arctan2(
            dx_px, -dy_px
        )  # Note: -dy_px to make 0° point up
        angle_deg = np.degrees(angle_rad)

        return (radius_px, angle_deg)

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
    TODO - this needs a lot of cleanup but no time :/
    """

    def __init__(self, frame_scale_factor: float = None):
        if frame_scale_factor is None:
            self.frame_scale_factor = self.DEFAULT_FRAME_SCALE_FACTOR
        else:
            self.frame_scale_factor = frame_scale_factor

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = None,
        size: float = None,
        thickness: int = None,
    ) -> None:
        """
        TODO
        """
        if color is None:
            color = self.WHITE
        if size is None:
            size = self.TEXT_SIZE
        if thickness is None:
            thickness = self.TEXT_THICKNESS

        cv.putText(
            frame,
            text,
            position,
            cv.FONT_HERSHEY_SIMPLEX,
            size,
            color,
            thickness,
        )

    def _draw_exit_instruction(self, frame: np.ndarray) -> None:
        """
        TODO
        """
        h, w = frame.shape[:2]
        exit_text = "Press 'q' to exit"
        self._draw_text(
            frame,
            exit_text,
            (w - 230, 40),
            self.RED,
            0.8,
            2,
        )

    def _draw_enter_instruction(self, frame: np.ndarray) -> None:
        """
        TODO
        """
        h, w = frame.shape[:2]
        enter_text = "Press 'Enter' to continue"
        text_size = cv.getTextSize(
            enter_text, cv.FONT_HERSHEY_SIMPLEX, 1.2, 3
        )[0]
        text_x = (w - text_size[0]) // 2  # Centered horizontally
        self._draw_text(frame, enter_text, (text_x, h // 4), self.CYAN, 1.2, 3)

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

        self._draw_text(
            vis_frame,
            f"Disk (r={radius}px)",
            label_pos,
            self.CYAN,
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
        Visualize robot with polar coordinates display.
        """
        vis_frame = frame.copy()

        # Draw the contour of the robot body
        cv.drawContours(vis_frame, [contour], -1, self.GREEN, 2)

        # Draw the centroid
        cv.circle(vis_frame, centroid_px, 4, self.RED, -1)

        # Create the label text with the polar coordinates
        radius_px, angle_deg = polar_coords
        label_text = f"Robot: r={radius_px:.1f}px, a={angle_deg:.1f}deg"

        # Position and draw the text label
        label_pos = (centroid_px[0] + 10, centroid_px[1])
        self._draw_text(
            vis_frame,
            label_text,
            label_pos,
            self.RED,
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

            self._draw_text(
                vis_frame,
                label_text,
                label_pos,
                self.RED,
            )
        return vis_frame

    def _visualize_workspace_arc(
        self,
        frame: np.ndarray,
        workspace_arcs: dict,
    ) -> np.ndarray:
        """
        Draw workspace arcs using pre-calculated geometry data.
        """
        vis_frame = frame.copy()

        # Get the arc data with updated names
        end_effector_arc = workspace_arcs["end_effector_arc"]
        robot_arc = workspace_arcs["robot_arc"]
        robot_endpoint_dots = workspace_arcs["robot_endpoint_dots"]
        end_effector_endpoint_dots = workspace_arcs[
            "end_effector_endpoint_dots"
        ]

        # Draw the magenta arc (end effector)
        cv.ellipse(
            vis_frame,
            end_effector_arc["center"],
            (end_effector_arc["radius"], end_effector_arc["radius"]),
            0,  # rotation angle
            end_effector_arc["start_angle"],
            end_effector_arc["end_angle"],
            self.YELLOW,
            self.END_EFFECTOR_ARC_THICKNESS,
        )

        # Draw the red arc (robot)
        cv.ellipse(
            vis_frame,
            robot_arc["center"],
            (robot_arc["radius"], robot_arc["radius"]),
            0,  # rotation angle
            robot_arc["start_angle"],
            robot_arc["end_angle"],
            self.YELLOW,
            2,  # thickness
        )

        # Draw robot endpoint dots
        for dot_pos in robot_endpoint_dots:
            cv.circle(vis_frame, dot_pos, 4, self.RED, -1)

        # Draw end effector endpoint dots
        for dot_pos in end_effector_endpoint_dots:
            cv.circle(vis_frame, dot_pos, 4, self.MAGENTA, -1)

        return vis_frame

    def _visualize_orientation_cross(
        self,
        frame: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        angle_deg: float,
        is_angle_ok: bool = False,
        is_position_ok: bool = False,
    ) -> None:
        """
        TODO
        """
        # Get frame dimensions
        h, w = frame.shape[:2]

        # Choose color based on angle tolerance - RED when bad, GREEN when good
        line_color = (
            self.GREEN if (is_angle_ok and is_position_ok) else self.RED
        )

        # Draw the main orientation line
        self._draw_line_to_edges(frame, center, angle_deg, line_color, h, w)

        # Draw the perpendicular line (90 degrees offset)
        perpendicular_angle = angle_deg + 90
        self._draw_line_to_edges(
            frame, center, perpendicular_angle, line_color, h, w
        )

    def _draw_line_to_edges(
        self,
        frame: np.ndarray,
        center: Tuple[int, int],
        angle_deg: float,
        color: Tuple[int, int, int],
        h: int,
        w: int,
    ) -> None:
        """
        TODO
        """

        # Convert angle to radians
        angle_rad = np.radians(angle_deg)

        # Calculate direction vector
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)

        # Calculate potential intersections with frame boundaries
        intersections = []

        # Top edge (y = 0)
        if dy != 0:
            t = -center[1] / dy
            x = center[0] + t * dx
            if 0 <= x <= w:
                intersections.append((int(x), 0))

        # Bottom edge (y = h)
        if dy != 0:
            t = (h - center[1]) / dy
            x = center[0] + t * dx
            if 0 <= x <= w:
                intersections.append((int(x), h))

        # Left edge (x = 0)
        if dx != 0:
            t = -center[0] / dx
            y = center[1] + t * dy
            if 0 <= y <= h:
                intersections.append((0, int(y)))

        # Right edge (x = w)
        if dx != 0:
            t = (w - center[0]) / dx
            y = center[1] + t * dy
            if 0 <= y <= h:
                intersections.append((w, int(y)))

        # Remove duplicates and sort by distance from center
        unique_intersections = []
        for point in intersections:
            if point not in unique_intersections:
                unique_intersections.append(point)

        if len(unique_intersections) >= 2:
            # Sort by distance from center to get the two endpoints
            unique_intersections.sort(
                key=lambda p: (p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2
            )
            start_point = unique_intersections[0]
            end_point = unique_intersections[-1]
        else:
            # Fallback: extend beyond frame if intersection calculation fails
            line_length = max(w, h)  # Use frame diagonal as max length
            dx_int = int(line_length * dx)
            dy_int = int(line_length * dy)
            start_point = (center[0] - dx_int, center[1] - dy_int)
            end_point = (center[0] + dx_int, center[1] + dy_int)

        # Draw the orientation line
        cv.line(frame, start_point, end_point, color, 2, cv.LINE_AA)

    def _add_height_setup_overlay(
        self,
        frame: np.ndarray,
        radius: int,
        status_color: tuple,
        instruction_text: str,
        instruction_color: tuple,
        is_ok: bool,
    ) -> None:
        """
        Add informational overlay to height setup frame.
        """
        h, w = frame.shape[:2]

        # Calculate actual range
        min_radius = self.SCALED_DISK_RADIUS - self.RADIUS_TOLERANCE
        max_radius = self.SCALED_DISK_RADIUS + self.RADIUS_TOLERANCE

        # Current radius display
        radius_text = f"Current Radius: {radius} pixels"
        self._draw_text(
            frame,
            radius_text,
            (20, 40),
            self.WHITE,
            0.8,
            2,
        )

        # Target range display
        target_text = f"Target: {self.SCALED_DISK_RADIUS} ({min_radius}-{max_radius}) pixels"
        self._draw_text(
            frame,
            target_text,
            (20, 65),
            self.CYAN,
            0.6,
            1,
        )

        if is_ok:
            self._draw_enter_instruction(frame)
        else:
            text_size = cv.getTextSize(
                instruction_text, cv.FONT_HERSHEY_SIMPLEX, 1.2, 3
            )[0]
            text_x = (w - text_size[0]) // 2
            self._draw_text(
                frame,
                instruction_text,
                (text_x, h // 4),
                instruction_color,
                1.2,
                3,
            )

        # Progress bar showing how close to target
        self._draw_radius_progress_bar(frame, radius, w, h)

        self._draw_exit_instruction(frame)

    def _draw_radius_progress_bar(
        self, frame: np.ndarray, current_radius: int, w: int, h: int
    ) -> None:
        """
        Draw a progress bar showing current radius relative to target.
        """
        # Progress bar dimensions
        bar_width = w - 40
        bar_height = 20
        bar_x = 20
        bar_y = h - 60

        # Calculate range for progress bar (extend beyond target for better visualization)
        range_min = self.SCALED_DISK_RADIUS - (self.RADIUS_TOLERANCE * 10)
        range_max = self.SCALED_DISK_RADIUS + (self.RADIUS_TOLERANCE * 10)

        # Draw background bar
        cv.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            self.WHITE,
            1,
        )

        # Draw target zone (green area)
        target_min = self.SCALED_DISK_RADIUS - self.RADIUS_TOLERANCE
        target_max = self.SCALED_DISK_RADIUS + self.RADIUS_TOLERANCE

        target_start_ratio = (target_min - range_min) / (range_max - range_min)
        target_end_ratio = (target_max - range_min) / (range_max - range_min)

        target_start_x = int(bar_x + target_start_ratio * bar_width)
        target_end_x = int(bar_x + target_end_ratio * bar_width)

        cv.rectangle(
            frame,
            (target_start_x, bar_y),
            (target_end_x, bar_y + bar_height),
            self.GREEN,
            -1,
        )

        # Draw optimal line (center of target zone)
        optimal_ratio = (self.SCALED_DISK_RADIUS - range_min) / (
            range_max - range_min
        )
        optimal_x = int(bar_x + optimal_ratio * bar_width)
        cv.line(
            frame,
            (optimal_x, bar_y - 5),
            (optimal_x, bar_y + bar_height + 5),
            self.CYAN,
            2,
        )

        # Draw current position indicator
        if range_min <= current_radius <= range_max:
            current_ratio = (current_radius - range_min) / (
                range_max - range_min
            )
            current_x = int(bar_x + current_ratio * bar_width)

            # Choose color based on whether it's in target range
            radius_error = abs(current_radius - self.SCALED_DISK_RADIUS)
            indicator_color = (
                self.GREEN
                if radius_error <= self.RADIUS_TOLERANCE
                else self.RED
            )

            cv.line(
                frame,
                (current_x, bar_y - 5),
                (current_x, bar_y + bar_height + 5),
                indicator_color,
                3,
            )

        # Add labels
        self._draw_text(
            frame,
            f"{range_min}",
            (bar_x, bar_y + bar_height + 15),
            self.WHITE,
            0.4,
            1,
        )
        self._draw_text(
            frame,
            f"{range_max}",
            (bar_x + bar_width - 20, bar_y + bar_height + 15),
            self.WHITE,
            0.4,
            1,
        )
        self._draw_text(
            frame,
            f"Target: {self.SCALED_DISK_RADIUS}",
            (optimal_x - 30, bar_y - 10),
            self.CYAN,
            0.4,
            1,
        )

    def _add_status_overlay(self, frame: np.ndarray, results: dict) -> None:
        """
        Add status information overlay to the display frame.
        """
        h, w = frame.shape[:2]

        # Add frame rate or detection count
        robot_status = "found" if results.get("robot") else "missing"
        rock_count = len(results.get("rocks", []))

        status_text = f"Robot: {robot_status} | Rocks: {rock_count}"

        # Add semi-transparent background for text
        overlay = frame.copy()
        cv.rectangle(overlay, (10, 10), (300, 50), self.BLACK, -1)
        cv.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        self._draw_text(frame, status_text, (20, 35), self.WHITE, 0.7, 2)
        self._draw_exit_instruction(frame)

    def visualize_height_setup_frame(
        self,
        frame: np.ndarray,
        radius: int,
        center: Tuple[int, int],
        status_color: tuple,
        instruction_text: str,
        instruction_color: tuple,
        is_ok: bool,
    ) -> np.ndarray:
        """
        Create a complete height setup visualization frame.
        """
        display_frame = frame.copy()

        # Choose circle color based on status
        circle_color = self.GREEN if is_ok else self.RED

        # Draw disk visualization
        cv.circle(display_frame, center, radius, circle_color, 3)
        cv.circle(display_frame, center, 8, self.RED, -1)

        # Add status information overlay
        self._add_height_setup_overlay(
            display_frame,
            radius,
            status_color,
            instruction_text,
            instruction_color,
            is_ok,
        )

        return display_frame

    def visualize_height_setup_no_disk(self, frame: np.ndarray) -> np.ndarray:
        """
        TODO
        """

        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        self._draw_text(
            display_frame,
            "NO DISK DETECTED",
            (w // 2 - 150, h // 2),
            self.RED,
            1.5,
            3,
        )
        self._draw_text(
            display_frame,
            "Ensure disk is visible in camera view",
            (w // 2 - 250, h // 2 + 50),
            self.WHITE,
            0.8,
            2,
        )

        return display_frame

    def visualize_position_setup_frame(
        self,
        frame: np.ndarray,
        detection: Optional[tuple],
        target_pos: Tuple[int, int],
    ) -> np.ndarray:
        """
        Create a complete position setup visualization frame.
        """
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # Draw crosshairs
        cv.line(
            display_frame,
            (0, target_pos[1]),
            (w, target_pos[1]),
            self.YELLOW,
            1,
        )
        cv.line(
            display_frame,
            (target_pos[0], 0),
            (target_pos[0], h),
            self.YELLOW,
            1,
        )

        # Default instruction text
        instruction_text = "Align the red and yellow crosses"
        instruction_color = self.MAGENTA

        if detection is not None:
            center, radius, orientation_angle = detection

            # Calculate position error
            pos_error = np.linalg.norm(np.array(center) - np.array(target_pos))
            is_position_ok = pos_error < self.SETUP_POSITION_TOLERANCE_PX
            is_angle_ok = (
                abs(orientation_angle) < self.SETUP_ANGLE_TOLERANCE_DEG
            )

            # Draw a small circle around the cross
            cv.circle(display_frame, center, 20, self.BLACK, 1)

            # Draw orientation line through disk center
            self._visualize_orientation_cross(
                display_frame,
                center,
                radius,
                orientation_angle,
                is_angle_ok,
                is_position_ok,
            )

            # Show the disk perimeter if position and angle are both correct
            if is_angle_ok and is_position_ok:
                cv.circle(display_frame, center, radius, self.GREEN, 3)

            status_text = f"Orientation: {-orientation_angle:.1f} deg"
            self._draw_text(
                display_frame,
                status_text,
                (20, 40),
                self.WHITE,
                0.7,
                2,
            )

            # Update instruction text based on alignment
            if is_position_ok and is_angle_ok:
                self._draw_enter_instruction(display_frame)
            else:
                # Add centered instruction text for non-ready state
                text_size = cv.getTextSize(
                    instruction_text, cv.FONT_HERSHEY_SIMPLEX, 2, 3
                )[0]
                text_x = (w - text_size[0]) // 2
                text_y = h // 4
                self._draw_text(
                    display_frame,
                    instruction_text,
                    (text_x, text_y),
                    instruction_color,
                    2,
                    3,
                )

        self._draw_exit_instruction(display_frame)

        return display_frame

    def visualize_error_frame(
        self, frame: np.ndarray, error_message: str
    ) -> np.ndarray:
        """
        Create an error visualization frame with the error message displayed.
        """
        error_frame = frame.copy()
        cv.putText(
            error_frame,
            str(error_message),
            (20, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.RED,
            2,
        )
        return error_frame

    def visualize_complete_scene(
        self,
        frame: np.ndarray,
        disk_center: Tuple[int, int],
        disk_radius: int,
        workspace_arcs: dict,
        robot_data: Optional[tuple] = None,
        rocks_data: Optional[List[dict]] = None,
    ) -> np.ndarray:
        """
        Create a complete scene visualization with all detected objects.
        """
        vis_frame = frame.copy()

        # Always visualize the disk
        vis_frame = self._visualize_disk(vis_frame, disk_center, disk_radius)

        # Visualize rocks if detected
        if rocks_data:
            vis_frame = self._visualize_rocks(vis_frame, rocks_data)

        # Visualize workspace arc
        vis_frame = self._visualize_workspace_arc(vis_frame, workspace_arcs)

        if "calibration_markers" in workspace_arcs:
            markers_data = workspace_arcs["calibration_markers"]
            vis_frame = self._visualize_centre_arc_markers(
                vis_frame, markers_data["positions"], markers_data["angles"]
            )

        # Visualize robot if detected
        if robot_data:
            vis_frame = self._visualize_robot(
                vis_frame, robot_data[0], robot_data[1], robot_data[2]
            )

        return vis_frame

    def visualize_main_display_frame(
        self, frame: np.ndarray, results: dict
    ) -> np.ndarray:
        """
        Create the main display frame with status overlay.
        """
        display_frame = frame.copy()
        self._add_status_overlay(display_frame, results)
        return display_frame

    def visualize_robot_setup_frame(
        self,
        frame: np.ndarray,
        disk_center: Tuple[int, int],
        disk_radius: int,
    ) -> np.ndarray:
        """
        Create a simple robot positioning setup visualization frame.
        """
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # Draw the disk
        cv.circle(display_frame, disk_center, disk_radius, self.CYAN, 2)

        # Draw outer ring for clarity (larger circle around the target)
        cv.circle(display_frame, disk_center, 20, self.BLACK, 1)

        # Draw the center dot (very small and precise)
        cv.circle(display_frame, disk_center, 2, self.RED, -1)

        # Simple instruction text
        instruction_text = "Press 'Enter' when robot tip is precisely over the dot and robot is touching the disk"
        text_size = cv.getTextSize(
            instruction_text, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )[0]
        text_x = (w - text_size[0]) // 2

        self._draw_text(
            display_frame,
            instruction_text,
            (text_x, h // 2 - 100),
            self.WHITE,
            0.8,
            2,
        )

        self._draw_exit_instruction(display_frame)

        return display_frame

    def _visualize_centre_arc_markers(
        self,
        frame: np.ndarray,
        marker_positions: List[Tuple[int, int]],
        marker_angles: List[float],
    ) -> np.ndarray:
        """
        Draw angle markers on the centre arc with labels.
        """
        vis_frame = frame.copy()

        for i, (pos, angle) in enumerate(zip(marker_positions, marker_angles)):
            # Draw the marker circle
            cv.circle(
                vis_frame,
                pos,
                self.CALIBRATION_MARKER_RADIUS,
                self.RED,
                self.CALIBRATION_MARKER_THICKNESS,
            )

            # Add angle label next to marker
            label_text = f"{angle}"
            label_pos = (pos[0] - 5, pos[1] - 15)

            self._draw_text(
                vis_frame,
                label_text,
                label_pos,
                self.RED,
                0.5,
                1,
            )

        return vis_frame


class Vision(VisionBase):
    """
    TODO
    """

    def __init__(
        self,
        enable_visualization: bool = True,
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

        self.enable_visualization = enable_visualization

        # Thread 1 (Camera I/O) starts here
        self.camera_manager = ThreadingCameraManager(self.camera_index)

        self.calibration_manager = CalibrationManager(calibration_debug)
        self.frame_processor = FrameProcessor(
            self.frame_scale_factor,
            self.calibration_manager._camera_matrix,
            self.calibration_manager._dist_coeffs,
        )
        self.object_detector = ObjectDetector(frame_scale_factor)
        self.visualizer = Visualizer(self.frame_scale_factor)

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

        self._run_setup()

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

        # Add robot positioning setup here
        self._run_robot_setup()

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

        # Thread 3 (Display) if enable_visualization is true.
        self._display_stopped = False
        if self.enable_visualization:
            self._display_thread = threading.Thread(target=self._display_loop)
            self._display_thread.daemon = True
            self._display_thread.start()
        else:
            self._display_thread = None

        print("\nVision: Started multithreading.")

    def __enter__(self):
        """
        TODO
        """

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        TODO
        """

        # Stop processing thread.
        self.processing_stopped = True
        if self.processing_thread:
            self.processing_thread.join()
            print("Vision: Processing thread terminated.")

        # Stop display thread
        if self._display_thread:
            self._display_stopped = True
            self._display_thread.join()
            cv.destroyAllWindows()
            print("Vision: Display thread terminated.")

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

            # Restart processing thread
            self.processing_stopped = False
            self.processing_thread = threading.Thread(
                target=self._processing_loop
            )
            self.processing_thread.daemon = True
            self.processing_thread.start()

            # Restart display thread if visualization enabled
            if self.enable_visualization:
                self._display_stopped = False
                self._display_thread = threading.Thread(
                    target=self._display_loop
                )
                self._display_thread.daemon = True
                self._display_thread.start()

            print("All threads restarted successfully.")
        except VisionConnectionError as e:
            raise VisionConnectionError(
                f"FATAL: Failed to restart threads: {e}"
            ) from e

    # Position setup functions-------------------------------------------------

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
                    # NEW: Also detect orientation angle
                    orientation_angle = (
                        self.frame_processor._detect_orientation_angle_error(
                            scaled_frame
                        )
                    )
                    detection_result = (center, radius, orientation_angle)
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

                # Calculate target position
                h, w = scaled_frame.shape[:2]
                target_pos = (
                    w // 2,
                    int(h * self.SETUP_TARGET_Y_POSITION_FACTOR),
                )

                # Get latest detection
                with self._setup_lock:
                    detection = self.latest_setup_detection

                # Use visualizer to create the display frame
                display_frame = self.visualizer.visualize_position_setup_frame(
                    scaled_frame, detection, target_pos
                )

                # Update shared frame atomically
                with self._setup_lock:
                    self.latest_setup_frame = display_frame

            except VisionConnectionError:
                time.sleep(0.1)

    def _run_height_setup(self):
        """
        Interactive camera height setup to achieve target disk radius.
        """

        while not self.camera_manager.stopped:
            try:
                # Get frame and detect disk
                raw_frame = self.camera_manager._capture_frame()
                scaled_frame = self.frame_processor._scale_frame(raw_frame)

                # Create display frame
                display_frame = scaled_frame.copy()

                try:
                    center, radius = self.object_detector._detect_disk(
                        scaled_frame
                    )

                    # Calculate how far from optimal
                    radius_error = abs(radius - self.SCALED_DISK_RADIUS)
                    is_radius_ok = radius_error <= self.RADIUS_TOLERANCE
                    is_radius_close = radius_error <= (
                        self.RADIUS_TOLERANCE * 2
                    )

                    # Choose colors and messages based on radius status
                    if is_radius_ok:
                        status_color = self.GREEN
                        instruction_text = ""  # Empty - will use _draw_enter_instruction instead
                        instruction_color = self.CYAN
                    elif is_radius_close:
                        status_color = self.YELLOW
                        instruction_text = "Adjust height slightly"
                        instruction_color = self.YELLOW
                    else:
                        status_color = self.RED
                        instruction_color = self.RED
                        if radius < self.SCALED_DISK_RADIUS:
                            instruction_text = (
                                "Move camera DOWN (closer to disk)"
                            )
                        else:
                            instruction_text = (
                                "Move camera UP (further from disk)"
                            )

                    # Use visualizer to create the display frame
                    display_frame = (
                        self.visualizer.visualize_height_setup_frame(
                            scaled_frame,
                            radius,
                            center,
                            status_color,
                            instruction_text,
                            instruction_color,
                            is_radius_ok,
                        )
                    )

                except VisionDetectionError:
                    display_frame = (
                        self.visualizer.visualize_height_setup_no_disk(
                            scaled_frame
                        )
                    )
                    is_radius_ok = False

                # Display the frame
                cv.imshow(self.HEIGHT_SETUP_WINDOW_NAME, display_frame)

                # Handle user input
                key = cv.waitKey(1) & 0xFF
                if key == ord("q"):
                    cv.destroyAllWindows()
                    raise VisionSetupCancelledError(
                        "User requested program exit during height setup"
                    )
                elif key == 13:  # Enter key
                    if is_radius_ok:
                        cv.destroyAllWindows()
                        print("Height setup complete!")
                        break
                    else:
                        print(
                            f"Radius not within acceptable range (±{self.RADIUS_TOLERANCE} pixels). Please adjust height."
                        )

            except VisionConnectionError:
                time.sleep(0.1)

        cv.destroyAllWindows()

    def _run_position_setup(self):
        """
        TODO
        """

        while not self.camera_manager.stopped:
            # Get the latest frame
            with self._setup_lock:
                if self.latest_setup_frame is not None:
                    display_frame = self.latest_setup_frame.copy()
                    detection = self.latest_setup_detection
                else:
                    continue

            cv.imshow(self.SETUP_WINDOW_NAME, display_frame)
            key = cv.waitKey(1) & 0xFF

            if key == ord("q"):
                cv.destroyAllWindows()
                raise VisionSetupCancelledError(
                    "User requested program exit during positioning setup"
                )
            elif key == 13:  # Enter key
                # Check if alignment is good before allowing continuation
                if detection is not None:
                    center, radius, orientation_angle = detection
                    h, w = display_frame.shape[:2]
                    target_pos = (
                        w // 2,
                        int(h * self.SETUP_TARGET_Y_POSITION_FACTOR),
                    )

                    pos_error = np.linalg.norm(
                        np.array(center) - np.array(target_pos)
                    )
                    is_position_ok = (
                        pos_error < self.SETUP_POSITION_TOLERANCE_PX
                    )
                    is_angle_ok = (
                        abs(orientation_angle) < self.SETUP_ANGLE_TOLERANCE_DEG
                    )

                    if is_position_ok and is_angle_ok:
                        cv.destroyAllWindows()
                        print("Positioning setup complete!")
                        break  # Exit the setup loop to continue
                    else:
                        print(
                            "Position/orientation not aligned. Please adjust."
                        )

        cv.destroyAllWindows()

    def _run_robot_setup(self):
        """
        Simple robot positioning setup - just wait for user confirmation.
        """
        print(
            "\nVision: Position robot tip over center dot and touching disk..."
        )

        while not self.camera_manager.stopped:
            try:
                # Get frame and process it
                raw_frame = self.camera_manager._capture_frame()
                scaled_frame = self.frame_processor._scale_frame(raw_frame)
                processed_frame = self.frame_processor._rotate_frame(
                    scaled_frame, self._stable_orientation_angle
                )

                # Use the stable disk parameters
                disk_center = self._stable_straightened_disk_center
                disk_radius = self._stable_disk_radius

                # Create simple display frame
                display_frame = self.visualizer.visualize_robot_setup_frame(
                    processed_frame, disk_center, disk_radius
                )

                # Display the frame
                cv.imshow(self.ROBOT_SETUP_WINDOW_NAME, display_frame)

                # Handle user input
                key = cv.waitKey(1) & 0xFF
                if key == ord("q"):
                    cv.destroyAllWindows()
                    raise VisionSetupCancelledError(
                        "User requested program exit during robot setup"
                    )
                elif key == 13:  # Enter key
                    cv.destroyAllWindows()
                    print("Robot positioning setup complete!")
                    break

            except VisionConnectionError:
                time.sleep(0.1)

        cv.destroyAllWindows()

    def _run_setup(self):
        """
        TODO
        """

        self._run_height_setup()
        self._run_position_setup()

    # Main loop functions------------------------------------------------------

    def _perform_post_setup_calibration(self):
        """
        TODO
        """

        print(
            "\nVision: Performing startup calibration for stable world frame. "
            "Please wait..."
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

        # Calculate static workspace arcs geometry once
        self._stable_workspace_arcs = (
            self.object_detector._detect_workspace_arcs(
                self._stable_straightened_disk_center, self._stable_disk_radius
            )
        )

        print("\nVision: Startup calibration complete.")

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
                workspace_arcs = self._stable_workspace_arcs

                robot_data = self.object_detector._detect_robot(
                    processed_frame, disk_center, disk_radius
                )

                robot_contour = robot_data[2] if robot_data else None

                rocks_data = self.object_detector._detect_rocks(
                    processed_frame, disk_center, disk_radius, robot_contour
                )

                # Conditional visualization based on parameter
                if self.enable_visualization:
                    output_frame = self.visualizer.visualize_complete_scene(
                        processed_frame,
                        disk_center,
                        disk_radius,
                        workspace_arcs,
                        robot_data,
                        rocks_data,
                    )
                else:
                    # Return the clean processed frame without overlays
                    output_frame = processed_frame

                # Atomically update the shared results
                with self._processing_lock:
                    self.latest_processed_frame = output_frame
                    self.latest_detection_results = {
                        "disk": (disk_center, disk_radius),
                        "robot": robot_data,
                        "rocks": rocks_data,
                        "workspace_arcs": workspace_arcs,
                    }

            except VisionDetectionError as e:
                # If a core object isn't found, show an error.
                with self._processing_lock:
                    try:
                        # Grab a raw frame to draw on
                        raw_frame = self.camera_manager._capture_frame()
                        scaled_frame = self.frame_processor._scale_frame(
                            raw_frame
                        )
                        self.latest_processed_frame = (
                            self.visualizer.visualize_error_frame(
                                scaled_frame, str(e)
                            )
                        )
                        self.latest_detection_results = {}
                    except VisionConnectionError:
                        time.sleep(0.01)
                        pass

            except VisionConnectionError:
                # If the camera stream stops, wait for it to come back.
                time.sleep(0.5)
                pass

    def _display_loop(self):
        """
        TODO
        """
        print("\nVision: Starting live feed.")

        while not self._display_stopped and not self.processing_stopped:
            try:
                # Get the latest frame (with or without overlays)
                with self._processing_lock:
                    if self.latest_processed_frame is not None:
                        display_frame = self.latest_processed_frame.copy()
                        results = self.latest_detection_results.copy()
                    else:
                        time.sleep(0.01)
                        continue

                # Create the display frame with status overlay
                display_frame = self.visualizer.visualize_main_display_frame(
                    display_frame, results
                )

                # Display the frame
                cv.imshow(self.MAIN_DISPLAY_WINDOW_NAME, display_frame)

                # Check for user input (optional - for closing window)
                key = cv.waitKey(self.DISPLAY_REFRESH_RATE_MS) & 0xFF
                if key == ord("q"):
                    print(
                        "Display window closed by user. "
                        "Program is still running!"
                    )
                    break

            except Exception as e:
                print(f"Display thread error: {e}")
                time.sleep(0.1)

        cv.destroyAllWindows()

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

    def detect_workspace_arcs(self):
        """
        Get the static workspace arc geometry data.
        """
        return getattr(self, "_stable_workspace_arcs", None)

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


class VisionSetupCancelledError(VisionError):
    """
    Exception raised when user cancels the setup process.

    This is raised when the user presses 'q' during the interactive setup,
    indicating they want to exit the program.
    """

    @tested
    def __init__(self, message: str = "Setup cancelled by user") -> None:
        super().__init__(f"Vision: {message}")


import keyboard

if __name__ == "__main__":
    with Vision(
        enable_visualization=True,
        frame_scale_factor=0.8,
        calibration_debug=False,
    ) as vis:
        while True:
            # Get the latest pre-computed frame.
            frame, results = vis.get_latest_output()

            if keyboard.is_pressed("/"):
                print("'/' key detected, exiting...\n")
                break

            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)

        print("Program ended.")
