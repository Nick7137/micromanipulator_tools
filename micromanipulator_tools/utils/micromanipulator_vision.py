# TODO turn into class later.
# TODO IMPORTANT must set the camera 90 mm above the disk
# only want to rotate one direction to stop backlash?
# depthanything v2
# TODO do formatting and get rid of inline comments.
# TODO get rid of magic numbers or magic strings
# TODO make the docstrings good too.
# TODO no inline if statements
# TODO make all error messages show the function they originate from
# TODO add in the class constants the default address for calibration images zoom_20%

# implement big brain algorithm that decides which rock to go for.

# ==============================================================================

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

    This is the parent class for all MicromanipulatorVision-specific exceptions.
    Catch this to handle any MicromanipulatorVision error generically.

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

    # -------------------------------------------------------------------------
    # Initialisation methods---------------------------------------------------
    # -------------------------------------------------------------------------

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
        self.r_vecs: Optional[List[np.ndarray]] = None
        self.t_vecs: Optional[List[np.ndarray]] = None
        self.reprojection_error: Optional[float] = None
        self.is_calibrated = False

        # TODO
        # if there is no calibration data, calibrate the camera, if cant do that raise error

    def __enter__(self) -> "MicromanipulatorVision":
        """
        Enter the runtime context for camera resource management.

        Returns:
            self: The MicromanipulatorVision instance
        """

        return self

    def __exit__(self):
        """
        TODO
        """
        pass

    def __str__(self) -> str:
        """
        Return a string representation of the MicromanipulatorVision object.

        Returns:
            str: Human-readable description of the object state
        """

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

    # -------------------------------------------------------------------------
    # Private Methods----------------------------------------------------------
    # -------------------------------------------------------------------------

    def _calibration_load_images(
        self, subfolder: str = "zoom_20%"
    ) -> List[np.ndarray]:
        """
        Load calibration images from the expected directory structure.

        Args:
            subfolder: Subfolder within calibration_images to use

        Returns:
            List of loaded images (BGR format)

        Raises:
            MicromanipulatorVisionCalibrationError: If no images found
        """
        # Find calibration images (reuse your existing logic)
        current_dir = os.path.dirname(__file__)
        root_dir = os.path.dirname(os.path.dirname(current_dir))
        calibration_dir = os.path.join(
            root_dir, "resources", "calibration_images", subfolder
        )

        img_path_list = glob.glob(os.path.join(calibration_dir, "*.jpg"))

        print(f"Looking for images in: {calibration_dir}")
        print(f"Found {len(img_path_list)} images")

        if len(img_path_list) == 0:
            raise MicromanipulatorVisionCalibrationError(
                f"No calibration images found in {calibration_dir}. "
                f"Please add .jpg calibration images to this directory."
            )

        # Load all images
        images = []
        for img_path in img_path_list:
            img = cv.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: Could not load {img_path}")

        if len(images) == 0:
            raise MicromanipulatorVisionCalibrationError(
                "No valid calibration images could be loaded"
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
            30,
            0.001,
        )

        # Create object points (3D coordinates of chessboard corners)
        num_rows, num_cols = self.checkerboard
        world_points_template = np.zeros((num_rows * num_cols, 3), np.float32)
        world_points_template[:, :2] = np.mgrid[
            0:num_rows, 0:num_cols
        ].T.reshape(-1, 2)

        object_points_list = []
        image_points_list = []

        for i, img_bgr in enumerate(images):
            print(f"Processing image {i + 1}/{len(images)}")

            if img_bgr is None:
                print(f"  ✗ Warning: Image {i + 1} is None")
                continue

            img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

            # Find chessboard corners
            corners_found, corners_original = cv.findChessboardCorners(
                img_gray, self.checkerboard, None
            )

            if corners_found:
                print(f"  ✓ Found chessboard corners in image {i + 1}")

                # Refine corner positions to subpixel accuracy
                corners_refined = cv.cornerSubPix(
                    img_gray,
                    corners_original,
                    (11, 11),
                    (-1, -1),
                    term_criteria,
                )

                object_points_list.append(world_points_template.copy())
                image_points_list.append(corners_refined)

            else:
                print(f"  ✗ No chessboard corners found in image {i + 1}")

        if len(object_points_list) == 0:
            raise MicromanipulatorVisionCalibrationError(
                "MicromanipulatorVision._calibration_find_chessboard_corners: No chessboard"
                " patterns were detected in any images."
            )

        print(
            f"Successfully processed {len(object_points_list)} out of {len(images)} images"
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
                "MicromanipulatorVision._calibration_solve_constants: Cannot calibrate: "
                "no object points or image points provided."
            )

        if len(object_points) != len(image_points):
            raise MicromanipulatorVisionCalibrationError(
                "MicromanipulatorVision._calibration_solve_constants: Mismatch: "
                f"{len(object_points)} object point sets vs "
                f"{len(image_points)} image point sets."
            )

        # Get image size from the first image point set. Assume all
        # images are the same size
        image_size = (self.target_width, self.target_height)

        print(f"Running calibration with {len(object_points)} image sets...")
        print(f"Image size: {image_size}")

        # Perform OpenCV camera calibration
        reprojection_error, camera_matrix, dist_coeffs, r_vecs, t_vecs = (
            cv.calibrateCamera(
                object_points,
                image_points,
                image_size,
                None,  # Initial camera matrix
                None,  # Initial distortion coefficients
            )
        )

        # Store calibration results
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.r_vecs = r_vecs  # Rotation vectors for each calibration image
        self.t_vecs = t_vecs  # Translation vectors for each calibration image
        self.reprojection_error = reprojection_error
        self.is_calibrated = True

        # Display results
        print("=" * 50)
        print("CALIBRATION COMPLETE")
        print("=" * 50)
        print("Camera Matrix:")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs)
        print(f"\nReprojection Error: {reprojection_error:.4f} pixels")

        # Evaluate calibration quality
        if reprojection_error < 0.5:
            print("✓ Excellent calibration quality!")
        elif reprojection_error < 1.0:
            print("✓ Very good calibration quality")
        elif reprojection_error < 2.0:
            print("⚠ OK calibration quality (acceptable)")
        else:
            print("⚠ Poor calibration quality - consider retaking images")
        print("=" * 50)

    def _calibration_save_constants(
        self, filename: str = "camera_calibration.npz"
    ) -> None:
        """
        Save the calibration results to a file.

        Args:
            filename: Name of the file to save calibration data to

        Raises:
            MicromanipulatorVisionCalibrationError: If no calibration
                data to save
        """

        if not self.is_calibrated:
            raise MicromanipulatorVisionCalibrationError(
                "No calibration data to save. Run calibration first."
            )

        if self.camera_matrix is None or self.dist_coeffs is None:
            raise MicromanipulatorVisionCalibrationError(
                "Calibration data is incomplete. Camera matrix or distortion coefficients missing."
            )

        # Create save path (same directory as the script)
        current_dir = os.path.dirname(__file__)
        save_path = os.path.join(current_dir, filename)

        # Save all calibration data
        np.savez(
            save_path,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            r_vecs=self.r_vecs,
            t_vecs=self.t_vecs,
            reprojection_error=self.reprojection_error,
            checkerboard_size=np.array(self.checkerboard),
            target_width=self.target_width,
            target_height=self.target_height,
        )

        print(f"✓ Calibration saved to: {save_path}")
        print(f"  - Camera matrix: {self.camera_matrix.shape}")
        print(f"  - Distortion coefficients: {self.dist_coeffs.shape}")
        print(f"  - Reprojection error: {self.reprojection_error:.4f} pixels")
        print(
            f"  - Number of calibration poses: {len(self.r_vecs) if self.r_vecs else 0}"
        )

    def _calibration_load_constants(
        self, filename: str = "camera_calibration.npz"
    ) -> bool:
        """
        Load existing calibration data from file.

        Args:
            filename: Name of the calibration file to load

        Returns:
            bool: True if calibration loaded successfully, False if file
                not found

        Raises:
            MicromanipulatorVisionCalibrationError: If file exists but
                data is invalid
        """

        current_dir = os.path.dirname(__file__)
        load_path = os.path.join(current_dir, filename)

        if not os.path.exists(load_path):
            print(f"No calibration file found at: {load_path}")
            return False

        try:
            # Load calibration data
            data = np.load(load_path)

            # Load core calibration data
            self.camera_matrix = data["camera_matrix"]
            self.dist_coeffs = data["dist_coeffs"]

            # Load optional fields with fallbacks
            self.r_vecs = data["r_vecs"].tolist() if "r_vecs" in data else None
            self.t_vecs = data["t_vecs"].tolist() if "t_vecs" in data else None
            self.reprojection_error = (
                float(data["reprojection_error"])
                if "reprojection_error" in data
                else None
            )

            self.is_calibrated = True

            print(f"✓ Calibration loaded from: {load_path}")
            print(
                f"  - Reprojection error: {self.reprojection_error:.4f} pixels"
                if self.reprojection_error
                else ""
            )

            return True

        except Exception as e:
            raise MicromanipulatorVisionCalibrationError(
                f"Failed to load calibration from {load_path}: {str(e)}"
            )

    # -------------------------------------------------------------------------
    # Public interface-----------------------------------------------------------
    # -------------------------------------------------------------------------

    def calibrate_camera(self) -> None:
        """
        Ensure the camera is calibrated by loading existing data or running calibration
        on pre-existing calibration images.

        Raises:
            MicromanipulatorVisionCalibrationError: If calibration cannot be completed
        """

        print("Initializing camera calibration...")

        # Try to load existing calibration first
        if self._calibration_load_constants():
            print("✓ Using existing calibration data")
            return

        print(
            "No existing calibration found - running calibration on existing images..."
        )

        try:
            # Load calibration images from disk
            images = self._calibration_load_images()

            # Find chessboard corners
            object_points, image_points = (
                self._calibration_find_chessboard_corners(images)
            )

            # Perform calibration
            self._calibration_solve_constants(object_points, image_points)

            # Save for future use
            self._calibration_save_constants()

            print("✓ Camera calibration completed successfully")

        except Exception as e:
            raise MicromanipulatorVisionCalibrationError(
                f"Failed to calibrate camera: {str(e)}"
            )

    def undistort_image(
        self, image: np.ndarray, alpha: float = 1.0
    ) -> np.ndarray:
        """
        Apply undistortion to an image using the calibrated camera parameters.

        Args:
            image: Input image to undistort (BGR format)
            alpha: Free scaling parameter (0.0 = crop to valid pixels, 1.0 = keep all pixels)

        Returns:
            np.ndarray: Undistorted image

        Raises:
            MicromanipulatorVisionCalibrationError: If camera is not calibrated
            ValueError: If alpha is not between 0.0 and 1.0
        """
        if not self.is_calibrated:
            raise MicromanipulatorVisionCalibrationError(
                "Camera not calibrated. Run calibration or load calibration data first."
            )

        if self.camera_matrix is None or self.dist_coeffs is None:
            raise MicromanipulatorVisionCalibrationError(
                "Camera matrix or distortion coefficients are missing."
            )

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha must be between 0.0 and 1.0, got {alpha}")

        # Get image dimensions
        height, width = image.shape[:2]

        # Get optimal new camera matrix
        camera_matrix_new, roi = cv.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            (width, height),
            alpha,
            (width, height),
        )

        # Apply undistortion
        undistorted_image = cv.undistort(
            image,
            self.camera_matrix,
            self.dist_coeffs,
            None,
            camera_matrix_new,
        )

        return undistorted_image
