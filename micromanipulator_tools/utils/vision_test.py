import numpy as np
import cv2 as cv
import glob
import os


def calibrate(show_pics=True):
    # get the directory of the current file
    current_dir = os.path.dirname(__file__)

    # Go up two levels to reach the root
    root_dir = os.path.dirname(os.path.dirname(current_dir))

    # Get the directory of the calibration images
    calibration_dir = os.path.join(
        root_dir, "resources", "calibration_images", "zoom_20%"
    )

    img_path_list = glob.glob(os.path.join(calibration_dir, "*.jpg"))

    # Debug: Check if images are found
    print(f"Looking for images in: {calibration_dir}")
    print(f"Found {len(img_path_list)} images")

    if len(img_path_list) == 0:
        raise FileNotFoundError(f"No .jpg images found in {calibration_dir}")

    # Initialise
    num_rows = 9
    num_cols = 6
    term_criteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    # Create the object points (3D coordinates of chessboard corners)
    world_points_template = np.zeros((num_rows * num_cols, 3), np.float32)
    world_points_template[:, :2] = np.mgrid[0:num_rows, 0:num_cols].T.reshape(
        -1, 2
    )

    world_points_list = []
    img_points_list = []
    image_size = None  # We'll set this when we process the first image

    # find corners
    for current_img_path in img_path_list:
        print(f"Processing: {os.path.basename(current_img_path)}")
        img_bgr = cv.imread(current_img_path)

        if img_bgr is None:
            print(f"Warning: Could not load image {current_img_path}")
            continue

        img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

        # Set image size from first successful image
        if image_size is None:
            image_size = img_gray.shape[::-1]

        corners_found, corners_original = cv.findChessboardCorners(
            img_gray, (num_rows, num_cols), None
        )

        if corners_found:
            print("  ✓ Found chessboard corners")
            world_points_list.append(world_points_template.copy())
            corners_refined = cv.cornerSubPix(
                img_gray,
                corners_original,
                (11, 11),
                (-1, -1),
                term_criteria,
            )
            img_points_list.append(corners_refined)

            if show_pics:
                cv.drawChessboardCorners(
                    img_bgr,
                    (num_rows, num_cols),
                    corners_refined,
                    corners_found,
                )
                cv.imshow("Chessboard", img_bgr)
                cv.waitKey(500)
        else:
            print("  ✗ No chessboard corners found")

    cv.destroyAllWindows()

    # Check if we found any valid images
    if len(world_points_list) == 0:
        raise RuntimeError(
            "No chessboard patterns were detected in any images"
        )

    if image_size is None:
        raise RuntimeError("No valid images were processed")

    print(f"Successfully processed {len(world_points_list)} images")

    # Do the actual calibration to find the intrinsics and extrinsics
    reprojection_error, camera_matrix, dist_coeff, r_vecs, t_vecs = (
        cv.calibrateCamera(
            world_points_list,
            img_points_list,
            image_size,  # Use the stored image size instead of img_gray
            None,
            None,
        )
    )
    print("Camera Matrix:\n", camera_matrix)
    print("Reprojection Error (pixels): {:.4f}".format(reprojection_error))

    # Save the calibration parameters in the current folder
    save_dir = os.path.join(current_dir, "camera_calibration.npz")
    np.savez(
        save_dir,
        reprojection_error=reprojection_error,
        camera_matrix=camera_matrix,
        dist_coeff=dist_coeff,
        r_vecs=r_vecs,
        t_vecs=t_vecs,
    )

    return camera_matrix, dist_coeff


def remove_distortion(img, camera_matrix, dist_coeff):
    height, width = img.shape[:2]
    camera_matrix_new, roi = cv.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeff, (width, height), 1, (width, height)
    )
    img_undistorted = cv.undistort(
        img, camera_matrix, dist_coeff, None, camera_matrix_new
    )

    return img_undistorted


if __name__ == "__main__":
    calibrate()
