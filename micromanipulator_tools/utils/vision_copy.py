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

from typing import Tuple, List


class Vision:
    # -------------------------------------------------------------------------
    # Private vision methods---------------------------------------------------
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Private helper functions for detection methods---------------------------
    # -------------------------------------------------------------------------

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
