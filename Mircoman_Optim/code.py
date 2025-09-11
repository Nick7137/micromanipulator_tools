import math
from typing import Tuple


def solve_alpha_theta(
    A: Tuple[float, float], D: Tuple[float, float], r: float
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute the possible values of alpha (disc rotation) and theta (grabber angle)
    given:
        A: (Ax, Ay) point on the disc
        D: (Dx, Dy) center of the disc
        r: grabber radius

    Returns:
        (alphas, thetas): each a tuple of two possible solutions (in radians)
    """
    # Vector from disc center to A
    dx = A[0] - D[0]
    dy = A[1] - D[1]

    # Distance from D to A
    L = math.hypot(dx, dy)

    # Angle beta (atan2 in this coordinate system: y downward, so clockwise positive)
    beta = math.atan2(dy, dx)

    # Compute S = L / (2r)
    S = L / (2 * r)

    if abs(S) > 1:
        raise ValueError(
            "No real solution: point cannot reach grabber arc (L > 2r)"
        )

    # Two solutions for alpha
    arcsin_S = math.asin(S)
    alpha1 = arcsin_S - beta
    alpha2 = math.pi - arcsin_S - beta

    # Normalize alphas to [0, 2pi)
    alpha1 = alpha1 % (2 * math.pi)
    alpha2 = alpha2 % (2 * math.pi)

    # Two solutions for theta
    theta1 = 2 * arcsin_S
    theta2 = 2 * math.pi - 2 * arcsin_S

    # Normalize to [0, 2pi)
    theta1 = theta1 % (2 * math.pi)
    theta2 = theta2 % (2 * math.pi)

    return (alpha1, alpha2), (theta1, theta2)


# Example usage
if __name__ == "__main__":
    A = (5.0, 2.0)
    D = (0.0, 0.0)
    r = 5.0

    alphas, thetas = solve_alpha_theta(A, D, r)
    print("Possible alpha values (radians):", alphas)
    print("Possible theta values (radians):", thetas)
