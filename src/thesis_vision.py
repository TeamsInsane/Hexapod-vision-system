from __future__ import annotations

import cv2
import numpy as np


CANNON_TAG_ID = 0
TARGET_TAG_ID = 1
EXPECTED_TAG_IDS = frozenset({CANNON_TAG_ID, TARGET_TAG_ID})
TAG_SIZE_METERS = 0.0693
CANNON_FORWARD_EDGE = "top"
AIM_TOLERANCE_DEGREES = 3.0

_EDGE_CORNERS = {
    "top": (0, 1),
    "right": (1, 2),
    "bottom": (2, 3),
    "left": (3, 0),
}


#Return the mean of the four image corners - Diplomska enačba 5.2
def marker_center(points: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(points, dtype=np.float64).reshape(4, 2), axis=0)


#Midpoint of one edge
def edge_center(points: np.ndarray, edge: str) -> np.ndarray:
    try:
        first, second = _EDGE_CORNERS[edge]
    except KeyError as error:
        raise ValueError(f"Unknown marker edge: {edge}") from error

    corners = np.asarray(points, dtype=np.float64).reshape(4, 2)
    return np.mean(corners[[first, second]], axis=0)


#gor    = 0°
#desno  = 90°
#dol    = 180°
#levo   = 270°
#Return clockwise image heading in degrees; diplomska enačba 5.3
def direction_heading(direction: np.ndarray) -> float:
    vector = np.asarray(direction, dtype=np.float64).reshape(2)

    if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0:
        raise ValueError("Cannot calculate a heading from an invalid vector")

    return float((np.degrees(np.arctan2(vector[0], -vector[1])) + 360.0) % 360.0)


def cannon_direction(points: np.ndarray) -> tuple[np.ndarray, float]:
    center = marker_center(points)
    direction = edge_center(points, CANNON_FORWARD_EDGE) - center
    length = float(np.linalg.norm(direction))

    if not np.isfinite(length) or length == 0:
        raise ValueError("The cannon tag cannot provide a direction")

    unit_direction = direction / length
    return unit_direction, direction_heading(unit_direction)


#Return target heading and signed shortest turn Diplomska enačba 5.4. (-180, 180)
def aiming_error(cannon_center: np.ndarray, target_center: np.ndarray, cannon_heading: float) -> tuple[float, float]:
    target_heading = direction_heading(np.asarray(target_center) - np.asarray(cannon_center))

    error = (target_heading - float(cannon_heading) + 180.0) % 360.0 - 180.0

    return target_heading, float(error)


def tag_object_points(tag_size_meters: float = TAG_SIZE_METERS) -> np.ndarray:
    if not np.isfinite(tag_size_meters) or tag_size_meters <= 0:
        raise ValueError("tag_size_meters must be finite and positive")

    half_size = tag_size_meters / 2.0
    return np.array(
        [
            [-half_size, half_size, 0.0],
            [half_size, half_size, 0.0],
            [half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0],
        ],
        dtype=np.float64,
    )


# Perspective-n-Point
def estimate_tag_position(points: np.ndarray, camera_matrix: np.ndarray, distortion_coefficients: np.ndarray) -> np.ndarray | None:
    # [X, Y, Z] od kamere
    success, _, translation_vector = cv2.solvePnP(
        tag_object_points(),
        np.asarray(points, dtype=np.float64).reshape(4, 2),
        camera_matrix,
        distortion_coefficients,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if not success:
        return None

    position = translation_vector.reshape(3)

    if not np.all(np.isfinite(position)) or position[2] <= 0:
        return None

    return position


#Return 3-D center distance d_3D in millimetres Diplomska enačba 5.5.
def tag_distance_mm(cannon_position: np.ndarray, target_position: np.ndarray) -> float:
    difference = np.asarray(target_position) - np.asarray(cannon_position)
    distance_meters = float(np.linalg.norm(difference))

    if not np.isfinite(distance_meters):
        raise ValueError("Tag positions must be finite")

    return 1000.0 * distance_meters
