import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .thesis_vision import (
        AIM_TOLERANCE_DEGREES,
        CANNON_TAG_ID,
        EXPECTED_TAG_IDS,
        TARGET_TAG_ID,
        aiming_error,
        cannon_direction,
        estimate_tag_position,
        marker_center,
        tag_distance_mm,
    )
else:
    from thesis_vision import (
        AIM_TOLERANCE_DEGREES,
        CANNON_TAG_ID,
        EXPECTED_TAG_IDS,
        TARGET_TAG_ID,
        aiming_error,
        cannon_direction,
        estimate_tag_position,
        marker_center,
        tag_distance_mm,
    )


CAMERA_INDEX = 4
CAMERA_CALIBRATION_PATH = Path(__file__).with_name(
    "camera_calibration.npz"
)

CANNON_ARROW_LENGTH = 180
CANNON_DIRECTION_COLOR = (0, 165, 255)
TARGET_DIRECTION_COLOR = (255, 255, 0)

ESP32_AIM_URL = "http://192.168.4.1/api/aim"
ESP32_AIM_LOST_URL = "http://192.168.4.1/api/aim/lost"
AIM_SEND_INTERVAL_SECONDS = 0.25
AIM_HTTP_TIMEOUT_SECONDS = 0.40
#Load camera calibration data
def load_camera_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int], float]:
    if not path.is_file():
        raise FileNotFoundError(f"Kalibracijska datoteka ne obstaja: {path}")

    with np.load(path, allow_pickle=False) as calibration:
        required_keys = {
            "camera_matrix",
            "distortion_coefficients",
            "image_width",
            "image_height",
            "rms_error",
        }

        missing_keys = required_keys.difference(calibration.files)
        if missing_keys:
            raise ValueError(f"V kalibracijski datoteki manjkajo podatki: {sorted(missing_keys)}")

        camera_matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)

        distortion_coefficients = np.asarray(calibration["distortion_coefficients"], dtype=np.float64)
        image_size = (int(calibration["image_width"]), int(calibration["image_height"]))
        rms_error = float(calibration["rms_error"])

    if camera_matrix.shape != (3, 3):
        raise ValueError("Matrika kamere mora imeti obliko 3 x 3")

    if distortion_coefficients.size < 4:
        raise ValueError("Kalibracija mora vsebovati vsaj štiri koeficiente popačenja.")

    if not np.all(np.isfinite(camera_matrix)) or not np.all(np.isfinite(distortion_coefficients)) or not np.isfinite(rms_error):
        raise ValueError("Kalibracijska datoteka vsebuje neveljavne vrednosti.")

    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError(f"Neveljavna kalibracijska ločljivost: {image_size}.")

    return camera_matrix, distortion_coefficients, image_size, rms_error


#Create the tag detector
def create_tag_detector() -> cv2.aruco.ArucoDetector:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, parameters)


#Map raw corners into the undistorted image
def undistort_tag_corners(corners: list[np.ndarray], camera_matrix: np.ndarray, distortion_coefficients: np.ndarray) -> list[np.ndarray]:
    return [
        cv2.undistortPoints(np.asarray(marker_corners, dtype=np.float32).reshape(-1, 1, 2), camera_matrix, distortion_coefficients, P=camera_matrix).reshape(1, 4, 2)
        for marker_corners in corners
    ]


def draw_text_box(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int], font_scale: float = 0.55, thickness: int = 2) -> tuple[int, int, int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    padding = 4
    frame_height, frame_width = frame.shape[:2]
    maximum_x = max(padding, frame_width - text_width - padding)
    x = min(max(int(origin[0]), padding), maximum_x)
    y = min(
        max(int(origin[1]), text_height + baseline + padding),
        frame_height - baseline - padding,
    )

    left = x - padding
    top = y - text_height - padding
    right = x + text_width + padding
    bottom = y + baseline + padding
    background = frame[top:bottom + 1, left:right + 1]
    cv2.addWeighted(background, 0.32, np.zeros_like(background), 0.68, 0.0, dst=background)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return left, top, right, bottom


def draw_centered_text_box(frame: np.ndarray, text: str, center: np.ndarray, color: tuple[int, int, int], font_scale: float = 0.55) -> None:
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    draw_text_box(
        frame,
        text,
        (
            int(round(float(center[0]) - text_width / 2.0)),
            int(round(float(center[1]) + text_height / 2.0)),
        ),
        color,
        font_scale,
    )


def draw_callout_label(frame: np.ndarray, text: str, anchor: np.ndarray, color: tuple[int, int, int]) -> None:
    frame_height, frame_width = frame.shape[:2]
    (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    anchor_x, anchor_y = np.rint(anchor).astype(int)

    if anchor_x < frame_width / 2:
        label_x = anchor_x + 42
    else:
        label_x = anchor_x - text_width - 50
    label_y = anchor_y + 48 if anchor_y < frame_height / 2 else anchor_y - 38

    left, top, right, bottom = draw_text_box(frame, text, (int(label_x), int(label_y)),color)
    line_end = (min(max(int(anchor_x), left), right), min(max(int(anchor_y), top), bottom))

    cv2.line(frame, (int(anchor_x), int(anchor_y)), line_end, color, 2, cv2.LINE_AA)
    cv2.circle(frame, (int(anchor_x), int(anchor_y)), 4, color, cv2.FILLED, cv2.LINE_AA)


@dataclass(frozen=True)
class TagObservation:
    marker_id: int
    points: np.ndarray
    center: np.ndarray
    position: np.ndarray | None = None
    forward_direction: np.ndarray | None = None
    heading: float | None = None


@dataclass(frozen=True)
class FrameAnalysis:
    tags: dict[int, TagObservation]
    target_heading: float | None = None
    aiming_error: float | None = None
    distance_mm: float | None = None

    @property
    def detected_ids(self) -> set[int]:
        return set(self.tags)

    @property
    def cannon(self) -> TagObservation | None:
        return self.tags.get(CANNON_TAG_ID)

    @property
    def target(self) -> TagObservation | None:
        return self.tags.get(TARGET_TAG_ID)

    @property
    def has_aim(self) -> bool:
        return (
            self.cannon is not None
            and self.target is not None
            and self.cannon.heading is not None
            and self.target_heading is not None
            and self.aiming_error is not None
        )


# Diplomska enačba 5.2--5.5
def analyze_detections(corners: list[np.ndarray], ids: np.ndarray | None, camera_matrix: np.ndarray, distortion_coefficients: np.ndarray) -> FrameAnalysis:
    tags: dict[int, TagObservation] = {}
    if ids is None:
        return FrameAnalysis(tags)

    for marker_corners, marker_id_value in zip(corners, np.asarray(ids).reshape(-1),):
        marker_id = int(marker_id_value)
        points = np.asarray(marker_corners).reshape(4, 2)
        center = marker_center(points)
        position = None
        direction = None
        heading = None

        if marker_id in EXPECTED_TAG_IDS:
            position = estimate_tag_position(points, camera_matrix, distortion_coefficients)

        if marker_id == CANNON_TAG_ID:
            try:
                direction, heading = cannon_direction(points)
            except ValueError:
                pass

        tags[marker_id] = TagObservation(
            marker_id=marker_id,
            points=points,
            center=center,
            position=position,
            forward_direction=direction,
            heading=heading,
        )

    cannon = tags.get(CANNON_TAG_ID)
    target = tags.get(TARGET_TAG_ID)
    distance_mm = None
    target_heading = None
    error = None
    if cannon and target:
        if cannon.position is not None and target.position is not None:
            distance_mm = tag_distance_mm(cannon.position, target.position)
        if cannon.heading is not None:
            target_heading, error = aiming_error(cannon.center, target.center, cannon.heading)

    return FrameAnalysis(tags, target_heading, error, distance_mm)


def draw_analysis(frame: np.ndarray, analysis: FrameAnalysis) -> None:
    for observation in analysis.tags.values():
        marker_id = observation.marker_id
        label = f"ID {marker_id} - NEZNANA OZNAKA"
        color = (0, 255, 0)

        if marker_id == CANNON_TAG_ID:
            label = "ID 0 - IZSTRELITVENI SISTEM"
            color = CANNON_DIRECTION_COLOR
            if observation.forward_direction is not None:
                end = np.rint(observation.center + observation.forward_direction * CANNON_ARROW_LENGTH).astype(int)
                cv2.arrowedLine(frame, tuple(np.rint(observation.center).astype(int)), tuple(end), color, 5, tipLength=0.20)
                draw_text_box(frame, "SMER CEVI", (int(end[0]) + 8, int(end[1]) - 8), color)

        elif marker_id == TARGET_TAG_ID:
            label = "ID 1 - TARCA"

        draw_callout_label(frame, label, observation.center, color)

    if not analysis.has_aim:
        return

    cannon_center = analysis.cannon.center
    target_center = analysis.target.center
    cv2.arrowedLine(frame, tuple(np.rint(cannon_center).astype(int)), tuple(np.rint(target_center).astype(int)), TARGET_DIRECTION_COLOR, 3, tipLength=0.08)

    delta = target_center - cannon_center
    delta_length = float(np.linalg.norm(delta))
    normal = (
        np.array([-delta[1], delta[0]], dtype=np.float64) / delta_length
        if delta_length > 0
        else np.array([0.0, 1.0])
    )
    midpoint = (cannon_center + target_center) / 2.0
    distance_label = "d_3D = NI NA VOLJO"

    if analysis.distance_mm is not None:
        value = f"{analysis.distance_mm / 10.0:.1f}".replace(".", ",")
        distance_label = f"d_3D = {value} cm"

    draw_centered_text_box(frame, distance_label, midpoint - normal * 27.0, TARGET_DIRECTION_COLOR)


@dataclass(frozen=True)
class AimDisplay:
    instruction: str
    details: str
    distance: str
    color: tuple[int, int, int]


def aim_display(analysis: FrameAnalysis) -> AimDisplay:
    if not analysis.has_aim:
        return AimDisplay(
            "Show both CANNON (0) and TARGET (1)",
            "",
            "",
            (0, 255, 255),
        )

    assert analysis.aiming_error is not None
    assert analysis.cannon is not None
    assert analysis.cannon.heading is not None
    assert analysis.target_heading is not None
    error = analysis.aiming_error
    if abs(error) <= AIM_TOLERANCE_DEGREES:
        instruction = f"ALIGNED ({abs(error):.1f} deg error)"
        color = (0, 255, 0)
    elif error > 0:
        instruction = f"TURN RIGHT {error:.1f} deg"
        color = (0, 165, 255)
    else:
        instruction = f"TURN LEFT {abs(error):.1f} deg"
        color = (0, 165, 255)

    details = (
        f"Cannon {analysis.cannon.heading:.1f} deg | "
        f"Target {analysis.target_heading:.1f} deg"
    )
    distance = "3D center distance unavailable"
    if analysis.distance_mm is not None:
        distance = f"3D center distance: {analysis.distance_mm / 10.0:.1f} cm"
    return AimDisplay(instruction, details, distance, color)


#POST one measured aiming error to the receive-only ESP32
def send_aim_measurement(sequence: int,rotation: float, cannon_heading: float, target_heading: float, target_distance_cm: float,) -> tuple[bool, str]:
    form_data = urllib.parse.urlencode(
        {
            "sequence": sequence,
            "error_deg": f"{rotation:.2f}",
            "cannon_heading_deg": f"{cannon_heading:.2f}",
            "target_heading_deg": f"{target_heading:.2f}",
            "target_distance_cm": f"{target_distance_cm:.2f}",
        }
    ).encode("ascii")

    request = urllib.request.Request(
        ESP32_AIM_URL,
        data=form_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=AIM_HTTP_TIMEOUT_SECONDS) as response:
            response.read()
            if response.status != 200:
                return False, f"ESP32: HTTP {response.status}"
    except (urllib.error.URLError, TimeoutError) as error:
        reason = getattr(error, "reason", error)
        return False, f"ESP32: send failed ({reason})"

    return True, f"ESP32: received measurement #{sequence}"


def send_aim_lost() -> tuple[bool, str]:
    request = urllib.request.Request(ESP32_AIM_LOST_URL, data=b"", method="POST")

    try:
        with urllib.request.urlopen(request, timeout=AIM_HTTP_TIMEOUT_SECONDS) as response:
            response.read()
            if response.status != 200:
                return False, f"ESP32: aim lost HTTP {response.status}"
    except (urllib.error.URLError, TimeoutError) as error:
        reason = getattr(error, "reason", error)
        return False, f"ESP32: aim lost send failed ({reason})"

    return True, "ESP32: aim lost"


class AimSender:

    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="esp32-aim-sender")
        self.future: Future[tuple[bool, str]] | None = None
        self.last_send_time = 0.0
        self.sequence = 0
        self.ok: bool | None = None
        self.status = "ESP32: waiting for both tags"
        self.measurement_available: bool | None = None

    def poll(self) -> None:
        if self.future is None or not self.future.done():
            return

        try:
            self.ok, self.status = self.future.result()
        except Exception as error:
            self.ok = False
            self.status = f"ESP32: sender error ({error})"

        self.future = None

    def send_if_due(self, analysis: FrameAnalysis) -> None:
        measurement_available = analysis.has_aim and analysis.distance_mm is not None
        if not measurement_available:
            if self.measurement_available is not False:
                self.future = self.executor.submit(send_aim_lost)
                self.ok = None
                self.status = "ESP32: sending aim lost"
            self.measurement_available = False
            return

        self.measurement_available = True
        if self.future is not None or time.monotonic() - self.last_send_time < AIM_SEND_INTERVAL_SECONDS:
            return

        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        self.future = self.executor.submit(
            send_aim_measurement,
            self.sequence,
            analysis.aiming_error,
            analysis.cannon.heading,
            analysis.target_heading,
            analysis.distance_mm / 10.0,
        )

        self.last_send_time = time.monotonic()
        self.ok = None
        self.status = f"ESP32: sending measurement #{self.sequence}"

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)


def open_camera(index: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(index)

    if not camera.isOpened():
        raise RuntimeError(f"Kamere z indeksom {index} ni bilo mogoče odpreti")

    if sys.platform.startswith("linux"):
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return camera


def draw_diagnostics(frame: np.ndarray, display: AimDisplay, sender: AimSender, undistortion_enabled: bool, calibration_rms_error: float) -> None:
    lines = ((display.instruction, (20, 35), 1.0, display.color, 3), (display.details, (20, 65), 0.7, display.color, 2))

    for text, position, scale, color, thickness in lines:
        if text:
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    sender_color = (
        (0, 255, 0)
        if sender.ok is True
        else (0, 0, 255)
        if sender.ok is False
        else (0, 255, 255)
    )
    bottom_lines = (
        ("Camera: "
            + ("UNDISTORTED" if undistortion_enabled else "RAW")
            + f" | RMS {calibration_rms_error:.3f} px | U toggle",
            frame.shape[0] - 55,
            (255, 255, 255),
        ),
        (sender.status, frame.shape[0] - 20, sender_color),
    )

    for text, y, color in bottom_lines:
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)


def main() -> None:
    (
        camera_matrix,
        distortion_coefficients,
        calibrated_image_size,
        calibration_rms_error,
    ) = load_camera_calibration(CAMERA_CALIBRATION_PATH)

    detector = create_tag_detector()
    camera = open_camera(CAMERA_INDEX)
    sender = AimSender()
    undistortion_maps: tuple[np.ndarray, np.ndarray] | None = None
    undistortion_enabled = True

    try:
        while True:
            sender.poll()
            success, frame = camera.read()

            if not success:
                print("Slike iz kamere ni bilo mogoče prebrati")
                break

            frame_size = (frame.shape[1], frame.shape[0]) # (širina, višina)
            if frame_size != calibrated_image_size:
                raise RuntimeError("Ločljivost kamere se ne ujema s kalibracijo")

            if undistortion_maps is None:
                undistortion_maps = cv2.initUndistortRectifyMap(
                    camera_matrix,
                    distortion_coefficients,
                    None,
                    camera_matrix,
                    frame_size,
                    cv2.CV_16SC2,
                )

            detection_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(detection_gray)

            if undistortion_enabled:
                frame = cv2.remap(frame, undistortion_maps[0], undistortion_maps[1], cv2.INTER_LINEAR)
                corners = undistort_tag_corners(corners, camera_matrix, distortion_coefficients)

                pose_distortion_coefficients = np.zeros_like(distortion_coefficients)
            else:
                pose_distortion_coefficients = distortion_coefficients

            analysis = analyze_detections(corners, ids, camera_matrix, pose_distortion_coefficients)
            draw_analysis(frame, analysis)
            display = aim_display(analysis)
            sender.send_if_due(analysis)

            draw_diagnostics(frame, display, sender, undistortion_enabled, calibration_rms_error)

            cv2.imshow("Hexapod tag detection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

            if key in (ord("u"), ord("U")):
                undistortion_enabled = not undistortion_enabled
    finally:
        sender.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
