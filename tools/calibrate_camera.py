from pathlib import Path

import cv2
import numpy as np


CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

CHARUCO_SQUARES_X = 5
CHARUCO_SQUARES_Y = 7

CHARUCO_SQUARE_LENGTH_METERS = 0.0285
CHARUCO_MARKER_LENGTH_METERS = 0.01425

MIN_CORNERS_PER_CAPTURE = 8
REQUIRED_CAPTURE_COUNT = 25
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "src" / "camera_calibration.npz"


class CharucoCameraCalibrator:
    def __init__(self, camera_index: int = CAMERA_INDEX, output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
        self.camera_index = camera_index
        self.output_path = output_path

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

        self.board = cv2.aruco.CharucoBoard((CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y), CHARUCO_SQUARE_LENGTH_METERS, CHARUCO_MARKER_LENGTH_METERS, self.dictionary)

        detector_parameters = cv2.aruco.DetectorParameters()
        detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        self.detector = cv2.aruco.CharucoDetector(self.board, cv2.aruco.CharucoParameters(), detector_parameters)

        self.object_points: list[np.ndarray] = []
        self.image_points: list[np.ndarray] = []
        self.image_size: tuple[int, int] | None = None

    @staticmethod
    def normalize_charuco_detection(charuco_corners: np.ndarray | None, charuco_ids: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
        if charuco_corners is None or charuco_ids is None:
            return None, None

        corners = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
        ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)

        if len(corners) == 0 or len(corners) != len(ids):
            return None, None

        return np.ascontiguousarray(corners.reshape(-1, 1, 2)), np.ascontiguousarray(ids.reshape(-1, 1))

    #Open the selected camera
    def open_camera(self) -> cv2.VideoCapture:
        camera = cv2.VideoCapture(self.camera_index)

        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(self.camera_index)

        if not camera.isOpened():
            raise RuntimeError(f"Kamere z indeksom {self.camera_index} ni mogoče odpreti")

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        return camera

    #Add one detected board view to table
    def add_capture(self, charuco_corners: np.ndarray | None, charuco_ids: np.ndarray | None) -> tuple[bool, str]:
        corner_count = 0 if charuco_ids is None else len(charuco_ids)
        if charuco_corners is None or charuco_ids is None or corner_count < MIN_CORNERS_PER_CAPTURE:
            return False, f"Premalo vogalov: {corner_count}/{MIN_CORNERS_PER_CAPTURE}"

        object_points, image_points = self.board.matchImagePoints(charuco_corners, charuco_ids)
        self.object_points.append(np.asarray(object_points, dtype=np.float32).copy())
        self.image_points.append(np.asarray(image_points, dtype=np.float32).copy())

        return True, f"Posnetek {len(self.object_points)} shranjen"

    #Remove the newest accepted
    def remove_last_capture(self) -> str:
        if not self.object_points:
            return "Ni shranjenih posnetkov"

        self.object_points.pop()
        self.image_points.pop()
        return f"Odstranjen zadnji posnetek"

    # Returns: RMS error, camera matrix, distortion coefficients, rotation vectors, translation vectors, and per-view errors.
    def calibrate(self) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray]:
        if len(self.object_points) < REQUIRED_CAPTURE_COUNT:
            raise RuntimeError(f"Za kalibracijo je potrebnih {REQUIRED_CAPTURE_COUNT} posnetkov.")

        if self.image_size is None:
            raise RuntimeError("Velikost slike ni znana")

        # Diplomska enačba 5.1
        """
        K = [fx  0  cx]
            [0  fy  cy]
            [0   0   1]
            fx, fy — goriščni razdalji v pikslih
            cx, cy — optično oziroma glavno središče slike
        """
        (
            rms_error,
            camera_matrix,
            distortion_coefficients,
            rotation_vectors,
            translation_vectors,
        ) = cv2.calibrateCamera(self.object_points,self.image_points,self.image_size,None,None)

        '''
        object_points — znane 3-D točke na tabli
        image_points — dejansko zaznane 2-D točke v sliki
        rotation_vector — nagib/zasuk table glede na kamero
        translation_vector — položaj table glede na kamero
        '''

        per_view_errors = []
        for object_points, image_points, rotation_vector, translation_vector in zip(self.object_points, self.image_points, rotation_vectors, translation_vectors):
            projected_points, _ = cv2.projectPoints(object_points, rotation_vector, translation_vector, camera_matrix, distortion_coefficients)
            differences = image_points.reshape(-1, 2) - projected_points.reshape(-1, 2)

            point_squared_errors = np.sum(differences**2, axis=1)
            per_view_errors.append(float(np.sqrt(np.mean(point_squared_errors))))

        return float(rms_error), camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors, np.asarray(per_view_errors, dtype=np.float64)

    #Save the calibration
    def save_calibration(self, rms_error: float, camera_matrix: np.ndarray, distortion_coefficients: np.ndarray, per_view_errors: np.ndarray) -> None:
        if self.image_size is None:
            raise RuntimeError("Velikost slike ni znana")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.output_path,
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients,
            image_width=self.image_size[0],
            image_height=self.image_size[1],
            rms_error=rms_error,
            per_view_errors=per_view_errors,
            capture_count=len(self.object_points),
            charuco_squares_x=CHARUCO_SQUARES_X,
            charuco_squares_y=CHARUCO_SQUARES_Y,
            charuco_square_length_m=CHARUCO_SQUARE_LENGTH_METERS,
            charuco_marker_length_m=CHARUCO_MARKER_LENGTH_METERS,
            charuco_dictionary="DICT_5X5_100",
        )

    #Draw text on a camera frame
    @staticmethod
    def draw_text(frame: np.ndarray, text: str, position: tuple[int, int], color: tuple[int, int, int], scale: float = 0.7) -> None:
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    def run(self) -> None:
        camera = self.open_camera()
        status = "Move board, press ENTER"
        status_color = (0, 255, 255)

        try:
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Slike iz kamere ni mogoče prebrati")

                height, width = frame.shape[:2]
                current_image_size = (width, height)
                if self.image_size is None:
                    self.image_size = current_image_size

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(gray)
                charuco_corners, charuco_ids = self.normalize_charuco_detection(charuco_corners, charuco_ids)

                if marker_ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

                if charuco_ids is not None:
                    cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, (255, 0, 255))

                corner_count = 0 if charuco_ids is None else len(charuco_ids)
                enough_corners = corner_count >= MIN_CORNERS_PER_CAPTURE
                corner_color = (0, 255, 0) if enough_corners else (0, 165, 255)

                self.draw_text(frame, f"Captured: {len(self.object_points)}/{REQUIRED_CAPTURE_COUNT}", (20, 35), (0, 255, 0))
                self.draw_text(frame, f"ChArUco corners: {corner_count}/{MIN_CORNERS_PER_CAPTURE}", (20, 70), corner_color)

                self.draw_text(frame, status, (20, 105), status_color)

                self.draw_text(frame, "ENTER capture | BACKSPACE undo | C calibrate | Q quit", (20, height - 20), (255, 255, 255), 0.6)

                cv2.imshow("ChArUco calibration", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    return

                if key == 8: #backspace
                    status = self.remove_last_capture()
                    status_color = (0, 255, 255)
                    continue

                if key in (10, 13): #enter
                    added, status = self.add_capture(charuco_corners, charuco_ids)
                    status_color = (0, 255, 0) if added else (0, 0, 255)
                    continue

                if key == ord("c"):
                    if len(self.object_points) < REQUIRED_CAPTURE_COUNT:
                        status = f"Need {REQUIRED_CAPTURE_COUNT} captures"
                        status_color = (0, 0, 255)
                        continue
                    break

        finally:
            camera.release()
            cv2.destroyAllWindows()

        (
            rms_error,
            camera_matrix,
            distortion_coefficients,
            _,
            _,
            per_view_errors,
        ) = self.calibrate()

        self.save_calibration(rms_error,camera_matrix,distortion_coefficients,per_view_errors)

        print(f"Kalibracija shranjena: {self.output_path.resolve()}")

def main() -> None:
    calibrator = CharucoCameraCalibrator()
    calibrator.run()


if __name__ == "__main__":
    main()
