from __future__ import annotations

from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "tags"
TAG_PIXELS = 1000
WHITE_MARGIN_PIXELS = 125
TAG_NAMES = {0: "cannon", 1: "target"}

#Return one marker
def printable_tag(dictionary: cv2.aruco.Dictionary, tag_id: int,) -> cv2.typing.MatLike:
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, TAG_PIXELS)
    return cv2.copyMakeBorder(
        marker,
        WHITE_MARGIN_PIXELS,
        WHITE_MARGIN_PIXELS,
        WHITE_MARGIN_PIXELS,
        WHITE_MARGIN_PIXELS,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def main() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    for tag_id, name in TAG_NAMES.items():
        path = OUTPUT_DIRECTORY / f"tag_{tag_id}_{name}.png"

        if not cv2.imwrite(str(path), printable_tag(dictionary, tag_id)):
            raise OSError(f"Slike ni bilo mogoče shraniti: {path}")
        print(f"Ustvarjen: {path}")


if __name__ == "__main__":
    main()
