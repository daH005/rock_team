import os
from pathlib import Path

import cv2

from .recognition_on_image import recognize_single_face_on_image_by_db

__all__ = (
    'calc_accuracy',
)


def calc_accuracy(db_path: os.PathLike[str] | str) -> float:
    db_path = Path(db_path)
    amount: int = 0
    corrected: int = 0

    for person_folder in db_path.iterdir():
        if not person_folder.is_dir():
            continue

        person_name = person_folder.name
        print(f'Processing the folder "{person_name}"...')
        for image_file in person_folder.iterdir():
            amount += 1
            try:
                if recognize_single_face_on_image_by_db(
                    cv2.imread(str(image_file)),
                    db_path,
                ) == person_name:
                    corrected += 1
                else:
                    raise ValueError
            except ValueError:
                continue

    return corrected / amount
