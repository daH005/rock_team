import os
from uuid import uuid4
from traceback import print_exc
from typing import TypeAlias, Iterable

import cv2
import numpy as np
from deepface import DeepFace

from .helpers.image_crop import crop_image

__all__ = (
    'recognize_faces_on_image_by_db',
    'recognize_single_face_on_image_by_db',
    'RecognitionOnImageResultType',
)

RecognitionOnImageResultType: TypeAlias = list[tuple[str, str, tuple[int, int, int, int]]]


def recognize_faces_on_image_by_db(image: np.ndarray,
                                   db_path: os.PathLike[str] | str,
                                   result_save_path: os.PathLike[str] | str,
                                   ignore_list: Iterable[str] | None = None
                                   ) -> RecognitionOnImageResultType:
    if ignore_list is None:
        ignore_list = []

    result: RecognitionOnImageResultType = []
    try:
        detected_faces = DeepFace.extract_faces(img_path=image, detector_backend='yolov8')

        for idx, face_info in enumerate(detected_faces):
            facial_area: dict[str, int] = face_info['facial_area']
            box = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            face_img: np.ndarray = crop_image(image, box)

            try:
                person_name: str = recognize_single_face_on_image_by_db(face_img, db_path)
            except ValueError:
                continue

            if person_name in ignore_list:
                continue

            filename = str(uuid4()) + '.jpg'
            cv2.imwrite(
                os.path.join(result_save_path, filename),
                face_img,
            )

            result.append((
                person_name,
                filename,
                box,
            ))

        return result
    except Exception as e:
        print(e)
        print_exc()


def recognize_single_face_on_image_by_db(image: np.ndarray,
                                         db_path: os.PathLike[str] | str,
                                         ) -> str:
    dfs = DeepFace.find(
        img_path=image,
        db_path=db_path,
        model_name='Facenet512',
        enforce_detection=False,
        threshold=0.3,
        silent=True,
    )

    if len(dfs) <= 0 or dfs[0].empty:
        raise ValueError

    identity_row = dfs[0].iloc[0]
    identity_path = identity_row['identity']
    person_name = os.path.basename(os.path.dirname(identity_path))
    return person_name
