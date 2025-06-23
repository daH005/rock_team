import os
from typing import TypeAlias

import cv2

from .recognition_on_image import recognize_faces_on_image_by_db, RecognitionOnImageResultType

__all__ = (
    'recognize_faces_on_video_by_db',
    'RecognitionOnVideoResultType',
)

RecognitionOnVideoResultType: TypeAlias = list[tuple[float, RecognitionOnImageResultType]]


def recognize_faces_on_video_by_db(video_path: os.PathLike[str] | str,
                                   db_path: os.PathLike[str] | str,
                                   result_save_path: os.PathLike[str] | str,
                                   ) -> RecognitionOnVideoResultType:
    result: RecognitionOnVideoResultType = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0

    ignore_list = set()
    time_in_seconds = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        prev_time_in_seconds = time_in_seconds
        time_in_seconds = round(frame_number / fps, 4)
        if int(prev_time_in_seconds) != int(time_in_seconds):
            # Update the ignore-list after new second
            ignore_list.clear()

        try:
            recognition_result = recognize_faces_on_image_by_db(frame, db_path, result_save_path, ignore_list)

            if recognition_result:
                result.append((time_in_seconds, recognition_result))
                # The ignore-list is needed to avoid big amount of frames in the result
                for record in recognition_result:
                    ignore_list.add(record[0])
        except Exception as e:
            print(e)

        frame_number += 1

    cap.release()
    return result
