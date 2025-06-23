import cv2
import numpy as np

__all__ = (
    'bytes_to_image',
    'image_to_bytes',
)


def bytes_to_image(image_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def image_to_bytes(image: np.ndarray, format_='JPEG') -> bytes:
    success, encoded_image = cv2.imencode(f'.{format_.lower()}', image)
    if success:
        return encoded_image.tobytes()
    else:
        raise ValueError
