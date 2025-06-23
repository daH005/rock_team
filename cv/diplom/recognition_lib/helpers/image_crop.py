import numpy as np

__all__ = (
    'crop_image',
)


def crop_image(image: np.ndarray,
               box: tuple[int, int, int, int],
               ) -> np.ndarray:
    left, top, width, height = box
    if top < 0 or left < 0 or top + height > image.shape[0] or left + width > image.shape[1]:
        raise ValueError
    return image[top:top+height, left:left+width]
