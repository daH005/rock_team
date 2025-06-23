import os
from typing import IO

__all__ = (
    'save_stream',
)


def save_stream(stream: IO,
                path: os.PathLike[str] | str,
                chunk_size=4096,
                ) -> None:
    with open(path, 'wb') as f:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
