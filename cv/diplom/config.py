import os

from dotenv import load_dotenv
from pathlib import Path

__all__ = (
    'BASE_DIR',
    'DB_PATH',
    'RESULT_IMAGES_DIR_PATH',
    'HOST',
    'PORT',
    'DEBUG',
)

load_dotenv()

BASE_DIR: Path = Path(__file__).parent
DB_PATH: Path = BASE_DIR.joinpath('photos')
RESULT_IMAGES_DIR_PATH: Path = BASE_DIR.joinpath('media/result_images')

HOST: str = os.getenv('HOST')
PORT: int = int(os.getenv('PORT'))
DEBUG: bool = os.getenv('DEBUG') == 'True'
