from base64 import b64encode, b64decode

__all__ = (
    'encode_bytes_to_base64',
    'decode_base64_to_bytes',
)


def encode_bytes_to_base64(data: bytes) -> str:
    return b64encode(data).decode()


def decode_base64_to_bytes(data: str) -> bytes:
    return b64decode(data)
