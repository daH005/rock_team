from flask import request, abort

from functools import wraps

__all__ = (
    'text_param_decorator',
)


def text_param_decorator(func):
    @wraps(func)
    def wrapper():
        try:
            text: str = request.args['text'][:100]
        except KeyError:
            return abort(400)
        return func(text)
    return wrapper
