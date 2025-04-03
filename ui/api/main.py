from flask import Flask

from translator import Translator
from decorator import text_param_decorator

app = Flask(__name__)
translator = Translator('gemma3')


@app.route('/translate/russian_to_elvish')
@text_param_decorator
def translate_russian_to_elvish(text: str) -> str:
    return translator.translate_russian_to_elvish(text)


@app.route('/translate/elvish_to_russian')
@text_param_decorator
def translate_elvish_to_russian(text: str) -> str:
    return translator.translate_elvish_to_russian(text)


@app.route('/translate/english_to_elvish')
@text_param_decorator
def translate_english_to_elvish(text: str) -> str:
    return translator.translate_english_to_elvish(text)


@app.route('/translate/elvish_to_english')
@text_param_decorator
def translate_elvish_to_english(text: str) -> str:
    return translator.translate_elvish_to_english(text)


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=8080,
    )
