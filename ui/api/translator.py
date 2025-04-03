import pandas as pd
import ollama
import pymorphy2
lemmer = pymorphy2.MorphAnalyzer()

import string

__all__ = (
    'Translator',
)

df = pd.read_csv('../../lr3/witcher_words.csv')


class Translator:
    """
    Обёртка над моделью LLM из Olama для реализации перевода.
    Реализованы двухсторонние переводы с русского/английского на Старшую речь.
    Поступаемый текст сначала токенизируется, лемматизируется, потом происходит поиск используемых слов
    по словарю.
    После всего этого исходный текст (всегда русский либо эльфский) и собранная часть словаря отправляются модели.
    Это ускорило обработку во много раз, поскольку изначально модели залился словарь целиком (это был этап "инициализации"),
    а потом происходили переводы.
    Здесь нам даже не нужно хранить историю предыдущих сообщений.
    """

    _RESULT_LIMITER: str = 'RESULT'
    _SYNONYM_SEPARATOR: str = ','

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    def translate_russian_to_elvish(self, text: str) -> str:
        """Переводит русский текст на Старшую Речь."""
        dictionary = self._make_dictionary_by_russian_text(text)
        return self._request(f'Переведи русский текст "{text}" на другой язык согласно словарю:\n' + dictionary)

    def translate_elvish_to_russian(self, text: str) -> str:
        """Переводит текст на языке Старшей Речи на русский."""
        dictionary = self._make_dictionary_by_elvish_text(text)
        return self._request(f'Переведи вымышленный текст "{text}" на русский согласно словарю:\n' + dictionary)

    def translate_english_to_elvish(self, text: str) -> str:
        """Переводит английский текст на Старшую Речь."""
        text = self._request(f'Переведи английский текст "{text}" на русский.')
        return self.translate_russian_to_elvish(text)

    def translate_elvish_to_english(self, text: str) -> str:
        """Переводит текст на языке Старшей Речи на английский."""
        text = self.translate_elvish_to_russian(text)
        return self._request(f'Переведи русский текст "{text}" на английский.')

    def _make_dictionary_by_russian_text(self, text: str) -> str:
        """Формирует текстовый словарь, опираясь на слова используемые в русском тексте."""
        dictionary: str = ''

        words = self._extract_lemms(text)
        for word in words:
            dictionary += word + ' - ' + _translate_russian_word(word, self._synonyms_of_word) + '\n'
            for translation, phrase in self._russian_phrases_that_starts_with_word_and_translations(word):
                dictionary += phrase + ' - ' + translation + '\n'

        return dictionary

    def _make_dictionary_by_elvish_text(self, text: str) -> str:
        """Формирует текстовый словарь, опираясь на слова используемые в тексте на языке Старшей Речи."""
        dictionary: str = ''

        words = self._extract_words(text)
        for word in words:
            for translation in self._elvish_word_translations(word):
                dictionary += word + ' - ' + translation + '\n'
            for phrase, translation in self._elvish_phrases_that_starts_with_word_and_translations(word):
                dictionary += phrase + ' - ' + translation + '\n'

        return dictionary

    def _extract_lemms(self, text: str) -> list[str]:
        """Извлекает из текста слова и приводит их к начальной форме."""
        return [_word_to_lemm(word) for word in self._extract_words(text)]

    def _extract_words(self, text: str) -> list[str]:
        """Извлекает из текста слова и переводит их в нижний регистр."""
        text = self._remove_punctuation(text)
        return text.lower().split()

    @staticmethod
    def _remove_punctuation(text: str) -> str:
        """Удаляет из текста все знаки препинания и пунктуации."""
        return text.translate(
            str.maketrans('', '', string.punctuation),
        )

    def _synonyms_of_word(self, word: str) -> list[str]:
        """Выдаёт синонимы слова, которые подобрала модель."""
        text = self._request(f'Подбери синонимы слова "{word}". Разделяй их через {self._SYNONYM_SEPARATOR}.')
        return [word.strip() for word in text.split(self._SYNONYM_SEPARATOR)]

    def _russian_phrases_that_starts_with_word_and_translations(self, word: str) -> list[tuple[str, str]]:
        """
        Выдает все фразы и их переводы, которые начинаются с переданного русского слова.
        Первое в каждом кортеже эльфское, второе - русское.
        """
        return self._phrases_that_starts_with_word_and_translations(word, 'translation')

    def _elvish_phrases_that_starts_with_word_and_translations(self, word: str) -> list[tuple[str, str]]:
        """
        Выдает все фразы и их переводы, которые начинаются с переданного слова на Старшей Речи.
        Первое в каждом кортеже эльфское, второе - русское.
        """
        return self._phrases_that_starts_with_word_and_translations(word, 'text')

    @staticmethod
    def _phrases_that_starts_with_word_and_translations(word: str, col: str) -> list[tuple[str, str]]:
        """
        Выдает все фразы и их переводы, которые начинаются с переданного слова в заданном столбце `df`.
        Первое в каждом кортеже эльфское, второе - русское.
        """
        return df[df[col].str.startswith(word + ' ')].itertuples(index=False)

    @staticmethod
    def _elvish_word_translations(word: str) -> list[str]:
        """Выдает всевозможные переводы слова на Старшей Речи."""
        result = df.loc[df['text'] == word, 'translation']
        return result.values

    def _request(self, text: str, extract_result: bool = True) -> str:
        """Отправляет запрос модели и выдает ответ."""
        if extract_result:
            text += f'\nСлева и справа от результата напиши "{self._RESULT_LIMITER}". Больше ничего лишнего не пиши.'

        answer = ollama.chat(model=self._model_name, messages=[{
            'role': 'user',
            'content': text,
        }])['message']['content']

        # DeepSeek thinking isn't needed :)
        if '</think>' in answer:
            answer = answer.split('</think>')[1]

        answer = answer.split(self._RESULT_LIMITER)[1]
        return answer.strip()


def _word_to_lemm(word: str) -> str:
    """Переводит слово/словосочетание в начальную форму."""
    parts = []
    for part in word.split():
        part = lemmer.parse(part)[0].normal_form.replace('ё', 'е')
        parts.append(part)

    return ' '.join(parts)


def _translate_russian_word(word: str, synonyms_finding_func) -> str:
    """
    Переводит русское слово/словосочетание на Старшую Речь.
    При отсутствии слова в датасете использует перебор синонимов из функции `synonyms_finding_func`.
    Если последнее так же не помогло, то делает латинизацию слова.
    """
    translation = None
    try:
        translation = _try_to_find_russian_word_translation(word)
    except ValueError:
        for synonym in synonyms_finding_func(word):
            try:
                translation = _try_to_find_russian_word_translation(synonym)
            except ValueError:
                continue

    if translation is None:
        translation = _transliterate_russian_to_latin(word)

    return translation


def _try_to_find_russian_word_translation(word: str) -> str:
    """Ищет перевод русского слова/словосочетания в датасете."""
    result = df.loc[df['translation'] == word, 'text']
    if not result.empty:
        return result.values[0]
    else:
        raise ValueError


def _find_phrases_that_starts_with_russian_word(word: str) -> list[str]:
    return df.loc[df['translation'].str.startswith(word + ' '), 'translation'].tolist()


def _transliterate_russian_to_latin(russian_word: str) -> str:
    """Функция для латинизации русского слова."""
    translit_dict = {
        'а': 'a',
        'б': 'b',
        'в': 'v',
        'г': 'g',
        'д': 'd',
        'е': 'e',
        'ё': 'yo',
        'ж': 'zh',
        'з': 'z',
        'и': 'i',
        'й': 'y',
        'к': 'k',
        'л': 'l',
        'м': 'm',
        'н': 'n',
        'о': 'o',
        'п': 'p',
        'р': 'r',
        'с': 's',
        'т': 't',
        'у': 'u',
        'ф': 'f',
        'х': 'kh',
        'ц': 'ts',
        'ч': 'ch',
        'ш': 'sh',
        'щ': 'shch',
        'ъ': '',
        'ы': 'y',
        'ь': '',
        'э': 'e',
        'ю': 'yu',
        'я': 'ya',
    }

    latin_word = ''.join(translit_dict.get(symbol, symbol) for symbol in russian_word.lower())
    return latin_word
