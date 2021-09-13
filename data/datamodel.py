import os
from collections import defaultdict
from enum import Enum, unique
from typing import Optional, Tuple, Dict, Iterable, Iterator, TypeVar, Type, List

import numpy as np

from models.registered import ExtractorName

Date = int  # month number
Value = float


_IDEnum = TypeVar('_IDEnum', bound='IDEnum')


def _id_generator() -> Iterator[int]:
    curr_id = 0
    while True:
        yield curr_id
        curr_id += 1


cls2id_generator: Dict[Type, Iterator[int]] = defaultdict(_id_generator)


class IDEnum(bytes, Enum):
    def __new__(cls, value: str):
        obj = bytes.__new__(cls, value, encoding='utf8')
        obj._value_ = value
        obj.id = next(cls2id_generator[cls])
        return obj

    @classmethod
    def from_id(cls: _IDEnum, target_id: int) -> _IDEnum:
        for member in cls.__members__.values():
            if member.id == target_id:
                return member

        raise ValueError(f'Couldn\'t find member for id {target_id} in {cls}')


@unique
class Category(IDEnum):
    SOUVENIR = 'Сувениры'
    FAST_FOOD = 'Фаст Фуд'
    SUPERMARKETS = 'Супермаркеты'
    HOUSEHOLD = 'Дом/Ремонт'
    SERVICES = 'Сервисные услуги'
    BEAUTY = 'Красота'
    MISCELLANEOUS = 'Разные товары'
    UNKNOWN = 'unk'
    TRANSPORT = 'Транспорт'
    MEDICINE = 'Медицинские услуги'
    FUEL = 'Топливо'
    CLOTHES = 'Одежда/Обувь'
    CASH = 'Наличные'
    COMMUNICATION = 'Связь/Телеком'
    PRIVATE = 'Частные услуги'
    FINANCIAL = 'Финансовые услуги'
    ENTERTAINMENT = 'Развлечения'
    NON_COMMERCIAL = 'НКО'
    CINEMA = 'Кино'
    AUTO = 'Автоуслуги'
    BOOKS = 'Книги'
    HOTEL = 'Отели'
    PHARMACY = 'Аптеки'
    FLOWER = 'Цветы'
    RAILROAD = 'Ж/д билеты'
    RESTAURANT = 'Рестораны'
    SPORT = 'Спорттовары'
    GOVERNMENT = 'Госсборы'
    AUTO_RENT = 'Аренда авто'
    ANIMAL = 'Животные'
    DUTY_FREE = 'Duty Free'
    TOURS = 'Турагентства'
    EDUCATION = 'Образование'
    ART = 'Искусство'
    PHOTO = 'Фото/Видео'


Expenses = Dict[Category, Value]


def empty_expenses() -> Expenses:
    return {category: 0 for category in Category}


@unique
class TransactionType(IDEnum):
    PURCHASE = 'Покупка'
    SERVICE = 'Оплата услуг'
    CASH = 'Снятие наличных'
    PAYMENT = 'Платеж'


@unique
class Impression(IDEnum):
    LIKE = 'like'
    FAVOURITE = 'favourite'
    DISLIKE = 'dislike'


@unique
class Gender(IDEnum):
    MALE = 'M'
    FEMALE = 'F'


@unique
class MaritalStatus(IDEnum):
    UNKNOWN = 'unk'
    MARRIED = 'Женат/замужем'
    WIDOWED = 'Вдовец, вдова'
    SINGLE = 'Холост/не замужем'
    ENGAGED = 'Гражданский брак'
    DIVORCED = 'Разведен (а)'


ProductVector = Tuple[int, int, int, int, int, int, int]


class Story(object):
    __slots__ = (
        '_id', '_name', '_text'
    )

    def __init__(self, story_id: int, name: str, text: str):
        self._id = story_id
        self._name = name
        self._text = text

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def text(self) -> str:
        return self._text


class TimeStamp(object):
    __slots__ = (
        '_date', '_impressions', '_balance_change', '_expenses', '_feature_vectors'
    )

    def __init__(self, date: Date, impressions: Optional[Dict[Impression, int]] = None, balance_change: Optional[Value] = None,
                 expenses: Optional[Dict[Category, Value]] = None, feature_vectors: Optional[Dict[ExtractorName, np.ndarray]] = None):
        self._date = date
        self._impressions = impressions if impressions is not None else {}
        self._balance_change = balance_change if balance_change is not None else 0
        self._expenses = expenses if expenses is not None else empty_expenses()
        self._feature_vectors = feature_vectors if feature_vectors is not None else {}

    @property
    def date(self) -> Date:
        return self._date

    @property
    def impressions(self) -> Dict[Impression, int]:
        return self._impressions

    @property
    def balance_change(self) -> Value:
        return self._balance_change

    @property
    def expenses(self) -> Dict[Category, Value]:
        return self._expenses

    @property
    def feature_vectors(self) -> Dict[ExtractorName, Value]:
        return self._feature_vectors

    def has_feature_vector(self, extractor: ExtractorName) -> bool:
        return extractor in self._feature_vectors

    def add_feature_vector(self, extractor: ExtractorName, feature_vector: np.ndarray) -> None:
        self._feature_vectors[extractor] = feature_vector

    def get_feature_vector(self, extractor: ExtractorName) -> np.ndarray:
        return self._feature_vectors[extractor]


def normalize_timeline(timeline: Iterable[TimeStamp]) -> Iterator[TimeStamp]:
    prev_timestamp = TimeStamp(date=-1)
    for timestamp in timeline:
        while timestamp.date - prev_timestamp.date > 1:
            new_date = prev_timestamp.date + 1
            prev_timestamp = TimeStamp(date=new_date)
            yield TimeStamp(date=new_date)

        prev_timestamp = timestamp
        yield timestamp


class User(object):
    __slots__ = (
        '_id', '_feature_vector', '_timeline'
    )

    def __init__(self, user_id: int, feature_vector: np.ndarray, timeline: Iterable[TimeStamp]):
        self._id = user_id
        self._feature_vector = feature_vector
        self._timeline: List[TimeStamp] = list(normalize_timeline(timeline))

    @property
    def id(self) -> int:
        return self._id

    @property
    def feature_vector(self) -> np.ndarray:
        return self._feature_vector

    def feature_vector_at(self, date: Date, extractor: ExtractorName) -> np.ndarray:
        return np.concatenate((self._feature_vector, self._timeline[date].get_feature_vector(extractor)), axis=0)

    @property
    def timeline(self) -> List[TimeStamp]:
        return self._timeline

    def timeline_at(self, date: Date):
        return self._timeline[:date]

    @staticmethod
    def calculate_feature_vector(gender: Gender, age: int, marital_status: MaritalStatus, children: int, region: int,
                                 product_vector: ProductVector):
        arr_shape = len(product_vector) + 5
        arr_buffer = [gender.id, age, marital_status.id, children, region] + list(product_vector)
        return np.ndarray(shape=(arr_shape, 1), buffer=arr_buffer, dtype=np.float)


class Dataset(object):
    __slots__ = (
        '_id2user', '_id2story'
    )

    def load_dataset(self, users: os.PathLike, stories: os.PathLike):
        pass
