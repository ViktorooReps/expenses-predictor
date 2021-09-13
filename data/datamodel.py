import os
from enum import Enum, unique
from typing import Optional, Any, Tuple, Dict, Iterable, Iterator

import numpy as np


Date = int  # temporary
Value = float


class IDEnum(bytes, Enum):
    def __new__(cls, value: str):
        obj = bytes.__new__(cls, value, encoding='utf8')
        obj._value_ = value
        obj.id = next(cls._id_generator())
        return obj

    @staticmethod
    def _id_generator() -> Iterator[int]:
        curr_id = 0
        while True:
            yield curr_id
            curr_id += 1


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
    # etc


def empty_expenses() -> Dict[Category, Value]:
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
        '_date', '_impressions', '_balance_change', '_expenses'
    )

    def __init__(self, date: Date, impressions: Optional[Dict[Impression, int]] = None, balance_change: Optional[Value] = None,
                 expenses: Optional[Dict[Category, Value]] = None):
        self._date = date
        self._impressions = impressions if impressions is not None else {}
        self._balance_change = balance_change if balance_change is not None else 0
        self._expenses = expenses if expenses is not None else empty_expenses()

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
        self._timeline = tuple(normalize_timeline(timeline))

    @property
    def feature_vector(self) -> np.ndarray:
        return self._feature_vector

    @property
    def timeline(self) -> Tuple[TimeStamp]:
        return self._timeline

    def get_timeline_at(self, date: Date):
        return self._timeline[:date]

    @staticmethod
    def calculate_feature_vector(gender: Gender, age: int, marital_status: MaritalStatus, children: int, region: int,
                                 product_vector: ProductVector):
        arr_shape = len(product_vector) + 5
        arr_buffer = [gender.id, age, marital_status.id, children, region] + list(product_vector)
        return np.ndarray(shape=(arr_shape, 1), buffer=arr_buffer)


class Dataset(object):
    __slots__ = (
        '_id2user', '_id2story'
    )

    def load_dataset(self, users: os.PathLike, stories: os.PathLike):
        pass
