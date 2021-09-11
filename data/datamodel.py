import os
from enum import Enum
from typing import Optional, Any, Tuple

import numpy as np


Date = int  # temporary
Value = float


class Category(Enum):
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


class TransactionType(Enum):
    PURCHASE = 'Покупка'
    SERVICE = 'Оплата услуг'
    CASH = 'Снятие наличных'
    PAYMENT = 'Платеж'


class Impression(Enum):
    LIKE = 'like'
    FAVOURITE = 'favourite'
    DISLIKE = 'dislike'


class Gender(Enum):
    MALE = 'M'
    FEMALE = 'F'


class MaritalStatus(Enum):
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
        pass


class User(object):
    __slots__ = (
        '_id', '_feature_vector', '_timeline'
    )

    def __init__(self, user_id: int, feature_vector: np.ndarray, timeline: Any):
        pass

    def get_feature_vector(self, timestamp: Optional[Date] = None) -> np.ndarray:
        pass

    def get_timeline(self, timestamp: Optional[Date] = None) -> np.ndarray:
        pass

    @staticmethod
    def calculate_feature_vector(gender: Gender, age: int, marital_status: MaritalStatus, children: int, region: int,
                                 product_vector: ProductVector):
        pass


class Dataset(object):
    __slots__ = (
        '_id2user', '_id2story'
    )

    def load_dataset(self, users: os.PathLike, stories: os.PathLike):
        pass
