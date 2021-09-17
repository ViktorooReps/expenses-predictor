import os
import pickle
from collections import defaultdict
from enum import Enum, unique
from typing import Optional, Tuple, Dict, Iterable, Iterator, TypeVar, Type, List, Callable

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
        obj.toJSON = lambda: value
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
    MUSIC = 'Музыка'
    AIR_TRAVEL = 'Авиабилеты'


Expenses = Dict[Category, Value]


def empty_expenses() -> Expenses:
    return {category: 0 for category in Category}


@unique
class Impression(IDEnum):
    LIKE = 'like'
    FAVORITE = 'favorite'
    DISLIKE = 'dislike'


@unique
class Gender(IDEnum):
    MALE = 'M'
    FEMALE = 'F'
    UNKNOWN = 'unk'


@unique
class MaritalStatus(IDEnum):
    UNKNOWN = 'unk'
    MARRIED = 'Женат/замужем'
    WIDOWED = 'Вдовец, вдова'
    SINGLE = 'Холост/не замужем'
    ENGAGED = 'Гражданский брак'
    DIVORCED = 'Разведен (а)'
    APART = 'Не проживает с супругом (ой)'


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

    def __init__(self, date: Date, impressions: Optional[Dict[int, Impression]] = None, balance_change: Optional[Value] = None,
                 expenses: Optional[Expenses] = None, feature_vectors: Optional[Dict[ExtractorName, np.array]] = None):
        self._date = date
        self._impressions = impressions if impressions is not None else {}
        self._balance_change = balance_change if balance_change is not None else 0
        self._expenses = expenses if expenses is not None else empty_expenses()
        self._feature_vectors = feature_vectors if feature_vectors is not None else {}

    @property
    def date(self) -> Date:
        return self._date

    @property
    def impressions(self) -> Dict[int, Impression]:
        return self._impressions

    @property
    def balance_change(self) -> Value:
        return self._balance_change

    @property
    def expenses(self) -> Expenses:
        return self._expenses

    @property
    def feature_vectors(self) -> Dict[ExtractorName, np.array]:
        return self._feature_vectors

    def has_feature_vector(self, extractor: ExtractorName) -> bool:
        return extractor in self._feature_vectors

    def add_feature_vector(self, extractor: ExtractorName, feature_vector: np.array) -> None:
        self._feature_vectors[extractor] = feature_vector

    def get_feature_vector(self, extractor: ExtractorName) -> np.array:
        return self._feature_vectors[extractor]

    def with_changes(self, *, impressions: Optional[Dict[int, Impression]] = None, balance_change: Optional[Value] = None,
                     expenses: Optional[Expenses] = None, feature_vectors: Optional[Dict[ExtractorName, np.array]] = None) -> 'TimeStamp':
        return TimeStamp(date=self._date,
                         impressions=impressions if impressions is not None else self._impressions,
                         balance_change=balance_change if balance_change is not None else self._balance_change,
                         expenses=expenses if expenses is not None else self._expenses,
                         feature_vectors=feature_vectors if feature_vectors is not None else self._feature_vectors)


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

    def __init__(self, user_id: int, feature_vector: np.array, timeline: Iterable[TimeStamp]):
        self._id = user_id
        self._feature_vector = feature_vector
        self._timeline: List[TimeStamp] = list(normalize_timeline(timeline))

    @property
    def id(self) -> int:
        return self._id

    @property
    def feature_vector(self) -> np.array:
        return self._feature_vector

    def feature_vector_at(self, date: Date, extractor: ExtractorName) -> np.array:
        return np.concatenate((self._feature_vector, self._timeline[date].get_feature_vector(extractor)), axis=0)

    @property
    def timeline(self) -> List[TimeStamp]:
        return self._timeline

    def timeline_at(self, date: Date):
        return self._timeline[:date]

    @staticmethod
    def calculate_feature_vector(gender: Gender, age: int, marital_status: MaritalStatus, children: int, region: int,
                                 product_vector: ProductVector):
        arr_buffer = [gender.id, age, marital_status.id, children, region] + list(product_vector)
        return np.array(arr_buffer, dtype=float)

    def popped(self) -> 'User':
        return User(user_id=self._id, feature_vector=self._feature_vector, timeline=self._timeline[:-1])

    def with_changes(self, *, feature_vector: Optional[np.array] = None, timeline: Optional[Iterable[TimeStamp]] = None) -> 'User':
        return User(user_id=self._id,
                    feature_vector=feature_vector if feature_vector is not None else self._feature_vector,
                    timeline=timeline if timeline is not None else self._timeline)


AnySerializable = TypeVar('AnySerializable', bound='Serializable')


class Serializable(object):

    def __init__(self, save_path: os.PathLike):
        self.save_path = save_path

    def save(self, save_path: Optional[os.PathLike] = None) -> None:
        if save_path is None:
            save_path = self.save_path

        with open(save_path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls: AnySerializable, load_path: os.PathLike) -> AnySerializable:
        with open(load_path, 'rb') as f:
            return pickle.load(f)


class Dataset(Serializable):
    __slots__ = (
        '_id2user', '_id2story'
    )

    def __init__(self, users: Iterable[User], stories: Iterable[Story], save_path: os.PathLike):
        super(Dataset, self).__init__(save_path)
        self._id2user = {user.id: user for user in users}
        self._id2story = {story.id: story for story in stories}

    @property
    def users(self) -> List[User]:
        return list(self._id2user.values())

    @property
    def stories(self) -> List[Story]:
        return list(self._id2story.values())

    @property
    def data(self) -> Tuple[List[User], List[Expenses]]:
        """Returns users without last timestamp (X) and their expenses on last timestamp (y)"""
        popped_users = [user.popped() for user in self.users]
        expenses = [user.timeline[-1].expenses for user in self.users]
        return popped_users, expenses

    def modify_users(self, modifier: Callable[[List[User]], List[User]]) -> None:
        self._id2user = {user.id: user for user in modifier(list(self._id2user.values()))}
