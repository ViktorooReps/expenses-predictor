import enum
import json
from typing import Tuple, Dict, Optional, List

import numpy as np
from pydantic import BaseModel

from data.datamodel import User, Story, Value, Impression, Gender, MaritalStatus, ProductVector, Date, TimeStamp, Expenses
from models.registered import ExtractorName


def _sanitize(o):
    if isinstance(o, dict):
        return {_sanitize(k): _sanitize(v) for k, v in o.items()}
    elif isinstance(o, (set, tuple, list)):
        return type(o)(_sanitize(x) for x in o)
    elif isinstance(o, enum.Enum):
        return o.value
    return o


class HandleKeyEnumEncoder(json.JSONEncoder):
    def encode(self, o):
        return super().encode(_sanitize(o))


class EnumAwareModel(BaseModel):
    def json(self, *args, **kwargs) -> str:
        return super().json(*args, **kwargs, cls=HandleKeyEnumEncoder)


class StoryModel(BaseModel):
    id: int
    name: str
    text: str

    def to_story(self) -> Story:
        return Story(story_id=self.id, name=self.name, text=self.text)


class ExpensesModel(EnumAwareModel):
    __root__: Expenses

    def to_expenses(self):
        return self.__root__

    @classmethod
    def from_expenses(cls, expenses: Expenses):
        return cls.construct(__root__=expenses)


class TimeStampModel(EnumAwareModel):
    date: Date
    impressions: Optional[Dict[int, Impression]]  # story id -> impression
    balance_change: Optional[Value]
    expenses: Optional[ExpensesModel]

    feature_vectors: Optional[Dict[ExtractorName, List[float]]]

    class Config:
        use_enum_values = True

    def to_time_stamp(self) -> TimeStamp:
        return TimeStamp(date=self.date, impressions=self.impressions, balance_change=self.balance_change,
                         expenses=self.expenses.to_expenses(),
                         feature_vectors=self._convert_to_numpy(self.feature_vectors) if self.feature_vectors is not None else {})

    @classmethod
    def from_time_stamp(cls, timestamp: TimeStamp):
        return cls.construct(date=timestamp.date, impressions=timestamp.impressions, balance_change=timestamp.balance_change,
                             expenses=ExpensesModel.from_expenses(timestamp.expenses),
                             feature_vectors=cls._convert_to_list(timestamp.feature_vectors))

    @staticmethod
    def _convert_to_numpy(features: Dict[ExtractorName, List[float]]) -> Dict[ExtractorName, np.array]:
        return {name: np.array(arr, dtype=np.float) for name, arr in features.items()}

    @staticmethod
    def _convert_to_list(features: Dict[ExtractorName, np.array]) -> Dict[ExtractorName, List[float]]:
        return {name: arr.tolist() for name, arr in features.items()}


class UserDataModel(EnumAwareModel):
    id: int
    gender: Gender
    age: int
    marital_status: MaritalStatus
    children: int
    region: int
    product_vector: ProductVector
    timeline: Tuple[TimeStampModel, ...]

    feature_vector: Optional[List[float]]

    class Config:
        use_enum_values = True

    def to_user(self) -> User:
        feature_vector = np.array(self.feature_vector, dtype=np.float) if self.feature_vector is not None else \
            User.calculate_feature_vector(gender=Gender(self.gender), age=self.age, marital_status=MaritalStatus(self.marital_status),
                                          children=self.children, region=self.region, product_vector=self.product_vector)

        return User(user_id=self.id, feature_vector=feature_vector, timeline=[tsm.to_time_stamp() for tsm in self.timeline])


class UserModel(BaseModel):
    id: int
    feature_vector: List[float]
    timeline: Tuple[TimeStampModel, ...]

    def to_user(self) -> User:
        return User(user_id=self.id, feature_vector=np.array(self.feature_vector, dtype=np.float),
                    timeline=[tsm.to_time_stamp() for tsm in self.timeline])

    @classmethod
    def from_user(cls, user: User) -> 'UserModel':
        return cls.construct(id=user.id, feature_vector=user.feature_vector.tolist(),
                             timeline=tuple(TimeStampModel.from_time_stamp(ts) for ts in user.timeline))


class StoryModelList(EnumAwareModel):
    __root__: List[StoryModel]


class UserDataModelList(EnumAwareModel):
    __root__: List[UserDataModel]
