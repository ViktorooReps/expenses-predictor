from typing import Tuple, Dict, Optional

import numpy as np
from pydantic import BaseModel

from data.datamodel import User, Story, Value, Impression, Gender, MaritalStatus, ProductVector, Date, TimeStamp, Expenses
from models.registered import ExtractorName


class StoryModel(BaseModel):
    id: int
    name: str
    text: str

    def to_story(self) -> Story:
        return Story(story_id=self.id, name=self.name, text=self.text)


class TimeStampModel(BaseModel):
    date: Date
    impressions: Optional[Dict[Impression, int]]  # impression -> story id
    balance_change: Optional[Value]
    expenses: Optional[Expenses]

    feature_vectors: Optional[Dict[ExtractorName, np.ndarray]]

    def to_time_stamp(self) -> TimeStamp:
        return TimeStamp(date=self.date, impressions=self.impressions, balance_change=self.balance_change, expenses=self.expenses,
                         feature_vectors=self.feature_vectors)

    @classmethod
    def from_time_stamp(cls, timestamp: TimeStamp):
        return cls.construct(date=timestamp.date, impressions=timestamp.impressions, balance_change=timestamp.balance_change,
                             expenses=timestamp.expenses, feature_vectors=timestamp.feature_vectors)


class UserDataModel(BaseModel):
    id: int
    gender: Gender
    age: int
    marital_status: MaritalStatus
    children: int
    region: int
    product_vector: ProductVector
    timeline: Tuple[TimeStampModel, ...]

    feature_vector: Optional[np.ndarray]

    def to_user(self) -> User:
        feature_vector = self.feature_vector if self.feature_vector is not None else User.calculate_feature_vector(
            gender=self.gender, age=self.age, marital_status=self.marital_status, children=self.children, region=self.region,
            product_vector=self.product_vector)

        return User(user_id=self.id, feature_vector=feature_vector, timeline=[tsm.to_time_stamp() for tsm in self.timeline])


class UserModel(BaseModel):
    id: int
    feature_vector: np.ndarray
    timeline: Tuple[TimeStampModel, ...]

    def to_user(self) -> User:
        return User(user_id=self.id, feature_vector=self.feature_vector, timeline=[tsm.to_time_stamp() for tsm in self.timeline])

    @classmethod
    def from_user(cls, user: User) -> 'UserModel':
        return cls.construct(id=user.id, feature_vector=user.feature_vector,
                             timeline=tuple(TimeStampModel.from_time_stamp(ts) for ts in user.timeline))
