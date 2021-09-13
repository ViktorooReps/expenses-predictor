from typing import Tuple, Dict, Optional

import numpy as np
from pydantic import BaseModel

from data.datamodel import User, Story, Value, Impression, Gender, MaritalStatus, ProductVector, TransactionType, Category, Date, TimeStamp


class StoryModel(BaseModel):
    id: int
    name: str
    text: str

    def to_story(self) -> Story:
        pass


class TimeStampModel(BaseModel):
    date: Date  # temporary
    impressions: Optional[Dict[Impression, int]]  # impression -> story id
    balance_change: Optional[Value]
    expenses: Optional[Dict[Category, Value]]

    def to_time_stamp(self) -> TimeStamp:
        pass


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
    timeline_features: Optional[Tuple[np.ndarray, ...]]

    def to_user(self) -> User:
        pass


class UserModel(BaseModel):
    id: int
    feature_vector: np.ndarray
    timeline: Tuple[TimeStampModel, ...]

    def to_user(self) -> User:
        pass

    @classmethod
    def from_user(cls) -> 'UserModel':
        pass
