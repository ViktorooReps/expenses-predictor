from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic, List, Iterable

import numpy as np

from data.datamodel import User, Date, TimeStamp, Expenses, Category, Value
from models.registered import ExtractorName


class AbstractPredictor(metaclass=ABCMeta):

    @abstractmethod
    def predict(self, user: User) -> Expenses:
        pass

    def predict_users(self, users: Iterable[User]) -> List[Expenses]:
        return list(self.predict(user) for user in users)

    def _pull_user_features(self, user: User, date: Date) -> np.array:
        return user.feature_vector

    def _pull_user_expenses(self, user: User, date: Date) -> np.array:
        total_expenses = len(Category)
        expenses = np.zeros(total_expenses, dtype=float)

        for category, expense in user.timeline[date].expenses.items():
            expenses[category.id] = expense

        return expenses


class AbstractExtractorAwarePredictor(AbstractPredictor, metaclass=ABCMeta):

    def __init__(self, extractor: ExtractorName):
        self._extractor = extractor

    def _pull_user_features(self, user: User, date: Date) -> np.array:
        return user.feature_vector_at(date, extractor=self._extractor)

    @property
    def extractor_name(self) -> ExtractorName:
        return self._extractor


_PredictorType = TypeVar('_PredictorType', bound=AbstractPredictor)


class AbstractPredictorWrapper(AbstractPredictor, Generic[_PredictorType], metaclass=ABCMeta):

    def __init__(self, predictor: _PredictorType):
        self._predictor = predictor


class AbstractExtractor(metaclass=ABCMeta):

    @abstractmethod
    def extract(self, timestamp: TimeStamp, *, force=False) -> np.array:
        pass
