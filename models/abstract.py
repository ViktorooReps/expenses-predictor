from abc import ABCMeta, abstractmethod
from typing import Optional, TypeVar, Generic

import numpy as np

from data.datamodel import User, Date, TimeStamp, Expenses, Category
from models.registered import ExtractorName


class AbstractPredictor(metaclass=ABCMeta):

    @abstractmethod
    def predict(self, user: User, date: Optional[Date] = None) -> Expenses:
        pass

    def _pull_user_features(self, user: User, date: Date) -> np.ndarray:
        return user.feature_vector

    def _pull_user_expenses(self, user: User, date: Date) -> np.ndarray:
        total_expenses = len(Category)
        expenses = [0] * total_expenses

        for category, expense in user.timeline[date].expenses:
            expenses[category.id] = expense

        return np.ndarray(shape=(total_expenses, 1), buffer=expenses, dtype=np.float)


class AbstractExtractorAwarePredictor(AbstractPredictor, metaclass=ABCMeta):

    def __init__(self, extractor: ExtractorName):
        self._extractor = extractor

    def _pull_user_features(self, user: User, date: Date) -> np.ndarray:
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
    def extract(self, timestamp: TimeStamp, *, force=False) -> np.ndarray:
        pass
