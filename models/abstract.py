from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic, List, Iterable

import numpy as np

from data.datamodel import User, Date, Expenses, Category, Dataset, Serializable
from models.registered import ExtractorName


class AbstractTrainableModel(Serializable):

    def fit(self, data: Dataset) -> None:
        pass


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


class AbstractTrainablePredictor(AbstractPredictor, AbstractTrainableModel, metaclass=ABCMeta):
    pass


class AbstractExtractorAwarePredictor(AbstractPredictor, metaclass=ABCMeta):

    def __init__(self, extractor: ExtractorName):
        self._extractor = extractor

    def _pull_user_features(self, user: User, date: Date) -> np.array:
        return user.feature_vector_at(date, extractor=self._extractor)

    @property
    def extractor_name(self) -> ExtractorName:
        return self._extractor


_PredictorType = TypeVar('_PredictorType', bound=AbstractPredictor)


class AbstractExtractor(metaclass=ABCMeta):

    @abstractmethod
    def extract(self, user: User, *, force=False) -> User:
        pass

    def extract_users(self, users: List[User], *, force=False) -> List[User]:
        return [self.extract(user, force=force) for user in users]


class AbstractTrainableExtractor(AbstractExtractor, AbstractTrainableModel, metaclass=ABCMeta):
    pass


class AbstractNormalizer(metaclass=ABCMeta):

    @abstractmethod
    def normalize(self, user: User) -> User:
        pass

    def normalize_users(self, users: List[User]) -> List[User]:
        return [self.normalize(user) for user in users]


class AbstractTrainableNormalizer(AbstractNormalizer, AbstractTrainableModel, metaclass=ABCMeta):
    pass
