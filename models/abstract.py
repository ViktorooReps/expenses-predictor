from abc import ABCMeta, abstractmethod
from typing import Optional

from data.datamodel import User, Value, Date


class AbstractPredictor(metaclass=ABCMeta):

    @abstractmethod
    def predict(self, user: User, timestamp: Optional[Date] = None) -> Value:
        pass
