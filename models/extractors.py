import numpy as np

from data.datamodel import TimeStamp
from models.abstract import AbstractExtractor


class StubExtractor(AbstractExtractor):

    def extract(self, timestamp: TimeStamp, *, force=False) -> np.ndarray:
        return np.ndarray(shape=(1, 1), buffer=timestamp.balance_change, dtype=np.float)
