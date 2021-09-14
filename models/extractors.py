from typing import Dict, Type

import numpy as np

from data.datamodel import TimeStamp
from models.abstract import AbstractExtractor
from models.registered import ExtractorName


class StubExtractor(AbstractExtractor):

    def extract(self, timestamp: TimeStamp, *, force=False) -> np.ndarray:
        return np.ndarray(shape=(1, 1), buffer=timestamp.balance_change, dtype=np.float)


name2extractor: Dict[ExtractorName, Type[AbstractExtractor]] = {
    ExtractorName.STUB: StubExtractor
}
