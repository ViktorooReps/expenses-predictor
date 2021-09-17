from typing import Dict, Type

import numpy as np

from data.datamodel import TimeStamp, User
from models.abstract import AbstractExtractor, AbstractTrainableExtractor
from models.registered import ExtractorName


class StubExtractor(AbstractExtractor):

    def extract(self, user: User, *, force=False) -> User:
        return user.with_changes(timeline=[self._extract(timestamp, force=force) for timestamp in user.timeline])

    @staticmethod
    def _extract(timestamp: TimeStamp, *, force=False) -> TimeStamp:
        if timestamp.has_feature_vector(ExtractorName.STUB) and not force:
            return timestamp
        else:
            return timestamp.with_changes(
                feature_vectors={**{ExtractorName.STUB: np.array([timestamp.balance_change], dtype=float)}, **timestamp.feature_vectors}
            )


name2extractor: Dict[ExtractorName, Type[AbstractExtractor]] = {
    ExtractorName.STUB: StubExtractor
}

name2trainable_extractor: Dict[ExtractorName, Type[AbstractTrainableExtractor]] = {}
