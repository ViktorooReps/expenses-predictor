from typing import Dict, Type

from data.datamodel import User
from models.abstract import AbstractNormalizer, AbstractTrainableNormalizer
from models.registered import NormalizerName


class StubNormalizer(AbstractNormalizer):

    def normalize(self, user: User) -> User:
        return user


name2normalizer: Dict[NormalizerName, Type[AbstractNormalizer]] = {
    NormalizerName.STUB: StubNormalizer
}

name2trainable_normalizer: Dict[NormalizerName, Type[AbstractTrainableNormalizer]] = {}
