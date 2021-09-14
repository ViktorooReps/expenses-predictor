from enum import Enum
""" circular import
from typing import Dict

from models.abstract import AbstractExtractor, AbstractPredictor
from models.extractors import StubExtractor
from models.predictors import StubPredictor"""


class ExtractorName(Enum):
    STUB = 'stub'


class PredictorName(Enum):
    STUB = 'stub'


""" circular import
extractors: Dict[ExtractorName, AbstractExtractor] = {
    ExtractorName.STUB: StubExtractor()
}

predictors: Dict[PredictorName, AbstractPredictor] = {
    PredictorName.STUB: StubPredictor()
}"""
