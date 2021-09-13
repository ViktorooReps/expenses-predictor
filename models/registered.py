from enum import Enum
from typing import Dict

from models.abstract import AbstractExtractor, AbstractPredictor
from models.extractors import StubExtractor
from models.predictors import StubPredictor


class ExtractorName(Enum):
    STUB = 'stub'


class PredictorName(Enum):
    STUB = 'stub'


extractors: Dict[ExtractorName, AbstractExtractor] = {
    ExtractorName.STUB: StubExtractor()
}

predictors: Dict[PredictorName, AbstractPredictor] = {
    PredictorName.STUB: StubPredictor()
}
