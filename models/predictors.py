from typing import Optional, Dict, Type

from data.datamodel import User, Date, Expenses, empty_expenses
from models.abstract import AbstractPredictor
from models.registered import PredictorName


class StubPredictor(AbstractPredictor):

    def predict(self, user: User, date: Optional[Date] = None) -> Expenses:
        return empty_expenses()


name2predictor: Dict[PredictorName, Type[AbstractPredictor]] = {
    PredictorName.STUB: StubPredictor
}
