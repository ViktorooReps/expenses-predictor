from typing import Optional

from data.datamodel import User, Date, Expenses, empty_expenses
from models.abstract import AbstractPredictor


class StubPredictor(AbstractPredictor):

    def predict(self, user: User, date: Optional[Date] = None) -> Expenses:
        return empty_expenses()
