from typing import Optional, Dict, Type, Iterable, List

import numpy as np
from fastapi.encoders import jsonable_encoder
from pydantic import parse_obj_as
from requests.api import post

from data.datamodel import User, Date, Expenses, empty_expenses, Category, Value
from data.json_model import UserModel, ExpensesModel
from models.abstract import AbstractPredictor
from models.registered import PredictorName


class StubPredictor(AbstractPredictor):

    def predict(self, user: User) -> Expenses:
        return empty_expenses()


class RESTPredictor(AbstractPredictor):

    def __init__(self, server_url: str):
        self._server_url = server_url

    def predict_users(self, users: Iterable[User]) -> List[Expenses]:
        request_models = [UserModel.from_user(user) for user in users]
        request_json = jsonable_encoder(request_models)
        response = post(self._server_url, json=request_json)
        response.raise_for_status()

        return [em.to_expenses() for em in parse_obj_as(List[ExpensesModel], response.json())]

    def predict(self, user: User, date: Optional[Date] = None) -> Expenses:
        return self.predict_users([user])[0]


class BasePredictor(AbstractPredictor):

    def predict_users(self, users: Iterable[User]) -> List[Expenses]:
        all_expenses = []
        for user in users:
            user_expenses = np.zeros(len(Category), dtype=float)
            for month in user.timeline:
                user_expenses += super()._pull_user_expenses(user, month.date)
            user_expenses /= len(user.timeline)
            expenses = empty_expenses()
            for id in range(len(Category)):
                expenses[Category.from_id(id)] = user_expenses[id]
            all_expenses.append(expenses)
        return all_expenses

    def predict(self, user: User, date: Optional[Date] = None) -> Expenses:
        return self.predict_users([user])[0]


name2predictor: Dict[str, Type[AbstractPredictor]] = {
    PredictorName.STUB.value: StubPredictor,
    PredictorName.BASELINE.value: BasePredictor
}
