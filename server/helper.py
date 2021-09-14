from typing import Callable, List, Iterable

from fastapi import FastAPI

from data.datamodel import Expenses, Date, User
from data.json_model import UserModel, ExpensesModel


def register_user_predictor(app: FastAPI, endpoint: str, predictor: Callable[[Iterable[User], Iterable[Date]], List[Expenses]]):

    @app.post(endpoint, response_model=List[ExpensesModel])
    async def predict(user_models: List[UserModel], dates: List[Date]):
        users = tuple(user_model.to_user() for user_model in user_models)

        response = list(ExpensesModel.from_expenses(exp) for exp in predictor(users, dates))
        return response
