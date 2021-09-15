import logging
from typing import Callable, List, Iterable

from fastapi import FastAPI


from data.datamodel import Expenses, User
from data.json_model import UserModel, ExpensesModel


logger = logging.Logger(__name__)


def register_users_predictor(app: FastAPI, endpoint: str, predictor: Callable[[Iterable[User]], List[Expenses]]):

    @app.post(endpoint, response_model=List[ExpensesModel])
    async def predict(user_models: List[UserModel]):
        users = tuple(user_model.to_user() for user_model in user_models)
        return predictor(users)
