import logging
from typing import Callable, List, Iterable, Optional

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette import status
from starlette.requests import Request
from starlette.responses import JSONResponse

from data.datamodel import Expenses, Date, User
from data.json_model import UserModel, ExpensesModel


logger = logging.Logger(__name__)


def register_request_validation_error_handler(app: FastAPI):
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(msg=str(exc.raw_errors))
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
        )


def register_user_predictor(app: FastAPI, endpoint: str, predictor: Callable[[Iterable[User], Optional[Iterable[Date]]], List[Expenses]]):

    @app.post(endpoint, response_model=List[ExpensesModel])
    async def predict(user_models: List[UserModel]):
        users = tuple(user_model.to_user() for user_model in user_models)

        response = list(ExpensesModel.from_expenses(exp) for exp in predictor(users, None))
        return response
