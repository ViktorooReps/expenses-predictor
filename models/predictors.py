from typing import Optional, Dict, Type, Iterable, List

import numpy as np
from fastapi.encoders import jsonable_encoder
from pydantic import parse_obj_as
from requests.api import post

import torch
import torch.nn as nn
import torch.nn.functional as nn_funct

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


class SimpleModel(nn.Module):
    def __init__(self, length):
        self.alphas = nn.Parameter(torch.Tensor(length).uniform_(0, 1), requires_grad=True)

    def forward(self, user_expenses, user_pred_expenses):
        user_pred_expenses += self.alphas * (user_expenses - user_pred_expenses)
        return user_pred_expenses

def one_step(self, model: SimpleModel, loss_funct: nn.Module, user: User, optimizer: nn.Module = None):

    user_expenses = torch.Tensor(super()._pull_user_expenses(user, user.timeline[0].date))
    user_pred_expenses = torch.Tensor(super()._pull_user_expenses(user, user.timeline[0].date))

    general_loss = 0
    count = 0

    for month in user.timeline[1:]:
        if optimizer is not None:
            optimizer.zero_grad()

        user_pred_expenses = model(user_expenses, user_pred_expenses)
        user_expenses = torch.Tensor(super()._pull_user_expenses(user, month.date))

        loss = loss_funct(user_pred_expenses, user_expenses)  # (user_pred_expenses - user_expenses)^2

        general_loss += loss.sum().item()  # Q(alpha)
        count += loss.size(0)

        loss = loss.mean()

        if optimizer is not None:
            # upgrade_learning_rate(lr, optimizer)
            loss.backward(retain_graph=True)

            # Gradient clipping by predefined norm value - usually 5.0
            # torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return np.exp(general_loss / count)


class EMAPredictor(AbstractPredictor):

    def best_alphas(self, user: User) -> np.array:
        lr = 0.001
        epochs = 20

        loss_funct = nn_funct.mse_loss()  # create loss function
        model = SimpleModel(len(Category))  # init our model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # init optimizer method

        for i in range(epochs):
            model.train()  # turn on train mode with gradients
            train_perp = one_step(model, loss_funct, user, optimizer)
            print(f'Epoch: {i + 1} Train Perplexity: {train_perp:.3f}. ')

        return model.get_parameter('alphas').numpy()

    def predict_users(self, users: Iterable[User]) -> List[Expenses]
        all_expenses = []
        for user in users:
            alphas = self.best_alphas(user)
            user_pred_expenses = super()._pull_user_expenses(user, user.timeline[0].date)
            for month in user.timeline[1:]:
                user_pred_expenses += alphas * (super()._pull_user_expenses(user, month.date) - user_pred_expenses)
            expenses = empty_expenses()
            for id in range(len(Category)):
                expenses[Category.from_id(id)] = user_pred_expenses[id]
            all_expenses.append(expenses)
        return all_expenses

    def predict(self, user: User, date: Optional[Date] = None) -> Expenses:
        return self.predict_users([user])[0]


name2predictor: Dict[PredictorName, Type[AbstractPredictor]] = {
    PredictorName.STUB: StubPredictor,
    PredictorName.BASELINE: BasePredictor,
    PredictorName.EMA: EMAPredictor
}
